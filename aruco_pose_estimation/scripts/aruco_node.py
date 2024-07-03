#!/usr/bin/env python3
"""
Modified by: Anuj Pai Raikar - for JLG Industries

Change log:

June 11, 2024
1. Changed topic subscriptions arguments FROM qos_profile_sensor_data TO qos_profile_system_default to be able to log the aruco topics
2. ADDED OpenCV resizing in case of RGB+Depth input - values such that RGB image dimensions are equated to match those of Depth image
3. Logging Statements : DEBUGGING have been commented

June 12, 2024
4. ADDED code for alignment of frames as a potential replacement for Change #2.: To prevent cv2.resize downsampling information loss 
[Assumption is that; Frames synchronize and depth+RGB resolution given from the launch file for realsense]

June 14, 2024
5. Added code for ground truth marker data publishing and viz. - using ZeroKey topic for GT

June 17, 2024
6. Had to comment it - incorrect remapping

June 21, 2024
7. Adding Error Logging Capability 

June 25, 2024
8. Added Bounding Box Size display
"""

"""
ROS2 wrapper code taken from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main

This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

    /aruco_image (sensor_msgs.msg.Image)
       Annotated image with marker locations and ids, with markers drawn on it

Parameters:
    marker_size - size of the markers in meters (default .065)
    aruco_dictionary_id - dictionary that was used to generate markers (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera_info_topic - camera info topic to subscribe to (default /camera/camera_info)
    camera_frame - camera optical frame to use (default "camera_depth_optical_frame")
    detected_markers_topic - topic to publish detected markers (default /aruco_markers)
    markers_visualization_topic - topic to publish markers visualization (default /aruco_poses)
    output_image_topic - topic to publish annotated image (default /aruco_image)

Author: Simone Giampà
Version: 2024-01-29

"""

# ROS2 imports
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data,qos_profile_system_default
from cv_bridge import CvBridge
import message_filters

# Python imports
import numpy as np
import cv2
import pyrealsense2 as rs

# Local imports for custom defined functions
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import pose_estimation


# ROS2 message imports
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, PoseStamped, Vector3Stamped
from aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import Float32MultiArray
import math
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import time

class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.initialize_parameters()

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)

            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
            
        except AttributeError:
            self.get_logger().error("bad aruco_dictionary_id: {}".format(self.dictionary_id_name))
            options = "\n".join([s for s in ARUCO_DICT])
            self.get_logger().error("valid options: {}".format(options))


        ## ADDED : DEBUGGING
        # self.get_logger().info("Parameters Initialized - Dictionary Valid")
        ##

        # Set up subscriptions to the camera info and camera image topics

        # camera info topic for the camera calibration parameters
        self.info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self.info_callback, qos_profile_system_default
        )

        
        # select the type of input to use for the pose estimation
        if bool(self.use_depth_input):
            # use both rgb and depth image topics for the pose estimation

            # create a message filter to synchronize the image and depth image topics
            self.image_sub = message_filters.Subscriber(
                self, Image, self.image_topic, qos_profile=qos_profile_system_default # qos_profile_sensor_data
            )
            self.depth_image_sub = message_filters.Subscriber(
                self, Image, self.depth_image_topic, qos_profile=qos_profile_system_default # qos_profile_sensor_data
            )
            
            ## ADDED : DEBUGGING
            # self.get_logger().info("Depth Image Subscription Initiated")
            ##

            # create synchronizer between the 2 topics using message filters and approximate time policy
            # slop is the maximum time difference between messages that are considered synchronized
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_image_sub], queue_size=10, slop=0.05
            )
            self.synchronizer.registerCallback(self.rgb_depth_sync_callback)


            # # ####### ADDED : Ground Truth

            # self.gt_pose_sub = message_filters.Subscriber(
            #     self, Vector3Stamped, self.gt_pose_topic, self.error_callback, qos_profile=qos_profile_system_default
            # )
            # # #######

        else:
            # rely only on the rgb image topic for the pose estimation

            ## ADDED : DEBUGGING
            self.get_logger().info("Regular Image Subscription Initiated")
            ##

            # create a subscription to the image topic
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback, qos_profile=qos_profile_system_default
            )

            # ###### ADDED : Ground Truth
            # self.gt_pose_sub = message_filters.Subscriber(
            #     Vector3Stamped, self.gt_pose_topic, self.error_callback, qos_profile=qos_profile_system_default
            # )
            # # ######

        ## ADDED : DEBUGGING
        # self.get_logger().info("About to publish the poses")
        ##

        # Set up publishers
        self.poses_pub = self.create_publisher(
            PoseArray, self.markers_visualization_topic, 10
        )

        self.markers_pub = self.create_publisher(
            ArucoMarkers, self.detected_markers_topic, 10
        )

        self.image_pub = self.create_publisher(
            Image, self.output_image_topic, 10
        )


        # # ###### ADDED: Ground Truth
        # #Bounding Box dimensions
        # self.bbdim_pub =self.create_publisher(
        #     Float32MultiArray, self.bb_dim_topic, 10
        # )

        # # self.error_pub = self.create_publisher(
        # #     Vector3Stamped, self.error_topic, 10
        # # )
        
        # # ######

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        # code for updated version of cv2 (4.7.0)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, 
                                                      self.aruco_parameters)

        # old code version
        # self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        # self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()

    def info_callback(self, info_msg):

        self.info_msg = info_msg
        # get the intrinsic matrix and distortion coefficients from the camera info
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

        self.get_logger().info("Camera info received.")
        self.get_logger().info("Intrinsic matrix: {}".format(self.intrinsic_mat))
        self.get_logger().info("Distortion coefficients: {}".format(self.distortion))
        self.get_logger().info(
            "Camera frame: {}x{}".format(self.info_msg.width, self.info_msg.height)
        )

        # ADDED : DEBUGGING
        # self.get_logger().info("Destroying camera info subscription")
        ##

        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return
        
        ## ADDED : DEBUGGING
        # self.get_logger().info("Image Callback Initiated")
        ##

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id

        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        """
        # OVERRIDE: use calibrated intrinsic matrix and distortion coefficients
        self.intrinsic_mat = np.reshape([615.95431, 0., 325.26983,
                                         0., 617.92586, 257.57722,
                                         0., 0., 1.], (3, 3))
        self.distortion = np.array([0.142588, -0.311967, 0.003950, -0.006346, 0.000000])
        """

        # ###### ADDED: Ground Truth
        # frame, bbh, bbw, pose_array, markers = pose_estimation(
        # ######
        
        # call the pose estimation function
        frame, pose_array, markers = pose_estimation(
            rgb_frame=cv_image,
            depth_frame=None,
            aruco_detector=self.aruco_detector,
            marker_size=self.marker_size,
            matrix_coefficients=self.intrinsic_mat,
            distortion_coefficients=self.distortion,
            pose_array=pose_array,
            markers=markers,
        )

        # ####### ADDED: Ground Truth
        # bbdim = Float32MultiArray
        # bbdim.data = [bbh, bbw]
        # ####### 

        ## ADDED : DEBUGGING
        # print("finished pose estimation")
        # print(markers)
        ##

        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # ####### ADDED: Ground Truth
        # self.bbdim_pub.publish(bbdim)
        # ######

        ## ADDED : DEBUGGING
        # print("publish aruco image")
        ##
    
    def depth_image_callback(self, depth_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

    def rgb_depth_sync_callback(self, rgb_msg: Image, depth_msg: Image):

        # convert the image messages to cv2 format
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        #### ADDED: Matching Shape of Depth and Color
        # h, w, *_ = cv_depth_image.shape
        # # print(h, w)
        # cv_image = cv2.resize(cv_image, (w, h))

        # cv_image = cv2.resize(cv_image, (848, 480))
        # OR

        ####

        ############## ADDING CODE FOR ALIGNMENT
        clipping_distance = 1
        depth_scale = 0.001
        depth_image = np.asanyarray(cv_depth_image)
        color_image = np.asanyarray(cv_image)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        #############

        # Create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
            
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = rgb_msg.header.stamp
        pose_array.header.stamp = rgb_msg.header.stamp

        # call the pose estimation function

        # ###### Ground Truth
        # frame, bb_height, bb_width, pose_array, markers = pose_estimation(
        # ###### Ground Truth

        frame, pose_array, markers = pose_estimation(
            rgb_frame=cv_image,
            depth_frame=cv_depth_image,
            aruco_detector=self.aruco_detector,
            marker_size=self.marker_size,
            matrix_coefficients=self.intrinsic_mat,
            distortion_coefficients=self.distortion,
            pose_array=pose_array,
            markers=markers,
        )


        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # ####### ADDED: Ground Truth
        # self.curr_pose = pose_array
        # #######

    # ####### ADDED: Ground Truth
    # # Error Computation Callback
    # def error_callback(self, gt_pose_msg: Vector3Stamped):  #gt_pose_msg -> Vector3Stamped
    #     if self.gt_pose_msg is None:
    #         self.get_logger().warn("No GROUND TRUTH INFO has been received!")
    #         return
        
    #     ## ADDED : DEBUGGING
    #     # self.get_logger().info("Image Callback Initiated")
    #     ##
    #     self.gt_pose_msg = gt_pose_msg
    #     curr_position = self.curr_pose.position
    #     gt_xy_yaw = self.gt_pose_msg
        
    #     error = Vector3Stamped()
    #     error.header.stamp = self.curr_pose.header.stamp
    #     error.header.frame_id = "Current Error"

    #     # tolerance_m = 0.015 ################## If needed
    #     # 1 deg

    #     error.x = curr_position.x - gt_xy_yaw.x
    #     error.y = curr_position.y - gt_xy_yaw.y

    #     curr_orientation = self.curr_pose.orientation

    #     (_, _, curr_yaw) = euler_from_quaternion(
    #         [
    #             curr_orientation.x,
    #             curr_orientation.y,
    #             curr_orientation.z,
    #             curr_orientation.w,
    #         ]
    #     )

    #     yaw_error_rad = math.fabs(curr_yaw - gt_xy_yaw.yaw)
    #     error.z = yaw_error_rad

    #     self.error_pub.publish(error)
        
                 
    #     return error

    # #######

    def initialize_parameters(self):
        # Declare and read parameters from aruco_params.yaml

        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="use_depth_input",
            value=True,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Use depth camera input for pose estimation instead of RGB image",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="depth_image_topic",
            value="/camera/depth/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Depth camera topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.declare_parameter(
            name="detected_markers_topic",
            value="/aruco_markers",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish detected markers as array of marker ids and poses",
            ),
        )

        self.declare_parameter(
            name="markers_visualization_topic",
            value="/aruco_poses",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish markers as pose array",
            ),
        )

        self.declare_parameter(
            name="output_image_topic",
            value="/aruco_image",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish annotated images with markers drawn on them",
            ),
        )

        # ##### ADDED: Ground Truth

        # self.declare_parameter(
        #     name= "gt_pose_topic",
        #     value="/zk/gt_xy_yaw",
        #     descriptor=ParameterDescriptor(
        #         type=ParameterType.PARAMETER_DOUBLE_ARRAY,
        #         description="Topic to obtain the ground truth pose of ArUco marker in terms of x, y and yaw",
        #     ),
        # )

        # self.declare_parameter(
        #     name= "error_topic",
        #     value="/error",
        #     descriptor=ParameterDescriptor(
        #     type=ParameterType.PARAMETER_DOUBLE_ARRAY,
        #     description="Topic to publish the error pose of ArUco marker in terms of x, y and yaw",
        #     ),
        # )
        # ######

        # Read parameters from aruco_params.yaml and store them
        self.marker_size = (
            self.get_parameter("marker_size")
            .get_parameter_value()
            .double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        self.dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id")
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"Marker type: {self.dictionary_id_name}")

        self.use_depth_input = (
            self.get_parameter("use_depth_input")
            .get_parameter_value()
            .bool_value
        )
        self.get_logger().info(f"Use depth input: {self.use_depth_input}")

        self.image_topic = (
            self.get_parameter("image_topic")
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"Input image topic: {self.image_topic}")

        self.depth_image_topic = (
            self.get_parameter("depth_image_topic")
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"Input depth image topic: {self.depth_image_topic}")

        self.info_topic = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"Image camera info topic: {self.info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame")
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(f"Camera frame: {self.camera_frame}")

        # Output topics
        self.detected_markers_topic = (
            self.get_parameter("detected_markers_topic")
            .get_parameter_value()
            .string_value
        )

        self.markers_visualization_topic = (
            self.get_parameter("markers_visualization_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_image_topic = (
            self.get_parameter("output_image_topic")
            .get_parameter_value()
            .string_value
        )

        # # ###### ADDED : Ground Truth

        # self.bb_dim_topic = (
        #     self.get_parameter("bounding_box_dimensions topic")
        #     .get_parameter_value()
        #     .string_value
        # )

        # self.gt_pose_topic = (
        #     self.get_parameter("gt_pose_topic")
        #     .get_parameter_value()
        #     .string_value
        # )
        # self.get_logger().info(f"Image camera info topic: {self.gt_pose_topic}")

        # self.gt_pose_markers_topic = (
        #     self.get_parameters("ground_truth_markers_topic")
        #     .get_parameter_value()
        #     .string_value
        # )

        # self.gt_pose_markers_visualization_topic = (
        #     self.get_parameters("ground_truth_markers visualization_topic")
        #     .get_parameter_value()
        #     .string_value
        # )

        # self.error_topic = (
        #     self.get_parameters("error_x_y_yaw_topic")
        #     .get_parameter_value()
        #     .string_value
        # )

        # # ###### 


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
