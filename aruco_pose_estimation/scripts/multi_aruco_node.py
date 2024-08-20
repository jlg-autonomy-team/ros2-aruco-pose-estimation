#!/usr/bin/env python3

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
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from cv_bridge import CvBridge
import message_filters
from rclpy.time import Time

# Python imports
import numpy as np
import cv2
import pyrealsense2 as rs

# Local imports for custom defined functions
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import pose_estimation

# ROS2 message imports
import rclpy.time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import TransformBroadcaster, TransformStamped
import tf2_ros
import tf2_geometry_msgs
import yaml
import os
from ament_index_python.packages import get_package_share_directory

import tf_transformations as tft


def mean_orientation(self, quaternions):
    # Convert quaternions to 4x4 transformation matrices
    rot_mats = [tft.quaternion_matrix(q)[:3, :3] for q in quaternions]

    # Compute the average of the rotation matrices
    avg_rot_mat = np.mean(rot_mats, axis=0)

    # Ensure the average matrix is orthogonal using SVD
    # U, _, Vt = np.linalg.svd(avg_rot_mat)
    # avg_rot_mat = np.dot(U, Vt)

    # Convert the average rotation matrix back to a quaternion
    avg_transformation_matrix = np.eye(4)
    avg_transformation_matrix[:3, :3] = avg_rot_mat
    mean_quaternion = tft.quaternion_from_matrix(avg_transformation_matrix)

    # Return the quaternion as a list of 4 values
    return list(mean_quaternion)  # Ensure the result is a list


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.initialize_parameters()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)

            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError

        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(self.dictionary_id_name)
            )
            options = "\n".join([s for s in ARUCO_DICT])
            self.get_logger().error("valid options: {}".format(options))

        # camera info topic for the camera calibration parameters
        self.info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self.info_callback, qos_profile_system_default
        )

        # select the type of input to use for the pose estimation
        if bool(self.use_depth_input):
            # use both rgb and depth image topics for the pose estimation

            # create a message filter to synchronize the image and depth image topics
            self.image_sub = message_filters.Subscriber(
                self,
                Image,
                self.image_topic,
                qos_profile=qos_profile_system_default,  # qos_profile_sensor_data
            )
            self.depth_image_sub = message_filters.Subscriber(
                self,
                Image,
                self.depth_image_topic,
                qos_profile=qos_profile_system_default,  # qos_profile_sensor_data
            )

            # create synchronizer between the 2 topics using message filters and approximate time policy
            # slop is the maximum time difference between messages that are considered synchronized
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_image_sub], queue_size=10, slop=0.05
            )
            self.synchronizer.registerCallback(self.rgb_depth_sync_callback)

        else:
            # rely only on the rgb image topic for the pose estimation

            # create a subscription to the image topic
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_profile=qos_profile_system_default,
            )

        # Set up publishers
        self.poses_pub = self.create_publisher(
            PoseArray, self.markers_visualization_topic, 10
        )

        self.markers_pub = self.create_publisher(
            ArucoMarkers, self.detected_markers_topic, 10
        )

        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        # Set up fields for camera parameters
        self.info_msg: CameraInfo = None
        self.intrinsic_mat = None
        self.distortion = None

        # code for updated version of cv2 (4.7.0)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dictionary, self.aruco_parameters
        )

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

        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

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

        # call the pose estimation function
        pose_array: PoseArray

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

        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)
            first_pose = Pose()
            first_pose = pose_array.poses[0]

            ##############ANUJ##################
            try:
                target_transform = self.tfBuffer.lookup_transform(
                    "camera_color_frame",
                    "camera_color_optical_frame",
                    rclpy.time.Time(seconds=(0.0)),
                )
                first_pose = tf2_geometry_msgs.do_transform_pose(
                    first_pose, target_transform
                )

            except Exception as e:
                print(e)

            # Assuming only 1 QR Code at a time
            t = TransformStamped()

            t.header.stamp = img_msg.header.stamp
            t.header.frame_id = "camera_color_frame"
            t.child_frame_id = "marker"

            print(
                "Marker Location is x: {0}, y: {1}, z: {2}".format(
                    first_pose.position.x, first_pose.position.y, first_pose.position.z
                )
            )

            print(
                "Marker Orientation is w: {0}, x: {1}, y: {2}, z:{3}".format(
                    first_pose.orientation.w,
                    first_pose.orientation.x,
                    first_pose.orientation.y,
                    first_pose.orientation.z,
                )
            )

            t.transform.translation.x = first_pose.position.x
            t.transform.translation.y = first_pose.position.y
            t.transform.translation.z = first_pose.position.z
            t.transform.rotation.z = first_pose.orientation.z
            t.transform.rotation.y = first_pose.orientation.y
            t.transform.rotation.x = first_pose.orientation.x
            t.transform.rotation.w = first_pose.orientation.w

            self.tf_broadcaster.sendTransform(t)

            t_goal = TransformStamped()

            # Define the transform of the charging station with respect to the Aruco marker
            t_goal.header.stamp = img_msg.header.stamp
            t_goal.header.frame_id = "marker"
            t_goal.child_frame_id = "station"

            t_goal.transform.translation.x = (
                -0.0
            )  # config["marker_to_station_offset"]["pos_x"]
            t_goal.transform.translation.y = (
                1.5
                - 0.43085  # config["marker_to_station_offset"]["pos_y"] - (1 - 0.56915(camera frame height))
            )
            t_goal.transform.translation.z = 2.5  # (0.7 + 1.8 = length of robot)
            # config["marker_to_station_offset"]["pos_z"]

            t_goal.transform.rotation.x = (
                0.5  # config["marker_to_station_offset"]["ori_x"]
            )
            t_goal.transform.rotation.y = (
                0.5  # config["marker_to_station_offset"]["ori_y"]
            )
            t_goal.transform.rotation.z = (
                -0.5
            )  # config["marker_to_station_offset"]["ori_z"]
            t_goal.transform.rotation.w = (
                0.5  # config["marker_to_station_offset"]["ori_w"]
            )

            self.tf_broadcaster.sendTransform(t_goal)

            t_good_vis = TransformStamped()

            # Define the transform of the charging station with respect to the Aruco marker
            t_good_vis.header.stamp = img_msg.header.stamp
            t_good_vis.header.frame_id = "station"
            t_good_vis.child_frame_id = "vista_point"

            t_good_vis.transform.translation.x = (
                -1.5
            )  # config["station_to_vista_offset"]["pos_x"]
            t_good_vis.transform.translation.y = (
                -0.0
            )  # config["station_to_vista_offset"]["pos_y"]
            t_good_vis.transform.translation.z = (
                0.0  # config["station_to_vista_offset"]["pos_z"]
            )

            t_good_vis.transform.rotation.x = (
                0.0  # config["station_to_vista_offset"]["ori_x"]
            )
            t_good_vis.transform.rotation.y = (
                0.0  # config["station_to_vista_offset"]["ori_y"]
            )
            t_good_vis.transform.rotation.z = (
                0.0  # config["station_to_vista_offset"]["ori_z"]
            )
            t_good_vis.transform.rotation.w = (
                1.0  # config["station_to_vista_offset"]["ori_w"]
            )

            self.tf_broadcaster.sendTransform(t_good_vis)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def depth_image_callback(self, depth_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

    def rgb_depth_sync_callback(self, rgb_msg: Image, depth_msg: Image):

        # convert the image messages to cv2 format
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        ############## ANUUJ ####################
        clipping_distance = 1
        depth_scale = 0.001
        depth_image = np.asanyarray(cv_depth_image)
        color_image = np.asanyarray(cv_image)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((bg_removed, depth_colormap))

        ################# ############

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

        # Read parameters from aruco_params.yaml and store them
        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )

        self.get_logger().info(f"Marker size: {self.marker_size}")

        self.dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )

        self.get_logger().info(f"Marker type: {self.dictionary_id_name}")

        self.use_depth_input = (
            self.get_parameter("use_depth_input").get_parameter_value().bool_value
        )
        self.get_logger().info(f"Use depth input: {self.use_depth_input}")

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Input image topic: {self.image_topic}")

        self.depth_image_topic = (
            self.get_parameter("depth_image_topic").get_parameter_value().string_value
        )

        self.get_logger().info(f"Input depth image topic: {self.depth_image_topic}")

        self.info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image camera info topic: {self.info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
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
            self.get_parameter("output_image_topic").get_parameter_value().string_value
        )


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
