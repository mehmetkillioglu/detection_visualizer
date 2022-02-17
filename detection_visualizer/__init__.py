# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DetectionVisualizerNode(Node):

    def __init__(self):
        super().__init__('detection_visualizer')

        self._bridge = cv_bridge.CvBridge()

        output_image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1)

        self._image_pub = self.create_publisher(Image, '~/dbg_images', output_image_qos)

        self._image_sub = self.create_subscription(Image, '~/images', self.on_camera, 10)
        self._detections_sub = self.create_subscription(Detection2DArray, '~/detections', self.on_detection, 10)

        self._detection = Detection2DArray()
        print("Subscribed to detections")

    def on_camera(self, image_msg):
        print("Received image")
        cv_image = self._bridge.imgmsg_to_cv2(image_msg)
        
        print("Have {} detections".format(len(self._detection.detections)))
        # Draw boxes on image
        for detection in self._detection.detections:
            max_class = None
            max_score = 0.0
            for hypothesis in detection.results:
                if hypothesis.hypothesis.score > max_score:
                    max_score = hypothesis.hypothesis.score
                    max_class = hypothesis.hypothesis.class_id
            if max_class is None:
                print("Failed to find class with highest score", file=sys.stderr)
                return

            cx = detection.bbox.center.x
            cy = detection.bbox.center.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            color = (0, 255, 0)
            thickness = 1
            cv2.rectangle(cv_image, min_pt, max_pt, color, thickness)

            label = '{} {} {:.3f}'.format(detection.id, max_class, max_score)
            pos = (min_pt[0], max_pt[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font, 0.75, color, 1, cv2.LINE_AA)
            
        detection_image_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        detection_image_msg.header = image_msg.header

        self._image_pub.publish(detection_image_msg)
        
    def on_detection(self, detections_msg):
        self._detection = detections_msg
        
    def on_detections(self, image_msg, detections_msg):
        print("Received detections")
        cv_image = self._bridge.imgmsg_to_cv2(image_msg)

        # Draw boxes on image
        for detection in detections_msg.detections:
            max_class = None
            max_score = 0.0
            for hypothesis in detection.results:
                if hypothesis.hypothesis.score > max_score:
                    max_score = hypothesis.hypothesis.score
                    max_class = hypothesis.hypothesis.class_id
            if max_class is None:
                print("Failed to find class with highest score", file=sys.stderr)
                return

            cx = detection.bbox.center.x
            cy = detection.bbox.center.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            color = (0, 255, 0)
            thickness = 1
            cv2.rectangle(cv_image, min_pt, max_pt, color, thickness)

            label = '{} {:.3f}'.format(max_class, max_score)
            pos = (min_pt[0], max_pt[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font, 0.75, color, 1, cv2.LINE_AA)
            
        detection_image_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        detection_image_msg.header = image_msg.header

        self._image_pub.publish(detection_image_msg)


def main():
    rclpy.init()
    rclpy.spin(DetectionVisualizerNode())
    rclpy.shutdown()
