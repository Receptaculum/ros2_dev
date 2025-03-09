### 시험용 ###
# cv_recognizer node에서 발행한 topic 데이터 점검을 위한 코드

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import cv_bridge

import torch
import numpy as np


## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "path_predictor"

# 토픽 이름
TOPIC_NAME = None

# 구독 토픽 이름
SUB_TOPIC_NAME = ["cv_lane_2", "cv_tl"]

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# CV 처리 영상 출력 여부
DEBUG = True

######################################################################################################


class path_preictor(Node):
    def __init__(self, node_name, topic_name : list, frame_size : list):
        super().__init__(node_name)

        self.qos_pub = QoSProfile( # Publisher QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.frame_size = frame_size
        
        # Subscriber 선언
        self.subscriber_lane = self.create_subscription(Float32MultiArray, topic_name[0], self.lane_callback, self.qos_sub)
        self.subscriber_traffic = self.create_subscription(Float32MultiArray, topic_name[1], self.traffic_callback, self.qos_sub)

    def lane_callback(self, msg):
        img_background = np.zeros([self.frame_size[1], self.frame_size[0]])

        img = cv2.polylines(img_background, [np.array(msg.data, dtype=int).reshape(-1, 2)], True, (255, ), 1)
        cv2.imshow("lane", img)
        cv2.waitKey(5)
        print(np.array(msg.data, dtype=int).reshape(-1, 2))

    def traffic_callback(self, msg):
        img_background = np.zeros([self.frame_size[1], self.frame_size[0]])

        img = cv2.polylines(img_background, [np.array(msg.data, dtype=int).reshape(-1, 2)], True, (255, ), 1)
        cv2.imshow("tf", img)
        cv2.waitKey(5)


def main():
    rclpy.init()
    path_preictor_node = path_preictor(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE)
    rclpy.spin(path_preictor_node)

    rclpy.shutdown()

if __name__== "__main__":
    main()