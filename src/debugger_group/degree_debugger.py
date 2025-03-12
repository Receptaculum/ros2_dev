### 시험용 ###
# path_predictor에서 발생한 각도 정보를 그래프로 출력

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import cv_bridge

import torch
import numpy as np

import matplotlib.pyplot as plt

import pickle

## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "degree_debugger"

# 구독 토픽 이름
SUB_TOPIC_NAME = "path_angle"

######################################################################################################

class degree_debugger(Node):
    def __init__(self, node_name, sub_topic_name):
        super().__init__(node_name)

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        self.plt_data = []

        self.subscriber = self.create_subscription(Int8, sub_topic_name, self.write_data_callback, self.qos_sub)

    def write_data_callback(self, msg):
        self.plt_data.append(msg.data)

        with open("deg.pickle", "wb") as f:
            pickle.dump(self.plt_data, f)
            self.get_logger().info("data saved in deg.pickle")
        
def main():
    rclpy.init()
    degree_debugger_node = degree_debugger(NODE_NAME, SUB_TOPIC_NAME)
    rclpy.spin(degree_debugger_node)

    degree_debugger_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
