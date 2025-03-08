import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import ultralytics
from ultralytics import YOLO

import cv2
import cv_bridge

import torch

import os


## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "path_predictor"

# 토픽 이름 (3개 입력 / 선로 1, 선로 2, 신호등)
TOPIC_NAME = ["cv_lane2", "cv_tf", "None"]

# 라벨 이름
LABEL_NAME = ["lane2", "traffic_light", "None"]

# 구독 토픽 이름
SUB_TOPIC_NAME = "camera_publisher"

# PT 파일 이름 지정 (확장자 포함)
PT_NAME = "car.pt"

# CV 처리 영상 출력 여부
DEBUG = True

######################################################################################################


## <로그 출력> #########################################################################################
# DEBUG	self.get_logger().debug("msg")
# INFO	self.get_logger().info("msg")
# WARN	self.get_logger().warn("msg")
# ERROR	self.get_logger().error("msg")
# FATAL	self.get_logger().fatal("msg")
#######################################################################################################


## <QOS> ##############################################################################################
# Reliability : RELIABLE(신뢰성 보장), BEST_EFFORT(손실 감수, 최대한 빠른 전송)
# Durability : VOLATILE(전달한 메시지 제거), TRANSIENT_LOCAL(전달한 메시지 유지) / (Subscriber가 없을 때에 한함)
# History : KEEP_LAST(depth 만큼의 메시지 유지), KEEP_ALL(모두 유지)
# Liveliness : 활성 상태 감시
# Deadline : 최소 동작 보장 
#######################################################################################################