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
import time


## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "cv_recognizer"

# 발행 토픽 이름 (3개 입력 / 선로 1, 선로 2, 신호등)
TOPIC_NAME = ["cv_lane_2", "cv_tl", "None"]

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


class cv_recognizer(Node):
    def __init__(self, node_name, topic_name : list, sub_topic_name, pt_name, debug, label_name):
        super().__init__(node_name)

        self.model = YOLO(os.path.dirname(__file__) + "/" + pt_name) # YOLO Model 선언

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
        
        # 이름 변수 선언
        self.topic_0 = topic_name[0]
        self.topic_1 = topic_name[1]
        self.topic_2 = topic_name[2]

        # 라벨 이름 변수 선언
        self.label_0 = label_name[0]
        self.label_1 = label_name[1]
        self.label_2 = label_name[2]

        # 디버그 변수 선언
        self.debug = debug

        # Publisher 선언
        self.publisher_0 = self.create_publisher(Float32MultiArray, self.topic_0, self.qos_pub) 
        self.publisher_1 = self.create_publisher(Float32MultiArray, self.topic_1, self.qos_pub) 
        self.publisher_2 = self.create_publisher(Float32MultiArray, self.topic_2, self.qos_pub)

        # Publishing을 위한 Message 선언
        self.msg_0 = Float32MultiArray()
        self.msg_1 = Float32MultiArray()
        self.msg_2 = Float32MultiArray()

        # Subscriber 선언
        self.subscriber = self.create_subscription(Image, sub_topic_name, self.recognizer_callback, self.qos_sub) # Topic 구독
        self.bridge = cv_bridge.CvBridge() # CV Bridge Object 선언

    def recognizer_callback(self, img_msg):
        self.frame = self.bridge.imgmsg_to_cv2(img_msg) # Frame 수령 및 처리
        self.predicted = self.model.predict(self.frame, verbose=False) # Frame Segmentation 처리

        if self.debug == True: # 디버깅(화면 출력) 여부 결정
            cv2.imshow("CV", self.predicted[0].plot())
            cv2.waitKey(5)

        # 카운트를 위한 변수 선언
        cnt_0 = 0
        cnt_1 = 0
        cnt_2 = 0

        # 확률에 대한 내림차순 정렬
        predict = self.predicted[0].boxes[torch.argsort(self.predicted[0].boxes.conf, descending=True)]
        self.get_logger().info(f"{len(predict.conf.tolist())} object(s) detected | value = {predict.conf.tolist()}")

        # Box : 상자 / Keypoint : 관절 표현 / Mask : 영역 표시
        for n, predict in enumerate(predict):
            name = self.predicted[0].names[int(predict.cls.item())].strip()

            if  name == self.label_0.strip() and cnt_0 == 0:
                self.msg_0.data = self.predicted[0].masks[n].xy[0].flatten().tolist()
                self.publisher_0.publish(self.msg_0)
                self.get_logger().info(f"{self.topic_0} is published | count = [{cnt_0}]")
                cnt_0 += 1
            
            elif name == self.label_1.strip() and cnt_1 == 0:
                self.msg_1.data = self.msg_2.data = self.predicted[0].boxes[n].xyxy[0].flatten().tolist()      
                self.publisher_1.publish(self.msg_1)
                self.get_logger().info(f"{self.topic_1} is published | count = [{cnt_1}]")
                cnt_1 += 1

            elif name == self.label_2.strip() and cnt_2 == 0:
                self.msg_2.data = self.predicted[0].masks[n].xy[0].flatten().tolist()
                self.publisher_2.publish(self.msg_2)
                self.get_logger().info(f"{self.topic_2} is published | count = [{cnt_2}]")  
                cnt_2 += 1          

                # Lane 전용
                # self.msg_2.data = self.predicted[0].masks[n].xy[0].flatten().tolist() 

                # Traffic Light 전용
                # self.msg_2.data = self.predicted[0].boxes[n].xyxy[0]             

    def shutdown(self):
        cv2.destroyAllWindows() # CV 창 닫기


def main():
    rclpy.init()
    cv_recognizer_node = cv_recognizer(NODE_NAME, TOPIC_NAME, SUB_TOPIC_NAME, PT_NAME, DEBUG, LABEL_NAME)
    rclpy.spin(cv_recognizer_node)

    cv_recognizer_node.shutdown()
    cv_recognizer_node.destroy_node()
    rclpy.shutdown()


if __name__== "__main__":
    main() 