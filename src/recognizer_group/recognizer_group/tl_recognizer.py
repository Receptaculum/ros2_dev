import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

import ultralytics
from ultralytics import YOLO

import cv2
import cv_bridge

import torch
import numpy as np

import os


## < Parameter > #####################################################################################

# 노드 이름
NODE_NAME = "tl_recognizer"

# 발행 토픽 이름
TOPIC_NAME = "tl_recognizer"

# 구독 토픽 이름 | 신호등 위치, 영상
SUB_TOPIC_NAME = ["cv_tl", "camera_publisher"]

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# CV 처리 영상 출력 여부
DEBUG = True

######################################################################################################

## <색상 코드> #########################################################################################
# HSV 기준 (0~179, 0~255, 0~255)
R_RANGE = [[[0, 20], [160, 179]], [[100, 255]], [[100, 255]]]
Y_RANGE = [[[25, 40]], [[100, 255]], [[100, 255]]]
G_RANGE = [[[45, 85]], [[8, 255]], [[100, 255]]]


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


class tl_recognizer(Node):
    def __init__(self, node_name, sub_topic_name, frame_size, debug):
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
        
        self.debug = debug

        self.subscriber_tl_location = self.create_subscription(Float32MultiArray, sub_topic_name[0], self.tl_location_callback, self.qos_sub)
        self.subscriber_frame = self.create_subscription(Image, sub_topic_name[1], self.frame_callback, self.qos_sub)


    def frame_callback(self, msg):
        self.img = np.array(msg.data).reshape([self.frame_size[1], self.frame_size[0], -1])

        try:
            self.img_cropped = self.img[self.y1 : self.y1 + self.h, self.x1 : self.x1 + self.w]

            self.r_sig, self.y_sig, self.g_sig = self.img_splitter(self.img_cropped, self.w)
            self.cnt_r = self.color_detector(self.r_sig)['R']
            self.cnt_y = self.color_detector(self.y_sig)['Y']
            self.cnt_g = self.color_detector(self.g_sig)['G']

            self.cnt = [self.cnt_r, self.cnt_y, self.cnt_g]

            self.num = self.cnt.index(max(self.cnt))

            if self.num == 0 and self.cnt[self.num] > 0.004:
                self.state = "R"

            elif self.num == 1  and self.cnt[self.num] > 0.003:
                self.state = "Y"

            elif self.num == 2  and self.cnt[self.num] > 0.001:
                self.state = "G"

            else:
                self.state = "N"

            if self.debug == True:
                cv2.imshow("TL_Image", self.img_cropped)
                cv2.imshow("TL_Image_r", self.r_sig)
                cv2.imshow("TL_Image_g", self.g_sig)
                cv2.waitKey(5)

            self.get_logger().info("State = " + str(self.state) + " | " + "HSV = " + str(self.cnt))

        except:
            self.state = None
            self.get_logger().warn("Unable to read frame")


    def tl_location_callback(self, msg):
        self.x1, self.y1, self.x2, self.y2 = map(int, msg.data)
        self.w = abs(self.x1 - self.x2)
        self.h = abs(self.y1 - self.y2)


    def img_splitter(self, img, width):
        r_sig = img[0 : -1, 0 : 0 + int(width/3)]
        y_sig = img[0 : -1, int(width/3) : int(width/3*2)]
        g_sig = img[0 : -1, int(width/3*2) : int(width)]

        return r_sig, y_sig, g_sig
    

    def color_detector(self, img):
        y_range, x_range, channel_range = img.shape
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        offset = np.array([[0, 128, 128]])

        cnt = {'R':0, 'Y':0, 'G':0, 'N':0}
        for y in range(y_range):
            for x in range(x_range):
                h, s, v = (img_lab[y][x] - offset).tolist()[0]

                if (
                   (R_RANGE[0][0][0] <= h <= R_RANGE[0][0][1] or R_RANGE[0][1][0] <= h <= R_RANGE[0][1][1]) and 
                    R_RANGE[1][0][0] <= s <= R_RANGE[1][0][1] and
                    R_RANGE[2][0][0] <= v <= R_RANGE[2][0][1]         
                    ):
                    cnt['R'] += 1

                elif (
                    Y_RANGE[0][0][0] <= h <= Y_RANGE[0][0][1] and 
                    Y_RANGE[1][0][0] <= s <= Y_RANGE[1][0][1] and
                    Y_RANGE[2][0][0] <= v <= Y_RANGE[2][0][1]         
                    ):
                    cnt['Y'] += 1

                elif (
                    G_RANGE[0][0][0] <= h <= G_RANGE[0][0][1] and 
                    G_RANGE[1][0][0] <= s <= G_RANGE[1][0][1] and
                    G_RANGE[2][0][0] <= v <= G_RANGE[2][0][1]         
                    ):
                    cnt['G'] += 1

                else:
                    cnt['N'] += 1
        
        for key in cnt.keys():
            cnt[key] = cnt[key] / sum(list(cnt.values()))

        return cnt

def main():
    rclpy.init()
    tl_recognizer_node = tl_recognizer(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE, DEBUG)
    rclpy.spin(tl_recognizer_node)

    tl_recognizer_node.destroy_node()
    rclpy.shutdown()

if __name__== "__main__":
    main()