import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int8
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import cv_bridge

import numpy as np

import math

## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "path_predictor"

# 발행 토픽 이름
TOPIC_NAME = "path_angle"

# 구독 토픽 이름 (2개 | lane1, lane2)
SUB_TOPIC_NAME = ["cv_lane_2", "cv_lane_2"]

# CV 처리 영상 출력 여부
DEBUG = True

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# 도로 추출 영역 (가로, 세로)
CUT_OFF = [[0, 639], [350, 475]]

# 화면 출력 여부
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

class path_preictor(Node):
    def __init__(self, node_name, sub_topic_name : list, frame_size, cut_off, debug, topic_name):
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
        self.cut_off = cut_off
        self.debug = debug

        # Subscriber 선언
        self.subscriber_lane_1 = Subscriber(self, Float32MultiArray, sub_topic_name[0], qos_profile=self.qos_sub)
        self.subscriber_lane_2 = Subscriber(self, Float32MultiArray, sub_topic_name[1], qos_profile=self.qos_sub)
        self.sync = ApproximateTimeSynchronizer([self.subscriber_lane_1, self.subscriber_lane_2], 1, 0.1, allow_headerless=True)

        self.sync.registerCallback(self.path_prediction_callback)

        # Publisher 선언
        self.publisher = self.create_publisher(Int8, topic_name, self.qos_pub)
        self.msg = Int8()

    def path_prediction_callback(self, msg1 : Float32MultiArray, msg2 : Float32MultiArray):
        data_msg1 = self.hough_extractor(msg1)
        center_msg1 = self.center_extractor(data_msg1)
        angle = int(self.angle_calculator(center_msg1)*180/math.pi)

        self.get_logger().info(f"Angle = {angle} [deg]")
        self.msg.data = angle
        self.publisher.publish(self.msg)   

        if self.debug == True:
                background = np.zeros([self.frame_size[1], self.frame_size[0]])
                
                for x, y, in center_msg1:
                        if 0 <= x < self.frame_size[0] and 0 <= y < self.frame_size[1]:
                                background[y][x] = 1

                cv2.imshow("Center", background)
                cv2.waitKey(2)

        

    def angle_calculator(self, data : list):
        try:
                p1 = data[0]
                p2 = data[-1]
                self.mem_p1 = data[0]
                self.mem_p2 = data[-1]

        except:
                p1 = self.mem_p1
                p2 = self.mem_p2


        try:
                return math.atan((p1[0]-p2[0])/(p1[1]-p2[1]))
        
        except:
                return math.pi/2 
        

    def center_extractor(self, data : dict):
        return_val = []

        if 'l' in data.keys() and 'r' in data.keys(): # l,r 둘 다 존재
                data_l = sorted(data['l'], key = lambda x : x[1], reverse=True)
                data_r = sorted(data['r'], key = lambda x : x[1], reverse=True)

                for k in range(0, self.cut_off[1][1] - self.cut_off[1][0], 15):
                        try:
                               delta = abs(data_l[k][0] - data_r[k][0])
                               return_val.append([int(data_l[k][0] + delta/2), data_l[k][1]])                        
                        except Exception as e:
                               self.get_logger().warn(f"error occured in center_extractor case 1 | {e}")
                               pass

        elif 'l' in data.keys() and not 'r' in data.keys(): # l만 존재
                data_l = sorted(data['l'], key = lambda x : x[1], reverse=True)

                for k in range(0, self.cut_off[1][1] - self.cut_off[1][0], 15):
                        try:
                               delta = 300
                               return_val.append([int(data_l[k][0] + delta/2), data_l[k][1]])  
                        except Exception as e:
                               self.get_logger().warn(f"error occured in center_extractor case 2 | {e}")                               
                               pass
         
        elif not 'l' in data.keys() and 'r' in data.keys(): # r만 존재
                data_r = sorted(data['r'], key = lambda x : x[1], reverse=True)
 
                for k in range(0, self.cut_off[1][1] - self.cut_off[1][0], 15):
                        try:
                               delta = 300
                               return_val.append([int(data_r[k][0] - delta/2), data_r[k][1]])  
                        except Exception as e:
                               self.get_logger().warn(f"error occured in center_extractor case 3 | {e}")                               
                               pass
        
        else:
                self.get_logger().warn("error occured in center_extractor case 4")
                pass

        return return_val



    def hough_extractor(self, msg : Float32MultiArray) -> dict:
        background = np.zeros([self.frame_size[1], self.frame_size[0]], dtype=np.uint8)
        point = np.array(msg.data, dtype=int).reshape(-1, 2)

        for x, y in point.tolist():
                if (self.cut_off[0][0] <= x <= self.cut_off[0][1] and
                    self.cut_off[1][0] <= y <= self.cut_off[1][1]):
                        background[y][x] = 255

        data = cv2.HoughLinesP(background, 0.5, np.pi/720, threshold=4, minLineLength=5, maxLineGap=300)
        out = np.zeros([self.frame_size[1], self.frame_size[0]], dtype=np.uint8)
        
        # return을 위한 dict 선언
        return_data = dict()


        try:
                # 왼쪽, 오른쪽 좌표값 저장
                lhs = []
                rhs = []

                # 좌표값의 크기(길이) 저장
                l_size = []
                r_size = []

                for line in data:
                        x1, y1, x2, y2 = line[0]

                        if min(x1, x2) < self.frame_size[0]/2:
                                lhs.append([x1, y1, x2, y2])
                                l_size.append(abs(y1-y2))

                                
                        if max(x1, x2) > self.frame_size[0]/2:
                                rhs.append([x1, y1, x2, y2])
                                r_size.append(abs(y1-y2))
      
                l_line = lhs[l_size.index(max(l_size))]                
                r_line = rhs[r_size.index(max(r_size))]

                # 2개의 선이 중첩되는 문제를 제거하는 코드
                if abs(max(l_line[0], l_line[2]) - max(r_line[0], r_line[2])) < 200:
                        if (l_line[0]-l_line[2])*(l_line[1]-l_line[3]) > 0:
                                lhs = []
                        else:
                                rhs = []

                # 왼쪽 직선 처리
                if lhs != []:
                        y = np.array(list(range(0, self.frame_size[1], 1)))
                        x =  (l_line[0] - l_line[2])/(l_line[1] - l_line[3])*(y - l_line[1]) + l_line[0]

                        y = y.tolist()
                        x = x.tolist()

                        xy = []
                        for n in range(len(x)):
                                if x[n] >= 0:
                                      xy.append([int(x[n]), y[n]])
    
                        cv2.line(out, (xy[0][0], xy[0][1]), (xy[-1][0], xy[-1][1]), (255, ))
                        return_data['l'] = xy
                
                # 오른쪽 직선 처리
                if rhs != []:
                        y = np.array(list(range(0, self.frame_size[1], 1)))
                        x =  (r_line[0] - r_line[2])/(r_line[1] - r_line[3])*(y - r_line[1]) + r_line[0]

                        y = y.tolist()
                        x = x.tolist()

                        xy = []
                        for n in range(len(x)):
                                if x[n] >= 0:
                                      xy.append([int(x[n]), y[n]])

                        cv2.line(out, (xy[0][0], xy[0][1]), (xy[-1][0], xy[-1][1]), (255, ))
                        return_data['r'] = xy

        except Exception as e:
                self.get_logger().warn(f"{e}")
                pass
        
        if self.debug == True:
                cv2.imshow("Hough", out)
                cv2.waitKey(2)

        return return_data

def main():
    rclpy.init()
    path_preictor_node = path_preictor(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE, CUT_OFF, DEBUG, TOPIC_NAME)
    rclpy.spin(path_preictor_node)

    path_preictor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()