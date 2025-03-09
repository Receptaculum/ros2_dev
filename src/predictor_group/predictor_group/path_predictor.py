import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import cv_bridge

import numpy as np


## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "path_predictor"

# 발행 토픽 이름
TOPIC_NAME = None

# 구독 토픽 이름 (2개 | lane1, lane2)
SUB_TOPIC_NAME = ["cv_lane_2", "cv_lane_2"]

# CV 처리 영상 출력 여부
DEBUG = True

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# 도로 추출 영역 (가로, 세로)
CUT_OFF = [[0, 639], [350, 475]]

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
    def __init__(self, node_name, sub_topic_name : list, frame_size, cut_off):
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

        # Subscriber 선언
        self.subscriber_lane_1 = Subscriber(self, Float32MultiArray, sub_topic_name[0], qos_profile=self.qos_sub)
        self.subscriber_lane_2 = Subscriber(self, Float32MultiArray, sub_topic_name[1], qos_profile=self.qos_sub)
        self.sync = ApproximateTimeSynchronizer([self.subscriber_lane_1, self.subscriber_lane_2], 1, 0.1, allow_headerless=True)

        self.sync.registerCallback(self.path_prediction_callback)


    def path_prediction_callback(self, msg1, msg2):
        background = np.zeros([self.frame_size[1], self.frame_size[0]], dtype=np.uint8)
        point = np.array(msg1.data, dtype=int).reshape(-1, 2)
        point_new = []

        for x, y in point.tolist():
                if (self.cut_off[0][0] <= x <= self.cut_off[0][1] and
                    self.cut_off[1][0] <= y <= self.cut_off[1][1]):
                        background[y][x] = 255

        data = cv2.HoughLinesP(background, 0.5, np.pi/720, threshold=4, minLineLength=5, maxLineGap=300)

        out = np.zeros([self.frame_size[1], self.frame_size[0]], dtype=np.uint8)
        

        try:
                lhs = []
                rhs = []

                l_size = []
                r_size = []

                try:
                        for line in data:
                                x1, y1, x2, y2 = line[0]

                                try:
                                        if min(x1, x2) < self.frame_size[0]/2:
                                                lhs.append([x1, y1, x2, y2])
                                                l_size.append(abs(y1-y2))
                                except:
                                       print("err1")


                                try:
                                        if max(x1, x2) > self.frame_size[0]/2:
                                                rhs.append([x1, y1, x2, y2])
                                                r_size.append(abs(y1-y2))
                                except:
                                       print("err2")


                except Exception as e:
                        print("internal", e)
                        pass

      
                l_line = lhs[l_size.index(max(l_size))]                
                r_line = rhs[r_size.index(max(r_size))]

                if abs(max(l_line[0], l_line[2]) - max(r_line[0], r_line[2])) < 200:
                       lhs = []

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
                        # cv2.line(out, (l_line[0], l_line[1]), (l_line[2], l_line[3]), (255, ))
                
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
                        # cv2.line(out, (r_line[0], r_line[1]), (r_line[2], r_line[3]), (255, ))

        except Exception as e:
                print(e)
                pass
        
        cv2.imshow("win", out)
        cv2.waitKey(5)






def main():
    rclpy.init()
    path_preictor_node = path_preictor(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE, CUT_OFF)
    rclpy.spin(path_preictor_node)

    path_preictor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()