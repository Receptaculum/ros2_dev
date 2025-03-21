import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import serial
import time


## < Parameter> #####################################################################################

# 통신 장치의 경로
PORT_NAME = "/dev/ttyUSB0"

# Baud Rate
BAUD_RATE = 38400

# 노드 이름
NODE_NAME = "ros_uart_communicator"

# 구독 토픽 이름
SUB_TOPIC_NAME = "motion_decider_command"

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


class ros_uart_communicator(Node):
    def __init__(self, node_name, port_name, baud_rate, sub_topic_name):
        super().__init__(node_name)

        self.serial = serial.Serial(port_name, baud_rate, rtscts=False)

        self.qos = QoSProfile( # QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        self.subscriber = self.create_subscription(UInt8MultiArray, sub_topic_name, self.send_callback, self.qos)

    def send_callback(self, msg):

        steer_angle = msg.data[0]
        left_motor_speed = msg.data[1]
        right_motor_speed = msg.data[2]
        
        # 조향 s (0~255) 1 byte | 속도 v (0~255) 2 byte
        # bbbbbbbb(조향) bbbbbbbb(속도_좌) bbbbbbbb(속도_우)
        steer_angle_bin = hex(steer_angle)[2:].zfill(2)
        left_motor_speed =  hex(left_motor_speed)[2:].zfill(2)
        right_motor_speed =  hex(right_motor_speed)[2:].zfill(2)

        msg_hex = '0x' + steer_angle_bin + left_motor_speed + right_motor_speed
        msg = int(msg_hex, 16).to_bytes(3, byteorder='big', signed=False)  

        self.serial.write(msg)
        self.get_logger().info(f"{msg}")


def main():
    rclpy.init()
    communication_node = ros_uart_communicator(NODE_NAME, PORT_NAME, BAUD_RATE, SUB_TOPIC_NAME)
    rclpy.spin(communication_node)

    communication_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()