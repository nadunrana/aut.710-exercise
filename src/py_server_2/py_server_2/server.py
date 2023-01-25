# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import rclpy
from rclpy.node import Node
import numpy as np
from math import sin, cos, pi, sqrt, pow
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt

INTERVAL = 0.1
D = 0.4
R = 0.1
coordinates_euler = []
coordinates_midpoint = []

plt.ion()
figure, ax = plt.subplots(figsize=(8, 8))
plt.xlim([-1, 3])
plt.ylim([-3, 1])
plt.grid(color='grey', linestyle='-', linewidth=0.1)
line1, line2 = ax.plot(0, 0, 'b+', 0, 0, 'r', linewidth=1)
plt.title("Trajectories")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")


class Server(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        self.x_euler = 0.0
        self.y_euler = 0.0
        self.yaw_euler = 0.0

        self.draw_trail = False

        self.x_midpoint = 0.0
        self.y_midpoint = 0.0
        self.yaw_midpoint = 0.0

        self.publisher_euler = self.create_publisher(Pose, 'pose_euler', 10)
        self.publisher_midpoint = self.create_publisher(Pose, 'pose_midpoint', 10)
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/Wheels_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        # for value in msg.data:
        #     print(value)
        self.x_euler, self.y_euler, self.yaw_euler = self.step_calculation(msg.data[0], msg.data[1])
        self.x_midpoint, self.y_midpoint, self.yaw_midpoint = self.step_calculation(msg.data[0], msg.data[1], method="midpoint")
        print("Euler:", self.x_euler, self.y_euler, self.yaw_euler,
              ", midpoint: ", self.x_midpoint, self.y_midpoint, self.yaw_midpoint)

        # Draw trajectories
        coordinates_euler.append((self.x_euler, self.y_euler))
        coordinates_midpoint.append((self.x_euler, self.y_euler))
        x_euler, y_euler = zip(*coordinates_euler)
        x_midpoint, y_midpoint = zip(*coordinates_midpoint)
        print("Difference:", euler_distance(self.x_euler, self.y_euler, self.x_midpoint, self.y_midpoint))
        line1.set_xdata(x_euler)
        line1.set_ydata(y_euler)
        line2.set_xdata(x_midpoint)
        line2.set_ydata(y_midpoint)
        figure.canvas.draw()
        figure.canvas.flush_events()

        # Publish to topics
        pose_euler = Pose()
        pose_midpoint = Pose()
        pose_euler.position.x = self.x_euler
        pose_euler.position.y = self.y_euler
        pose_midpoint.position.x = self.x_midpoint
        pose_midpoint.position.y = self.y_midpoint
        self.publisher_euler.publish(pose_euler)
        self.publisher_midpoint.publish(pose_midpoint)

    def step_calculation(self, wl, wr, method="euler"):
        wl = wl * pi / 30
        wr = wr * pi / 30

        if method == "euler":
            state_k = np.matrix([[self.x_euler],
                                 [self.y_euler],
                                 [self.yaw_euler]])
            func = np.matrix([[cos(state_k[2, 0]) / 2, cos(state_k[2, 0]) / 2],
                              [sin(state_k[2, 0]) / 2, sin(state_k[2, 0]) / 2],
                              [1 / D, -1 / D]])
            inputs = np.matrix([[wl], [wr]])
            state_kp1 = state_k + INTERVAL * R * func * inputs

            new_yaw = state_kp1[2, 0]
            if new_yaw > pi:
                new_yaw -= 2 * pi
            elif new_yaw < -pi:
                new_yaw += 2 * pi

            return state_kp1[0, 0], state_kp1[1, 0], new_yaw

        elif method == "midpoint":
            state_k = np.matrix([[self.x_midpoint],
                                 [self.y_midpoint],
                                 [self.yaw_midpoint]])
            func = np.matrix([[cos(state_k[2, 0]) / 2, cos(state_k[2, 0]) / 2],
                              [sin(state_k[2, 0]) / 2, sin(state_k[2, 0]) / 2],
                              [1 / D, -1 / D]])
            inputs = np.matrix([[wl], [wr]])
            state_kp1 = state_k + INTERVAL * R * func * inputs
            mid_state = (state_k + state_kp1) / 2
            mid_yaw = mid_state[2, 0]
            mid_func = np.matrix([[cos(mid_yaw) / 2, cos(mid_yaw) / 2],
                                  [sin(mid_yaw) / 2, sin(mid_yaw) / 2],
                                  [1/D, -1/D]])

            # Assume the input at k+1 is not change
            state_kp1 = state_k + INTERVAL * R * mid_func * inputs
            self.previous_input = inputs

            new_yaw = state_kp1[2, 0]
            if new_yaw > pi:
                new_yaw -= 2 * pi
            elif new_yaw < -pi:
                new_yaw += 2 * pi

            return state_kp1[0, 0], state_kp1[1, 0], new_yaw


def euler_distance(x1, y1, x2, y2):
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def main(args=None):
    rclpy.init(args=args)

    server = Server()

    rclpy.spin(server)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
