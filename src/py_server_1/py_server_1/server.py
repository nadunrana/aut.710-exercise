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

# import csv
import rclpy
from rclpy.node import Node
import numpy as np
from math import sin, cos, pi, sqrt, pow
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


INTERVAL = 0.1
D = 0.4
R = 0.1
UNCERTAINTY_MODEL = [10, 3, 15, 4]
MESSAGE_LIMIT = 2216
coordinates_line = []
coordinates_cloud = []


# plt.ion()
# figure, ax = plt.subplots(figsize=(8, 8))
# plt.xlim([-1, 3])
# plt.ylim([-0.5, 3.5])
# plt.grid(color='grey', linestyle='-', linewidth=0.1)
# # line1, line2 = ax.plot(0, 0, 'b+', 0, 0, 'r--', linewidth=1)
# # line, = ax.plot(0, 0, 'b+', markersize=2, label="Trajectory")
# # cloud, = ax.plot(0, 0, 'r', markersize=2, label="Particles Cloud")
# # ax.legend()
# plt.title("Trajectories")
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")

# with open('/home/kido/fom_ws/record.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['x1', 'y1', 'theta1', 'x2', 'y2', 'theta2'])


class Server(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        self.x_euler = np.zeros((101, MESSAGE_LIMIT + 1))
        self.y_euler = np.zeros((101, MESSAGE_LIMIT + 1))
        self.yaw_euler = np.zeros((101, MESSAGE_LIMIT + 1))
        self.message_count = 0

        # self.publisher_euler = self.create_publisher(Pose, 'pose_euler', 10)
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/Wheels_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        while self.message_count < MESSAGE_LIMIT:
            rclpy.spin_once(self)

        self.plot()

    def plot(self):
        # plt.xlim([-1, 3])
        # plt.ylim([-0.5, 3.5])
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.grid(color='grey', linestyle='-', linewidth=0.1)
        plt.title("Trajectories")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.plot(self.x_euler[0, :], self.y_euler[0, :], 'b')
        # plt.plot(self.x_euler[1, :], self.y_euler[1, :], 'r')
        indices = [500, 1000, 1500, 2000]
        for index in indices:
            plt.plot(self.x_euler[:, index], self.y_euler[:, index], linestyle='None', marker=".")
            cov = np.cov(self.x_euler[:, index], self.y_euler[:, index])
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            for level in range(1, 4):
                ell = Ellipse(xy=(self.x_euler[0, index], self.y_euler[0, index]),
                              width=lambda_[0] * level * 2, height=lambda_[1] * level * 2,
                              angle=np.rad2deg(np.arccos(v[0, 0])))
                ell.set_facecolor('none')
                ell.set_edgecolor('black')
                ax.add_artist(ell)

        plt.show()

    def listener_callback(self, msg):
        self.message_count += 1
        self.get_logger().info('I heard: #%d "%s"' % (self.message_count, msg.data))
        self.step_calculation(msg.data[0], msg.data[1])

        # self.x_midpoint, self.y_midpoint, self.yaw_midpoint = self.step_calculation(msg.data[0], msg.data[1], method="midpoint")
        # print("Euler:", self.x_euler, self.y_euler, self.yaw_euler,
        #       ", midpoint: ", self.x_midpoint, self.y_midpoint, self.yaw_midpoint)

        # Draw trajectories
        # if self.message_count % 700 == 0:
        #     for i in range(1, 101):
        #         coordinates_cloud.append((self.x_euler[i], self.y_euler[i]))
        #     x_cloud, y_cloud = zip(*coordinates_cloud)
        #     cloud.set_xdata(x_cloud)
        #     cloud.set_ydata(y_cloud)

        # coordinates_line.append((self.x_euler[0], self.y_euler[0]))
        # x_line, y_line = zip(*coordinates_line)
        # line.set_xdata(x_line)
        # line.set_ydata(y_line)
        # figure.canvas.draw()
        # figure.canvas.flush_events()

        # Publish to topics
        # pose_euler = Pose()
        # pose_midpoint = Pose()
        # pose_euler.position.x = self.x_euler
        # pose_euler.position.y = self.y_euler
        # pose_midpoint.position.x = self.x_midpoint
        # pose_midpoint.position.y = self.y_midpoint
        # self.publisher_euler.publish(pose_euler)
        # self.publisher_midpoint.publish(pose_midpoint)

        # with open('/home/kido/fom_ws/record.csv', 'a+', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([self.x_euler, self.y_euler, self.yaw_euler, self.x_midpoint, self.y_midpoint, self.yaw_midpoint])

    def step_calculation(self, wl, wr):
        wl = wl * pi / 30
        wr = wr * pi / 30
        # inputs = np.matrix([[wl], [wr]])
        rot_head = INTERVAL * R * (wr - wl) / (2 * D)
        trans_head = INTERVAL * R * (wr + wl) / 2
        epsilon_rot = UNCERTAINTY_MODEL[0] * (rot_head ** 2) + UNCERTAINTY_MODEL[1] * (trans_head ** 2)
        epsilon_trans = UNCERTAINTY_MODEL[2] * (trans_head ** 2) + UNCERTAINTY_MODEL[3] * 2 * (rot_head ** 2)

        for i in range(0, 101):
            state_k = np.matrix([[self.x_euler[i, self.message_count - 1]],
                                 [self.y_euler[i, self.message_count - 1]],
                                 [self.yaw_euler[i, self.message_count - 1]]])

            if i > 0:
                sigma_rot_1 = rot_head + np.random.normal(0.0, epsilon_rot)
                sigma_rot_2 = rot_head + np.random.normal(0.0, epsilon_rot)
                sigma_trans = trans_head + np.random.normal(0.0, epsilon_trans)
                x_euler = state_k[0] + sigma_trans * cos(state_k[2] + sigma_rot_1)
                y_euler = state_k[1] + sigma_trans * sin(state_k[2] + sigma_rot_1)
                yaw_euler = state_k[2] + sigma_rot_1 + sigma_rot_2
                # print(sigma_rot_1, sigma_rot_2, sigma_trans)
            else:
                x_euler = state_k[0] + trans_head * cos(state_k[2])
                y_euler = state_k[1] + trans_head * sin(state_k[2])
                yaw_euler = state_k[2] + 2 * rot_head

            if yaw_euler > pi:
                yaw_euler -= 2 * pi
            elif yaw_euler < -pi:
                yaw_euler += 2 * pi

            self.x_euler[i, self.message_count] = x_euler
            self.y_euler[i, self.message_count] = y_euler
            self.yaw_euler[i, self.message_count] = yaw_euler


def main(args=None):
    rclpy.init(args=args)

    server = Server()

    # rclpy.spin(server)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
