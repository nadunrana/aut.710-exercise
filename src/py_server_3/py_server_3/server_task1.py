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
from math import sin, cos, sqrt, pi
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Constants
INTERVAL = 0.1
D = 0.44
Radius = 0.12
ODOM_LIMIT = 408
SENSOR_LIMIT = 136
EPSILON = 0.25
R = np.eye(6) * 0.000025
M = np.matrix('0.001, 0, 0; 0, 0.0002, 0; 0, 0, 0.0002')


# L0 = np.matrix('7.3; -4.5')
# L1 = np.matrix('-5.5; 8.3')
# L2 = np.matrix('-7.5; -6.3')


class Server(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Temporary variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # self.L0 = np.zeros((2, 1))
        # self.L1 = np.zeros((2, 1))
        # self.L2 = np.zeros((2, 1))
        self.L0 = np.matrix('10;0')
        self.L1 = np.matrix('10;10')
        self.L2 = np.matrix('0;10')
        # self.P = np.zeros((9, 9))
        self.P = np.eye(9)
        self.Q = np.zeros((9, 9))
        self.A = np.eye(9)

        # Logged variables
        self.x_predict = np.zeros(ODOM_LIMIT + 1)
        self.y_predict = np.zeros(ODOM_LIMIT + 1)
        self.yaw_predict = np.zeros(ODOM_LIMIT + 1)
        self.P_predict = np.zeros((9, 9, ODOM_LIMIT + 1))
        self.P_predict[:, :, 0] = self.P

        self.x_update = np.zeros(SENSOR_LIMIT + 1)
        self.y_update = np.zeros(SENSOR_LIMIT + 1)
        self.yaw_update = np.zeros(SENSOR_LIMIT + 1)
        self.P_update = np.zeros((9, 9, SENSOR_LIMIT + 1))
        self.L0_update = np.zeros((2, SENSOR_LIMIT + 1))
        self.L1_update = np.zeros((2, SENSOR_LIMIT + 1))
        self.L2_update = np.zeros((2, SENSOR_LIMIT + 1))
        self.L0_update[:, 0:1] = self.L0
        self.L1_update[:, 0:1] = self.L1
        self.L2_update[:, 0:1] = self.L2
        self.P_update[:, :, 0] = self.P

        self.wl = np.zeros(ODOM_LIMIT + 1)
        self.wr = np.zeros(ODOM_LIMIT + 1)

        self.measure_hat = np.zeros((6, SENSOR_LIMIT + 1))
        self.measure = np.zeros((6, SENSOR_LIMIT + 1))

        # Counter for indexing
        self.odom_count = 0
        self.sensor_count = 0

        # Subscriber
        self.sensor_subscription = self.create_subscription(
            JointState,
            '/landmarks',
            self.sensor_callback,
            20)
        self.odom_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.odom_callback,
            20)

        self.sensor_subscription  # prevent unused variable warning
        self.odom_subscription

        # while self.odom_count < ODOM_LIMIT:
        #     rclpy.spin_once(self)
        #
        # self.plot()

    def canvas(self):
        # Turn on interactive mode
        plt.ion()
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.xlim([-10, 30])
        plt.ylim([-7, 9])
        plt.grid(color='grey', linestyle='-', linewidth=0.1)
        plt.title("Trajectories")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")

        # Landmarks
        # plt.plot(L0[0, 0], L0[1, 0], linestyle='None', marker="o", label='Landmark 0')
        # plt.plot(L1[0, 0], L1[1, 0], linestyle='None', marker="o", label='Landmark 1')
        # plt.plot(L2[0, 0], L2[1, 0], linestyle='None', marker="o", label='Landmark 2')

        # Initialization for animation and text labels
        predicted_coordinates = []
        updated_coordinates = []
        predict, = ax.plot(0, 0, 'b+', markersize=2, label="Predicted position")
        update, = ax.plot(0, 0, 'r--', linewidth=1, label="Updated position")
        ax.legend()
        text0 = plt.text(0.2, 0.1, "Time: 0.0", fontsize=10, horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes)
        text1 = plt.text(0.2, 0.2, "Coordinate: 0.0, 0.0", fontsize=10, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes)
        text2 = plt.text(0.2, 0.3, "Predicted Dist: 0.0 , 0.0, 0.0", fontsize=10, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes)
        text3 = plt.text(0.2, 0.4, "Measured Dist: 0.0, 0.0, 0.0", fontsize=10, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes)

        # Iterate through time array to plot the robot position and update state and measurement
        for index in range(0, ODOM_LIMIT):
            # Predicted position
            predicted_coordinates.append((self.x_predict[index], self.y_predict[index]))
            x_predict, y_predict = zip(*predicted_coordinates)
            predict.set_xdata(x_predict)
            predict.set_ydata(y_predict)
            text0.set_text(f"Time: {round(index * INTERVAL, 2)}")
            text1.set_text(f"Coordinate: {round(self.x_predict[index], 2)}, {round(self.y_predict[index], 2)}")

            if index % 5 == 0:
                # Updated position
                i = int(index / 5)
                updated_coordinates.append((self.x_update[i], self.y_update[i]))
                text2.set_text(
                    f"Predicted Dist: {round(self.dist_hat[0, i], 2)}, {round(self.dist_hat[1, i], 2)}, {round(self.dist_hat[2, i], 2)}")
                text3.set_text(
                    f"Measured Dist: {round(self.dist[0, i], 2)}, {round(self.dist[1, i], 2)}, {round(self.dist[2, i], 2)}")
                x_update, y_update = zip(*updated_coordinates)
                update.set_xdata(x_update)
                update.set_ydata(y_update)
                lambda_, v = np.linalg.eig(self.P_update[:, :, i])
                ell = Ellipse(xy=(self.x_update[i], self.y_update[i]),
                              width=lambda_[0] * 2, height=lambda_[1] * 2,
                              angle=np.rad2deg(np.arccos(v[0, 0])))
                ell.set_facecolor('none')
                ell.set_edgecolor('black')
                ax.add_artist(ell)

            # Update the frame ~ making animation
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Stop the interactive mode
            if index == ODOM_LIMIT - 1:
                plt.ioff()
                plt.show()

    def plot(self):
        # Create the time arrays
        tspan1 = np.arange(0, INTERVAL * 409 - 0.05, INTERVAL)
        tspan2 = np.arange(0, 3 * INTERVAL * 137 - 0.01, 3 * INTERVAL)

        # Plot the wheels velocity
        fig1 = plt.figure(2)
        plt.plot(tspan1, self.wl, label="Left wheel")
        plt.plot(tspan1, self.wr, label="Right wheel")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Wheels velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (rad/s)")

        # Plot the predicted and updated covariances
        fig2 = plt.figure(3)
        plt.plot(tspan1, self.P_predict[0, 0, :], label="Predicted covariance of x")
        plt.plot(tspan1, self.P_predict[1, 1, :], label="Predicted covariance of y")
        plt.plot(tspan1, self.P_predict[2, 2, :], label="Predicted covariance of psi")
        plt.plot(tspan2, self.P_update[0, 0, :], label="Updated covariance of x")
        plt.plot(tspan2, self.P_update[1, 1, :], label="Updated covariance of y")
        plt.plot(tspan2, self.P_update[2, 2, :], label="Updated covariance of psi")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Robot Covariances")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")

        fig3 = plt.figure(4)
        plt.plot(tspan1, self.P_predict[3, 3, :], label="Predicted covariance of m1x")
        plt.plot(tspan1, self.P_predict[4, 4, :], label="Predicted covariance of m1y")
        plt.plot(tspan1, self.P_predict[5, 5, :], label="Predicted covariance of m2x")
        plt.plot(tspan1, self.P_predict[6, 6, :], label="Predicted covariance of m2y")
        plt.plot(tspan1, self.P_predict[7, 7, :], label="Predicted covariance of m3x")
        plt.plot(tspan1, self.P_predict[8, 8, :], label="Predicted covariance of m3y")
        plt.plot(tspan2, self.P_update[3, 3, :], label="Updated covariance of m1x")
        plt.plot(tspan2, self.P_update[4, 4, :], label="Updated covariance of m1y")
        plt.plot(tspan2, self.P_update[5, 5, :], label="Updated covariance of m2x")
        plt.plot(tspan2, self.P_update[6, 6, :], label="Updated covariance of m2y")
        plt.plot(tspan2, self.P_update[7, 7, :], label="Updated covariance of m3x")
        plt.plot(tspan2, self.P_update[8, 8, :], label="Updated covariance of m3y")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Landmark Covariances")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")

        # Plot the predicted and measured distance to landmark
        fig4 = plt.figure(5)
        plt.plot(tspan2, self.measure[0, :], label="Measured distance LM1")
        plt.plot(tspan2, self.measure[1, :], label="Measured distance LM2")
        plt.plot(tspan2, self.measure[2, :], label="Measured distance LM3")
        plt.plot(tspan2, self.measure_hat[0, :], label="Predicted distance LM1")
        plt.plot(tspan2, self.measure_hat[1, :], label="Predicted distance LM2")
        plt.plot(tspan2, self.measure_hat[2, :], label="Predicted distance LM3")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Distance to Landmarks")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance")

        fig5 = plt.figure(6)
        plt.plot(tspan2, self.measure[3, :], label="Measured bearing LM1")
        plt.plot(tspan2, self.measure[4, :], label="Measured bearing LM2")
        plt.plot(tspan2, self.measure[5, :], label="Measured bearing LM3")
        plt.plot(tspan2, self.measure_hat[3, :], label="Predicted bearing LM1")
        plt.plot(tspan2, self.measure_hat[4, :], label="Predicted bearing LM2")
        plt.plot(tspan2, self.measure_hat[5, :], label="Predicted bearing LM3")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Angle to Landmarks")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle")
        fig1.legend()
        fig2.legend()
        fig3.legend()
        fig4.legend()
        fig5.legend()

        # Show the plot but continue the thread
        # plt.show(block=False)
        plt.show()

    def odom_callback(self, msg):
        # Increase the counter and process the prediction
        self.odom_count += 1
        self.get_logger().info('Odom says: #%d "%s"' % (self.odom_count, msg.velocity))
        self.step_calculation(msg.velocity[0], msg.velocity[1])

    def sensor_callback(self, msg):
        # Increase the counter and process the update
        self.sensor_count += 1
        self.get_logger().info('Sensor says: #%d "%s %s"' % (self.sensor_count, msg.position, msg.velocity))
        self.update(msg.position[0], msg.position[1], msg.position[2],
                    msg.velocity[0], msg.velocity[1], msg.velocity[2])

    def update(self, dist_0, dist_1, dist_2, bear_0, bear_1, bear_2):
        index = self.sensor_count
        P_neg = self.P
        x = self.x
        y = self.y
        yaw = self.yaw

        L0 = self.L0
        L1 = self.L1
        L2 = self.L2

        bear_0 = bear_0 * pi / 180
        bear_1 = bear_1 * pi / 180
        bear_2 = bear_2 * pi / 180

        # Predict the measurement
        dist_hat_0 = sqrt((x - L0[0, 0]) ** 2 + (y - L0[1, 0]) ** 2)
        dist_hat_1 = sqrt((x - L1[0, 0]) ** 2 + (y - L1[1, 0]) ** 2)
        dist_hat_2 = sqrt((x - L2[0, 0]) ** 2 + (y - L2[1, 0]) ** 2)
        bear_hat_0 = regulate(np.arctan2(L0[0, 0] - x, L0[1, 0] - y) - yaw)
        bear_hat_1 = regulate(np.arctan2(L1[0, 0] - x, L1[1, 0] - y) - yaw)
        bear_hat_2 = regulate(np.arctan2(L2[0, 0] - x, L2[1, 0] - y) - yaw)

        z = np.matrix(f"{dist_0};{dist_1};{dist_2};{bear_0};{bear_1};{bear_2}")
        zhat = np.matrix(f"{dist_hat_0};{dist_hat_1};{dist_hat_2};{bear_hat_0};{bear_hat_1};{bear_hat_2}")

        # H linearization
        H = np.zeros((6, 9))
        H[0, :] = np.matrix([x - L0[0, 0], y - L0[1, 0], 0,
                             L0[0, 0] - x, L0[1, 0] - y, 0,
                             0, 0, 0]) / dist_hat_0

        H[1, :] = np.matrix([x - L1[0, 0], y - L1[1, 0], 0,
                             0, 0, L1[0, 0] - x,
                             L1[1, 0] - y, 0, 0]) / dist_hat_1

        H[2, :] = np.matrix([x - L2[0, 0], y - L2[1, 0], 0,
                             0, 0, 0,
                             0, L2[0, 0] - x, L2[1, 0] - y]) / dist_hat_2

        H[3, :] = np.matrix([y - L0[1, 0], L0[0, 0] - x, -(dist_hat_0 ** 2),
                             L0[1, 0] - y, x - L0[0, 0], 0,
                             0, 0, 0]) / (dist_hat_0 ** 2)

        H[4, :] = np.matrix([y - L1[1, 0], L1[0, 0] - x, -(dist_hat_1 ** 2),
                             0, 0, L1[1, 0] - y,
                             x - L1[0, 0], 0, 0]) / (dist_hat_1 ** 2)

        H[5, :] = np.matrix([y - L2[1, 0], L2[0, 0] - x, -(dist_hat_2 ** 2),
                             0, 0, 0,
                             0, L2[1, 0] - y, x - L2[0, 0]]) / (dist_hat_2 ** 2)

        # Posterior
        K = P_neg * H.T * np.linalg.inv(H * P_neg * H.T + R)
        print(H)
        P = (np.eye(9) - K * H) * P_neg
        P_pos = 0.5 * (P + P.T)
        updated_position = np.matrix(f'{x};{y};{yaw};{L0[0, 0]}; {L0[1, 0]};{L1[0, 0]}; {L1[1, 0]}; {L2[0, 0]}; {L2[1, 0]}') + K * (z - zhat)
        # print(updated_position)
        # Save to temporary variables and logged variables
        self.x = updated_position[0, 0]
        self.y = updated_position[1, 0]
        self.yaw = updated_position[2, 0]
        self.P = P_pos
        self.L0 = updated_position[3:5, 0]
        self.L1 = updated_position[5:7, 0]
        self.L2 = updated_position[7:9, 0]
        # print(self.L0_update[:, index:index + 1], updated_position[3:5, 0])
        self.L0_update[:, index:index + 1] = updated_position[3:5, 0]
        self.L1_update[:, index:index + 1] = updated_position[5:7, 0]
        self.L2_update[:, index:index + 1] = updated_position[7:9, 0]

        self.x_update[index] = updated_position[0, 0]
        self.y_update[index] = updated_position[1, 0]
        self.yaw_update[index] = updated_position[2, 0]
        self.P_update[:, :, index] = P_pos
        print("Measure Diff:", dist_0 - dist_hat_0, dist_1 - dist_hat_1, dist_2 - dist_hat_2,
                               bear_0 - bear_hat_0, bear_1 - bear_hat_1, bear_2 - bear_hat_2)
        print("New Landmark:", self.L0, self.L1, self.L2)
        self.measure_hat[:, index:index + 1] = zhat
        self.measure[:, index:index + 1] = z


    def step_calculation(self, wl, wr):
        rot_head = INTERVAL * Radius * (wr - wl) / (2 * D)
        trans_head = INTERVAL * Radius * (wr + wl) / 2

        index = self.odom_count
        x = self.x
        y = self.y
        yaw = self.yaw

        A = self.A
        P = self.P

        # Log the wheels speed
        self.wl[index] = wl
        self.wr[index] = wr

        x_predict = x + trans_head * cos(yaw + rot_head)
        y_predict = y + trans_head * sin(yaw + rot_head)
        yaw_predict = yaw + rot_head * 2
        # print(round(yaw_predict, 4))

        # Calculate the predicted positions and save to temporary and logged variables
        self.x_predict[index] = x_predict
        self.y_predict[index] = y_predict
        self.yaw_predict[index] = yaw_predict

        self.x = x_predict
        self.y = y_predict
        self.yaw = yaw_predict

        # Linearization
        # L = np.eye(9)
        L = np.zeros((9, 3))
        L[0, 0] = cos(yaw + rot_head)
        L[0, 1] = -trans_head * sin(yaw + rot_head)
        L[1, 0] = sin(yaw + rot_head)
        L[1, 1] = trans_head * cos(yaw + rot_head)
        L[2, 1] = 1
        L[2, 2] = 1

        A[0, 2] = - trans_head * sin(yaw + rot_head)
        A[1, 2] = trans_head * cos(yaw + rot_head)
        # P = A * P * A.T + M
        # print(L)
        # print(M)

        P = A * P * A.T + L * M * L.T
        P = 0.5 * (P + P.T)
        # np.matmul(np.matmul(L, M).reshape((3, 3)), L.T).reshape((9, 9))
        # Save to temporary and logged variables
        self.A = A
        self.P = P
        self.P_predict[:, :, index] = P

        print(self.A)

        # Plot after the last messages
        if index == ODOM_LIMIT:
            # print(self.x_predict.shape, self.x_update.shape)
            # print(self.x_update)
            # print(self.yaw_predict)
            self.plot()
            # self.canvas()


def regulate(angle):
    if angle < -pi:
        angle += 2 * pi
    elif angle > pi:
        angle -= 2 * pi
    return angle

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
