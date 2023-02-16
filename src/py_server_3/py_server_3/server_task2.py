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
ODOM_LIMIT = 383
SENSOR_LIMIT = 127
EPSILON = 0.25
R = np.eye(3) * 0.000025
M = np.matrix('0.001, 0, 0; 0, 0.0002, 0; 0, 0, 0.0002')
L0 = np.matrix('7.5; -4.0')
L1 = np.matrix('-5.0; 8.0')
L2 = np.matrix('-7.0; -6.5')


class Server(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Temporary variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.P = np.zeros((3, 3))
        self.Q = np.zeros((3, 3))
        self.A = np.eye(3)

        # Logged variables
        self.x_predict = np.zeros(ODOM_LIMIT + 1)
        self.y_predict = np.zeros(ODOM_LIMIT + 1)
        self.yaw_predict = np.zeros(ODOM_LIMIT + 1)
        self.P_predict = np.zeros((3, 3, ODOM_LIMIT + 1))

        self.x_update = np.zeros(SENSOR_LIMIT + 1)
        self.y_update = np.zeros(SENSOR_LIMIT + 1)
        self.yaw_update = np.zeros(SENSOR_LIMIT + 1)
        self.P_update = np.zeros((3, 3, SENSOR_LIMIT + 1))

        self.wl = np.zeros(ODOM_LIMIT + 1)
        self.wr = np.zeros(ODOM_LIMIT + 1)

        self.dist_hat = np.zeros((3, SENSOR_LIMIT + 1))
        self.dist = np.zeros((3, SENSOR_LIMIT + 1))

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
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.grid(color='grey', linestyle='-', linewidth=0.1)
        plt.title("Trajectories")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")

        # Landmarks
        plt.plot(L0[0, 0], L0[1, 0], linestyle='None', marker="o", label='Landmark 0')
        plt.plot(L1[0, 0], L1[1, 0], linestyle='None', marker="o", label='Landmark 1')
        plt.plot(L2[0, 0], L2[1, 0], linestyle='None', marker="o", label='Landmark 2')

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

            if index % 3 == 0:
                # Updated position
                i = int(index / 3)
                updated_coordinates.append((self.x_update[i], self.y_update[i]))
                text2.set_text(f"Predicted Dist: {round(self.dist_hat[0, i], 2)}, {round(self.dist_hat[1, i], 2)}, {round(self.dist_hat[2, i], 2)}")
                text3.set_text(f"Measured Dist: {round(self.dist[0, i], 2)}, {round(self.dist[1, i], 2)}, {round(self.dist[2, i], 2)}")
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
        tspan1 = np.arange(0, INTERVAL * ODOM_LIMIT + 0.05, INTERVAL)
        tspan2 = np.arange(0, 3 * INTERVAL * SENSOR_LIMIT + 0.05, 3 * INTERVAL)

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
        plt.title("Covariances")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")

        # Plot the predicted and measured distance to landmark
        fig3 = plt.figure(4)
        # plt.plot(tspan2, self.dist[0, :], label="Measured distance LM1")
        # plt.plot(tspan2, self.dist[1, :], label="Measured distance LM2")
        # plt.plot(tspan2, self.dist[2, :], label="Measured distance LM3")
        plt.plot(tspan2, self.dist_hat[0, :], label="Predicted distance LM1")
        plt.plot(tspan2, self.dist_hat[1, :], label="Predicted distance LM2")
        plt.plot(tspan2, self.dist_hat[2, :], label="Predicted distance LM3")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Distance to Landmarks")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance")
        fig1.legend()
        fig2.legend()
        fig3.legend()

        # Show the plot but continue the thread
        plt.show(block=False)

    def odom_callback(self, msg):
        # Increase the counter and process the prediction
        self.odom_count += 1
        self.get_logger().info('Odom says: #%d "%s"' % (self.odom_count, msg.velocity))
        self.step_calculation(msg.velocity[0], msg.velocity[1])

    def sensor_callback(self, msg):
        # Increase the counter and process the update
        self.sensor_count += 1
        self.get_logger().info('Sensor says: #%d "%s"' % (self.sensor_count, msg.position))
        self.update(msg.position[0], msg.position[1], msg.position[2])

    def update(self, dist_0, dist_1, dist_2):
        index = self.sensor_count
        P = self.P
        x = self.x
        y = self.y
        yaw = self.yaw

        # Predict the measurement
        dist_hat_0 = sqrt((x - L0[0, 0]) ** 2 + (y - L0[1, 0]) ** 2)
        dist_hat_1 = sqrt((x - L1[0, 0]) ** 2 + (y - L1[1, 0]) ** 2)
        dist_hat_2 = sqrt((x - L2[0, 0]) ** 2 + (y - L2[1, 0]) ** 2)

        z = np.array([dist_0, dist_1, dist_2])
        zh = np.array([dist_hat_0, dist_hat_1, dist_hat_2])


        H = np.zeros((3, 3))
        H[0, 0] = (x - L0[0, 0]) / dist_hat_0
        H[0, 1] = (y - L0[1, 0]) / dist_hat_0
        H[1, 0] = (x - L1[0, 0]) / dist_hat_1
        H[1, 1] = (y - L1[1, 0]) / dist_hat_1
        H[2, 0] = (x - L2[0, 0]) / dist_hat_2
        H[2, 1] = (y - L2[1, 0]) / dist_hat_2

        state = np.matrix(f'{x};{y};{yaw}')

        for i in range(0, 3):
            vi = []
            Si = []
            for j in range(0, 3):
                vij = z[i] - zh[j]
                Sij = H[j:j + 1, 0:3] * P * H[j:j + 1, 0:3].T + R[j, j]
                Xij = vij * vij / Sij
                vi.append(Xij)
                Si.append(Sij)

            j = np.argmin(vi)
            K = P * H[j:j + 1, 0:3].T * np.linalg.inv(Si[j])
            P = (np.eye(3) - K * H[j:j + 1, 0:3]) * P
            P = 0.5 * (P + P.T)
            state = state + K * (z[i] - zh[j])
            print("#" + str(i) + ":", z[i] - zh[j])

        """

        V = z[:, np.newaxis] - zh

        # z = np.matrix(f"{dist_0}; {dist_1}; {dist_2}")
        # zhat = np.matrix(f"{dist_hat_0}; {dist_hat_1}; {dist_hat_2}")


        
        # H linearization
        H = np.zeros((3, 3))
        H[0, 0] = (x - L0[0, 0]) / dist_hat_0
        H[0, 1] = (y - L0[1, 0]) / dist_hat_0
        H[1, 0] = (x - L1[0, 0]) / dist_hat_1
        H[1, 1] = (y - L1[1, 0]) / dist_hat_1
        H[2, 0] = (x - L2[0, 0]) / dist_hat_2
        H[2, 1] = (y - L2[1, 0]) / dist_hat_2

        S = H * P * H.T + R
        X = V.T * np.linalg.inv(S) * V
        indices = np.argmin(X, axis=1)
        print(indices, X)

        state = np.matrix(f'{x};{y};{yaw}')

        for i in range(0, 3):
            j = int(indices[i])
            h = np.zeros((3, 3))
            h[j:j+1, 0:3] = H[j:j + 1, 0:3]
            # K = P * H[j:j + 1, 0:3].T * np.linalg.inv(H[j:j + 1, 0:3] * P * H[j:j + 1, 0:3].T + R[j, j])
            # P = (np.eye(3) - K * H[j:j + 1, 0:3]) * P
            K = P * h.T * np.linalg.inv(h * P * h.T + R)
            P = (np.eye(3) - K * h) * P
            P = 0.5 * (P + P.T)
            state = state + K * (z[i] - zh[j])
            print("#" + str(i) + ":", z[i] - zh[j])
        """
        # zhat = np.matrix(f'{z}')

        # Posterior
        # K = P * H.T * np.linalg.inv(H * P * H.T + R)
        # P_pos = (np.eye(3) - K * H) * P
        # updated_position = np.matrix(f'{x}; {y}; {yaw}') + K * (z - zhat)


        # Save to temporary variables and logged variables
        self.x = state[0, 0]
        self.y = state[1, 0]
        self.yaw = state[2, 0]
        self.P = P

        self.x_update[index] = state[0, 0]
        self.y_update[index] = state[1, 0]
        self.yaw_update[index] = state[2, 0]
        self.P_update[:, :, index] = P
        print("State diff:", state[0, 0] - x, state[1, 0] - y)

        self.dist_hat[0, index] = np.matrix(dist_hat_0)
        self.dist_hat[1, index] = np.matrix(dist_hat_1)
        self.dist_hat[2, index] = np.matrix(dist_hat_2)
        # self.dist[0, index] = np.matrix(dist_0)
        # self.dist[1, index] = np.matrix(dist_1)
        # self.dist[2, index] = np.matrix(dist_2)

    def step_calculation(self, wl, wr):
        wl = wl * pi / 30
        wr = wr * pi / 30

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
        L = np.eye(3)
        L[0, 0] = cos(yaw + rot_head)
        L[0, 1] = -trans_head * sin(yaw + rot_head)
        L[1, 0] = sin(yaw + rot_head)
        L[1, 1] = trans_head * cos(yaw + rot_head)
        L[2, 1] = 1
        L[2, 2] = 1

        A[0, 2] = - trans_head * sin(yaw + rot_head)
        A[1, 2] = trans_head * cos(yaw + rot_head)
        # P = A * P * A.T + M
        P = A * P * A.T + L * M * L.T

        P = 0.5 * (P + P.T)

        # Save to temporary and logged variables
        self.A = A
        self.P = P
        self.P_predict[:, :, index] = P

        # print(self.P)

        # Plot after the last messages
        if index == ODOM_LIMIT:
            print("Attempt #6")
            # print(self.x_update)
            # print(self.yaw_predict)
            self.plot()
            self.canvas()


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
