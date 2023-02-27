# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 03 - Problem 01
# Hoang Pham, Nadun Ranasinghe

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


class Server(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Temporary variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.L0 = np.matrix('10; 0')
        self.L1 = np.matrix('10; 10')
        self.L2 = np.matrix('0; 10')
        self.P = np.eye(9) * 0.1
        self.Q = np.zeros((9, 9))
        self.A = np.eye(9)
        self.ts1 = 0
        self.ts2 = 0

        # Logged variables
        self.x_predict = np.zeros(ODOM_LIMIT + 1)
        self.y_predict = np.zeros(ODOM_LIMIT + 1)
        self.yaw_predict = np.zeros(ODOM_LIMIT + 1)
        self.P_predict = np.zeros((9, 9, ODOM_LIMIT + 1))
        self.P_predict[:, :, 0] = self.P
        self.yaw_predict[0] = self.yaw

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

        self.tspan1 = [0]
        self.tspan2 = [0]

        self.wl = np.zeros(ODOM_LIMIT + 1)
        self.wr = np.zeros(ODOM_LIMIT + 1)

        self.measure_hat = np.zeros((6, SENSOR_LIMIT + 1))
        self.measure = np.zeros((6, SENSOR_LIMIT + 1))

        # Counter for indexing
        self.odom_count = 0
        self.sensor_count = 0

        # Subscriber
        self.sensor_subscription = self.create_subscription(JointState, '/landmarks', self.sensor_callback, 20)
        self.odom_subscription = self.create_subscription(JointState, '/joint_states', self.odom_callback, 20)
        self.sensor_subscription  # prevent unused variable warning
        self.odom_subscription

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
        plt.plot(self.L0[0, 0], self.L0[1, 0], linestyle='None', marker="o", label='Landmark 1')
        plt.plot(self.L1[0, 0], self.L1[1, 0], linestyle='None', marker="o", label='Landmark 2')
        plt.plot(self.L2[0, 0], self.L2[1, 0], linestyle='None', marker="o", label='Landmark 3')

        # Initialization for animation and text labels
        predicted_coordinates = []
        updated_coordinates = []
        predict, = ax.plot(0, 0, 'b+', markersize=1, label="Predicted position")
        update, = ax.plot(0, 0, 'r--', linewidth=1, label="Updated position")
        ax.legend()
        text0 = plt.text(0.2, 0.1, "Time: 0.0", fontsize=10, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes)
        text1 = plt.text(0.2, 0.2, "Coordinate: 0.0, 0.0", fontsize=10, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes)
        text2 = plt.text(0.2, 0.3, "Predicted Meas: 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0", fontsize=10,
                         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        text3 = plt.text(0.2, 0.4, "Measured Meas: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0", fontsize=10,
                         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        # Iterate through time array to plot the robot position and update state and measurement
        for index in range(0, ODOM_LIMIT):
            # Predicted position
            predicted_coordinates.append((self.x_predict[index], self.y_predict[index]))
            x_predict, y_predict = zip(*predicted_coordinates)
            predict.set_xdata(x_predict)
            predict.set_ydata(y_predict)
            text0.set_text(f"Time: {round(self.tspan1[index], 2)}")
            text1.set_text(f"Coordinate: {round(self.x_predict[index], 2)}, {round(self.y_predict[index], 2)}")

            if index % 3 == 0:
                # Updated position
                i = int(index / 3)
                updated_coordinates.append((self.x_update[i], self.y_update[i]))

                text2.set_text(f"Predicted: {round(self.measure_hat[0, i], 2)}, {round(self.measure_hat[1, i], 2)}, "
                               f"{round(self.measure_hat[2, i], 2)}, {round(self.measure_hat[3, i], 2)}, "
                               f"{round(self.measure_hat[4, i], 2)}, {round(self.measure_hat[5, i], 2)}")
                text3.set_text(f"Measured: {round(self.measure[0, i], 2)}, {round(self.measure[1, i], 2)}, "
                               f"{round(self.measure[2, i], 2)}, {round(self.measure[3, i], 2)}, "
                               f"{round(self.measure[4, i], 2)}, {round(self.measure[5, i], 2)}")

                x_update, y_update = zip(*updated_coordinates)
                update.set_xdata(x_update)
                update.set_ydata(y_update)

                lambda_, v = np.linalg.eig(self.P_update[0:3, 0:3, i])
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
        tspan1 = np.array(self.tspan1)
        tspan2 = np.array(self.tspan2)

        # Plot the wheels velocity
        fig1 = plt.figure(1)
        plt.plot(tspan1, self.wl, label="Left wheel")
        plt.plot(tspan1, self.wr, label="Right wheel")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.title("Wheels velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (rad/s)")

        # Plot the predicted and updated covariances
        fig2 = plt.figure(2)
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

        fig3 = plt.figure(3)
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
        fig4 = plt.figure(4)
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

        fig5 = plt.figure(5)
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

        fig6, axs6 = plt.subplots(2)
        p_x0 = np.sqrt(self.P_update[3, 3, :])
        p_y0 = np.sqrt(self.P_update[4, 4, :])
        p_x1 = np.sqrt(self.P_update[5, 5, :])
        p_y1 = np.sqrt(self.P_update[6, 6, :])
        p_x2 = np.sqrt(self.P_update[7, 7, :])
        p_y2 = np.sqrt(self.P_update[8, 8, :])

        # Plot the landmark coordinate with +/- standard deviation
        axs6[0].plot(tspan2, self.L0_update[0, :], label="LM1")
        axs6[0].plot(tspan2, self.L0_update[0, :] + p_x0, label="LM1 + Sigma")
        axs6[0].plot(tspan2, self.L0_update[0, :] - p_x0, label="LM1 - Sigma")
        axs6[0].plot(tspan2, self.L1_update[0, :], label="LM2")
        axs6[0].plot(tspan2, self.L1_update[0, :] + p_x1, label="LM2 + Sigma")
        axs6[0].plot(tspan2, self.L1_update[0, :] - p_x1, label="LM2 - Sigma")
        axs6[0].plot(tspan2, self.L2_update[0, :], label="LM3")
        axs6[0].plot(tspan2, self.L2_update[0, :] + p_x2, label="LM3 + Sigma")
        axs6[0].plot(tspan2, self.L2_update[0, :] - p_x2, label="LM3 - Sigma")
        axs6[0].set_title("X versus Time")
        axs6[0].set_xlabel("Time (s)")
        axs6[0].set_ylabel("Value")
        axs6[0].grid(color='gray', linestyle='-', linewidth=0.5)

        axs6[1].plot(tspan2, self.L0_update[1, :])
        axs6[1].plot(tspan2, self.L0_update[1, :] + p_y0)
        axs6[1].plot(tspan2, self.L0_update[1, :] - p_y0)
        axs6[1].plot(tspan2, self.L1_update[1, :])
        axs6[1].plot(tspan2, self.L1_update[1, :] + p_y1)
        axs6[1].plot(tspan2, self.L1_update[1, :] - p_y1)
        axs6[1].plot(tspan2, self.L2_update[1, :])
        axs6[1].plot(tspan2, self.L2_update[1, :] + p_y2)
        axs6[1].plot(tspan2, self.L2_update[1, :] - p_y2)
        axs6[1].set_title("Y versus Time")
        axs6[1].set_xlabel("Time (s)")
        axs6[1].set_ylabel("Value")
        axs6[1].grid(color='gray', linestyle='-', linewidth=0.5)

        # Finish the plot
        fig1.legend()
        fig2.legend()
        fig3.legend()
        fig4.legend()
        fig5.legend()
        fig6.legend()

        # Show the plot but continue the thread
        plt.show(block=False)

    def odom_callback(self, msg):
        # Increase the counter and process the prediction
        self.odom_count += 1
        # self.get_logger().info('Odom says: #%d "%s"' % (self.odom_count, msg.velocity))

        # Calculate dt and save the timestamp
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e+9

        if self.odom_count == 1:
            self.tspan1.append(0.1)
        else:
            self.tspan1.append(ts - self.ts1 + self.tspan1[-1])

        self.ts1 = ts

        # Prediction
        self.step_calculation(msg.velocity[1], msg.velocity[0], round(self.tspan1[-1] - self.tspan1[-2], 4))

    def sensor_callback(self, msg):
        # Increase the counter and process the update
        self.sensor_count += 1
        # self.get_logger().info('Sensor says: #%d "%s %s"' % (self.sensor_count, msg.position, msg.velocity))

        # Calculate dt and save the timestamp
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e+9

        if self.ts2 == 0:
            self.tspan2.append(msg.header.stamp.nanosec / 1e+9)
        else:
            self.tspan2.append(ts - self.ts2 + self.tspan2[-1])
        self.ts2 = ts

        # Update
        self.update(msg.position[0], msg.position[1], msg.position[2],
                    msg.velocity[0], msg.velocity[1], msg.velocity[2])

    def update(self, dist_0, dist_1, dist_2, bear_0, bear_1, bear_2):
        # Extract the last value of P, x, y, psi, L0, L1, L2
        index = self.sensor_count
        P_neg = self.P
        x = self.x
        y = self.y
        yaw = self.yaw
        L0 = self.L0
        L1 = self.L1
        L2 = self.L2

        # Convert degree to radian
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

        # Create measurement vector
        z = np.matrix(f"{dist_0};{dist_1};{dist_2};{bear_0};{bear_1};{bear_2}")
        zhat = np.matrix(f"{dist_hat_0};{dist_hat_1};{dist_hat_2};{bear_hat_0};{bear_hat_1};{bear_hat_2}")

        # H linearization
        dx0 = x - L0[0, 0]
        dx1 = x - L1[0, 0]
        dx2 = x - L2[0, 0]
        dy0 = y - L0[1, 0]
        dy1 = y - L1[1, 0]
        dy2 = y - L2[1, 0]

        H = np.zeros((6, 9))
        H[0, :] = np.matrix([dx0,  dy0, 0, -dx0, -dy0, 0, 0, 0, 0]) / dist_hat_0
        H[1, :] = np.matrix([dx1,  dy1, 0, 0, 0, -dx1, -dy1, 0, 0]) / dist_hat_1
        H[2, :] = np.matrix([dx2,  dy2, 0, 0, 0, 0, 0, -dx2, -dy2]) / dist_hat_2
        H[3, :] = np.matrix([dy0, -dx0, -(dist_hat_0 ** 2), -dy0, dx0, 0, 0, 0, 0]) / (dist_hat_0 ** 2)
        H[4, :] = np.matrix([dy1, -dx1, -(dist_hat_1 ** 2), 0, 0, -dy1, dx1, 0, 0]) / (dist_hat_1 ** 2)
        H[5, :] = np.matrix([dy2, -dx2, -(dist_hat_2 ** 2), 0, 0, 0, 0, -dy2, dx2]) / (dist_hat_2 ** 2)

        # Posterior
        K = P_neg @ H.T @ np.linalg.inv(H @ P_neg @ H.T + R)
        # print(H)
        P = (np.eye(9) - K @ H) @ P_neg
        P_pos = 0.5 * (P + P.T)
        old_state = np.matrix(f'{x};{y};{yaw};{L0[0, 0]}; {L0[1, 0]};{L1[0, 0]}; {L1[1, 0]}; {L2[0, 0]}; {L2[1, 0]}')
        new_state = old_state + K @ (z - zhat)

        # Save to temporary variables and logged variables
        self.x = new_state[0, 0]
        self.y = new_state[1, 0]
        self.yaw = new_state[2, 0]
        self.P = P_pos
        self.L0 = new_state[3:5, 0]
        self.L1 = new_state[5:7, 0]
        self.L2 = new_state[7:9, 0]

        # Save new location coordinates
        self.L0_update[:, index:index + 1] = new_state[3:5, 0]
        self.L1_update[:, index:index + 1] = new_state[5:7, 0]
        self.L2_update[:, index:index + 1] = new_state[7:9, 0]

        # Save new robot state and covariances matrix
        self.x_update[index] = new_state[0, 0]
        self.y_update[index] = new_state[1, 0]
        self.yaw_update[index] = new_state[2, 0]
        self.P_update[:, :, index] = P_pos
        # print("Measure Diff:", dist_0 - dist_hat_0, dist_1 - dist_hat_1, dist_2 - dist_hat_2,
        #                        bear_0 - bear_hat_0, bear_1 - bear_hat_1, bear_2 - bear_hat_2)

        # Save the measurement of the current timestamp
        self.measure_hat[:, index:index + 1] = zhat
        self.measure[:, index:index + 1] = z

    def step_calculation(self, wl, wr, interval=INTERVAL):
        rot_head = interval * Radius * (wr - wl) / (2 * D)
        trans_head = interval * Radius * (wr + wl) / 2

        # Extract the last value of the robot state, A, and P matrix
        index = self.odom_count
        x = self.x
        y = self.y
        yaw = self.yaw
        A = self.A
        P = self.P

        # Save the wheels speed
        self.wl[index] = wl
        self.wr[index] = wr

        # Function f
        x_predict = x + trans_head * cos(yaw + rot_head)
        y_predict = y + trans_head * sin(yaw + rot_head)
        yaw_predict = yaw + rot_head * 2

        # Calculate the predicted positions and save to temporary and logged variables
        self.x_predict[index] = x_predict
        self.y_predict[index] = y_predict
        self.yaw_predict[index] = yaw_predict

        self.x = x_predict
        self.y = y_predict
        self.yaw = yaw_predict

        # Linearization
        L = np.zeros((9, 3))
        L[0, :] = np.matrix([cos(yaw + rot_head), -trans_head * sin(yaw + rot_head), 0])
        L[1, :] = np.matrix([sin(yaw + rot_head), trans_head * cos(yaw + rot_head), 0])
        L[2, :] = np.matrix([0, 1, 1])

        A[0, 2] = - trans_head * sin(yaw + rot_head)
        A[1, 2] = trans_head * cos(yaw + rot_head)

        # Calculate new covariances matrix
        P = A @ P @ A.T + L @ M @ L.T
        P = 0.5 * (P + P.T)

        # Save to temporary and logged variables
        self.A = A
        self.P = P
        self.P_predict[:, :, index] = P

        # Plot after the last messages
        if index == ODOM_LIMIT:
            self.plot()
            self.canvas()


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
