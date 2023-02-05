1. Change directory to the folder containing /src
2. Run: colcon build
3. Run: source install/setup.bash

# aut.710-exercise01
4. Run: ros2 run py_server_2 server
5. On another terminal: ros2 bag play rosbag2_2023_01_12-15_43_23/rosbag2_2023_01_12-15_43_23_0.db3 -r 0.5

# aut.710-exercise02-task2
4. Run: ros2 run py_server_1 task2
5. On another terminal: ros2 bag play rosbag2_2023_01_12-15_43_23/rosbag2_2023_01_12-15_43_23_0.db3 -r 5.0