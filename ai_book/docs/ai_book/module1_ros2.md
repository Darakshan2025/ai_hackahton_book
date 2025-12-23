---
title: Module 1 - The Robotic Nervous System (ROS 2)
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

## Introduction

In the realm of physical AI and humanoid robotics, the Robot Operating System 2 (ROS 2) serves as the essential "nervous system" that enables effective communication and coordination between various components of a robotic system. Just as the biological nervous system facilitates communication between different parts of the human body, ROS 2 provides the middleware infrastructure that allows different software components of a robot to communicate, share data, and coordinate actions seamlessly.

This module introduces the fundamental concepts of ROS 2, focusing on how it enables the integration of AI agents with robotic control systems. We will explore the architecture of ROS 2, its core communication patterns, and how to implement these concepts using Python. By the end of this module, you will understand how to create nodes that communicate through topics and services, how to define robot structure using URDF (Unified Robot Description Format), and how to integrate these components into simulation environments.

## 1. ROS 2 Core Concepts and Architecture

### 1.1 What is ROS 2?

The Robot Operating System 2 (ROS 2) is not an operating system in the traditional sense, but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms, from research robots to commercial products (Quigley et al., 2009).

ROS 2 evolved from the original ROS (ROS 1) to address several limitations, particularly in the areas of real-time systems, multi-robot systems, and commercial applications. The key improvements in ROS 2 include:

- **Quality of Service (QoS) settings**: Allow fine-tuning of communication behavior
- **Security features**: Built-in security architecture for safe deployment
- **Real-time support**: Better integration with real-time systems
- **Multi-robot support**: Improved capabilities for coordinating multiple robots
- **Official support for Windows and macOS**: In addition to Linux

### 1.2 Middleware Architecture

ROS 2 uses a middleware approach based on the Data Distribution Service (DDS) standard. DDS provides a publisher-subscriber communication model that enables decoupled, asynchronous communication between different components of a robotic system (DDS, 2015).

The architecture consists of:

1. **Nodes**: Processes that perform computation
2. **Topics**: Named buses over which nodes exchange messages
3. **Services**: Synchronous request/response communication
4. **Actions**: Asynchronous goal-oriented communication with feedback
5. **Parameters**: Configuration values that can be changed at runtime
6. **Launch files**: Configuration files for starting multiple nodes together

### 1.3 Client Libraries

ROS 2 supports multiple client libraries to enable development in different programming languages:

- **rclcpp**: C++ client library
- **rclpy**: Python client library (used extensively in this module)
- **rcl**: C client library (the base implementation)
- **rclc**: C client library optimized for microcontrollers
- **Other languages**: Java, Rust, Go, and others through ROS 2's architecture

## 2. ROS 2 Communication Patterns

### 2.1 Nodes

A node is the fundamental unit of computation in ROS 2. It is a process that performs computation and communicates with other nodes through topics, services, actions, and parameters. Each node is responsible for a specific task and can be developed independently, making the system modular and maintainable.

In Python, nodes are created by inheriting from the `Node` class provided by the `rclpy` library:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.2 Topics and Publishers/Subscribers

Topics in ROS 2 enable asynchronous, decoupled communication between nodes using a publisher-subscriber model. Publishers send messages to a topic, and subscribers receive messages from that topic. This pattern allows for loose coupling between nodes, as publishers do not need to know which subscribers exist, and subscribers do not need to know which publishers exist.

The communication is one-way: data flows from publishers to subscribers. This makes topics ideal for sensor data, robot state information, and other continuous streams of data.

Here's a complete example of a publisher and subscriber:

**Publisher (publisher_member_function.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Subscriber (subscriber_member_function.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.3 Services

Services provide synchronous request/response communication between nodes. When a client sends a request to a service, it waits for a response before continuing. This pattern is useful for tasks that require immediate feedback or when the result of an operation is needed before proceeding.

Services are defined using service definition files (`.srv`) that specify the request and response message types.

**Service Definition (AddTwoInts.srv):**
```
int64 a
int64 b
---
int64 sum
```

**Service Server:**
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Service Client:**
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.4 Actions

Actions provide asynchronous goal-oriented communication with feedback and status updates. They are ideal for long-running tasks where you want to monitor progress, cancel operations, or receive periodic updates on the status of a task.

Actions are defined using action definition files (`.action`) and are implemented using the `rclpy.action` module.

**Action Definition (Fibonacci.action):**
```
int32 order
---
int32[] sequence
---
int32 feedback
```

**Action Server:**
```python
from example_interfaces.action import Fibonacci
from rclpy.action import ActionServer
from rclpy.node import Node
import rclpy

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Python Integration with rclpy

### 3.1 Setting Up the Python Environment

To work with ROS 2 in Python, you need to set up your environment properly. The `rclpy` client library provides Python bindings for ROS 2 functionality.

First, ensure your ROS 2 environment is sourced:
```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
```

Then create a Python package for your ROS 2 nodes:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_pkg
cd my_robot_pkg
```

### 3.2 Creating Nodes with rclpy

The `rclpy` library provides a Pythonic interface to ROS 2 functionality. Here's a more complex example that demonstrates various ROS 2 concepts:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.robot_pose = None
        self.obstacle_distance = float('inf')
        self.target_reached = False

        self.get_logger().info('Robot controller initialized')

    def laser_callback(self, msg):
        # Process laser scan to detect obstacles
        if len(msg.ranges) > 0:
            # Get the minimum distance (front of robot)
            self.obstacle_distance = min(msg.ranges)
            self.get_logger().info(f'Obstacle distance: {self.obstacle_distance:.2f}m')

    def odom_callback(self, msg):
        # Update robot pose from odometry
        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def quaternion_to_yaw(self, quaternion):
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (quaternion.z * quaternion.w + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # Simple obstacle avoidance and navigation control
        cmd_vel = Twist()

        if self.obstacle_distance < 1.0:  # Obstacle too close
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn right to avoid obstacle
            self.get_logger().info('Avoiding obstacle')
        else:
            cmd_vel.linear.x = 0.5  # Move forward
            cmd_vel.angular.z = 0.0  # Go straight

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.3 Parameters in ROS 2

Parameters allow nodes to be configured at runtime. They can be set at launch time or changed during execution:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterExample(Node):
    def __init__(self):
        super().__init__('parameter_example')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety distance: {self.safety_distance}')

        # Set up parameter callback to handle changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.max_velocity = param.value
                self.get_logger().info(f'Updated max velocity: {self.max_velocity}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. URDF (Unified Robot Description Format)

### 4.1 Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robots in ROS. It defines the physical structure of a robot, including its links (rigid parts), joints (connections between links), visual properties, collision properties, and inertial properties.

URDF is essential for:
- Robot simulation in Gazebo and other simulators
- Robot visualization in RViz
- Kinematic analysis and inverse kinematics
- Motion planning and control

### 4.2 URDF Structure

A URDF file consists of several main elements:

1. **Links**: Represent rigid parts of the robot
2. **Joints**: Define how links connect and move relative to each other
3. **Visual**: Define how the robot looks (for visualization)
4. **Collision**: Define collision properties (for physics simulation)
5. **Inertial**: Define mass and inertial properties (for physics simulation)

### 4.3 Creating a Simple Humanoid Robot URDF

Here's an example of a simple humanoid robot URDF that includes the basic structure of a bipedal robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Joint connecting torso to head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Joints for left arm -->
  <joint name="torso_to_left_upper_arm" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <joint name="left_upper_arm_to_lower_arm" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1"/>
  </joint>

  <!-- Right Arm (similar to left) -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="torso_to_right_upper_arm" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <joint name="right_upper_arm_to_lower_arm" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/>
    </inertial>
  </link>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints for left leg -->
  <joint name="torso_to_left_upper_leg" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <joint name="left_upper_leg_to_lower_leg" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="80" velocity="1"/>
  </joint>

  <joint name="left_lower_leg_to_foot" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.175" rpy="0 0 0"/>
  </joint>

  <!-- Right Leg (similar to left) -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
      <material name="purple">
        <color rgba="0.5 0 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/>
    </inertial>
  </link>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_to_right_upper_leg" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.05 0 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <joint name="right_upper_leg_to_lower_leg" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="80" velocity="1"/>
  </joint>

  <joint name="right_lower_leg_to_foot" type="fixed">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.175" rpy="0 0 0"/>
  </joint>
</robot>
```

### 4.4 URDF with Gazebo Integration

To make the URDF work properly in Gazebo simulation, we need to add Gazebo-specific tags for physics properties, sensors, and plugins:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid_gazebo" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Red</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <!-- Add gazebo plugins for ROS 2 control -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Add sensors to the robot -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.03"/>
      </geometry>
      <material name="silver">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="torso_to_lidar" type="fixed">
    <parent link="torso"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugin for LiDAR sensor -->
  <gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1.0</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/my_robot</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## 5. Simulation Integration

### 5.1 Launch Files

Launch files in ROS 2 allow you to start multiple nodes with specific configurations. Here's an example launch file that starts a robot simulation:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('my_robot_pkg')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Robot state publisher node
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(os.path.join(pkg_share, 'urdf', 'simple_humanoid.urdf')).read()
        }]
    )

    # Joint state publisher node
    joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    # Add nodes to launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add nodes
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(joint_state_publisher_cmd)

    return ld
```

### 5.2 Robot State Publisher

The robot state publisher is a crucial node that publishes the state of all joints in the robot to the `/tf` and `/tf_static` topics, which are used by ROS 2 tools for visualization and transformation calculations:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing state
        self.timer = self.create_timer(0.1, self.publish_state)

        # Initialize joint positions
        self.joint_names = [
            'torso_to_head',
            'torso_to_left_upper_arm',
            'left_upper_arm_to_lower_arm',
            'torso_to_right_upper_arm',
            'right_upper_arm_to_lower_arm',
            'torso_to_left_upper_leg',
            'left_upper_leg_to_lower_leg',
            'torso_to_right_upper_leg',
            'right_upper_leg_to_lower_leg'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

    def publish_state(self):
        # Get current time
        current_time = self.get_clock().now().to_msg()

        # Create joint state message
        joint_state = JointState()
        joint_state.header.stamp = current_time
        joint_state.name = self.joint_names
        joint_state.position = self.joint_positions
        joint_state.velocity = self.joint_velocities
        joint_state.effort = self.joint_efforts

        # Publish joint state
        self.joint_pub.publish(joint_state)

        # Broadcast transforms
        self.broadcast_transforms(current_time)

    def broadcast_transforms(self, timestamp):
        # In a real implementation, you would calculate transforms based on joint angles
        # For this example, we'll broadcast a simple transform
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6. Integrating AI Agents with ROS 2

### 6.1 AI Agent Communication Pattern

AI agents can be integrated with ROS 2 by creating nodes that subscribe to sensor data, process it using AI algorithms, and publish commands to control the robot. Here's an example of an AI-based navigation agent:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math

class AINavigationAgent(Node):
    def __init__(self):
        super().__init__('ai_navigation_agent')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Timer for AI decision making
        self.ai_timer = self.create_timer(0.2, self.ai_decision_loop)

        # Robot state
        self.robot_pose = None
        self.laser_ranges = []
        self.current_goal = None
        self.status = "IDLE"

        self.get_logger().info('AI Navigation Agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_ranges = list(msg.ranges)
        # Filter out invalid ranges
        self.laser_ranges = [r if not math.isinf(r) else msg.range_max
                            for r in self.laser_ranges]

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.z * quaternion.w + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def ai_decision_loop(self):
        """Main AI decision making loop"""
        if self.robot_pose is None or len(self.laser_ranges) == 0:
            return

        # Simple AI decision making: obstacle avoidance + goal following
        cmd_vel = Twist()

        # Check for obstacles
        if self.has_obstacle_ahead():
            # Turn to avoid obstacle
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5
            self.status = "AVOIDING OBSTACLE"
        elif self.current_goal and not self.reached_goal():
            # Move toward goal
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = self.calculate_heading_to_goal()
            self.status = "NAVIGATING TO GOAL"
        else:
            # Stop if no goal or reached goal
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.status = "REACHED GOAL" if self.current_goal else "NO GOAL SET"

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish status
        status_msg = String()
        status_msg.data = self.status
        self.status_pub.publish(status_msg)

    def has_obstacle_ahead(self):
        """Check if there's an obstacle directly ahead"""
        if len(self.laser_ranges) == 0:
            return False

        # Check the front 30 degrees (15 degrees each side)
        front_range_start = len(self.laser_ranges) // 2 - len(self.laser_ranges) // 12
        front_range_end = len(self.laser_ranges) // 2 + len(self.laser_ranges) // 12

        # Get minimum distance in front
        front_distances = self.laser_ranges[front_range_start:front_range_end]
        min_distance = min(front_distances) if front_distances else float('inf')

        return min_distance < 1.0  # 1 meter threshold

    def calculate_heading_to_goal(self):
        """Calculate angular velocity to turn toward goal"""
        if not self.current_goal or not self.robot_pose:
            return 0.0

        # Calculate desired heading
        dx = self.current_goal['x'] - self.robot_pose['x']
        dy = self.current_goal['y'] - self.robot_pose['y']
        desired_theta = math.atan2(dy, dx)

        # Calculate error
        error = desired_theta - self.robot_pose['theta']

        # Normalize error to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        # Simple proportional controller
        return max(-0.5, min(0.5, error * 0.5))

    def reached_goal(self):
        """Check if robot has reached the goal"""
        if not self.current_goal or not self.robot_pose:
            return False

        dx = self.current_goal['x'] - self.robot_pose['x']
        dy = self.current_goal['y'] - self.robot_pose['y']
        distance = math.sqrt(dx*dx + dy*dy)

        return distance < 0.5  # 0.5 meter threshold

    def set_goal(self, x, y):
        """Set a new navigation goal"""
        self.current_goal = {'x': x, 'y': y}
        self.status = f"GOAL SET TO ({x}, {y})"

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AINavigationAgent()

    # Example: set a goal after initialization
    ai_agent.set_goal(5.0, 5.0)

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 6.2 Advanced AI Integration with Behavior Trees

For more complex AI behaviors, we can implement behavior trees that allow for hierarchical decision making:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

class BehaviorNode:
    """Base class for behavior tree nodes"""
    def __init__(self, name):
        self.name = name
        self.status = "IDLE"

    def tick(self):
        """Execute the behavior and return status"""
        pass

class SequenceNode(BehaviorNode):
    """Execute children in sequence until one fails"""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status == "FAILURE":
                return "FAILURE"
        return "SUCCESS"

class SelectorNode(BehaviorNode):
    """Execute children in sequence until one succeeds"""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status == "SUCCESS":
                return "SUCCESS"
        return "FAILURE"

class CheckObstacleNode(BehaviorNode):
    """Check if there's an obstacle ahead"""
    def __init__(self, name, robot_node):
        super().__init__(name)
        self.robot_node = robot_node

    def tick(self):
        if self.robot_node.has_obstacle_ahead():
            return "SUCCESS"
        return "FAILURE"

class MoveForwardNode(BehaviorNode):
    """Move robot forward"""
    def __init__(self, name, robot_node):
        super().__init__(name)
        self.robot_node = robot_node

    def tick(self):
        self.robot_node.move_forward()
        return "SUCCESS"

class TurnNode(BehaviorNode):
    """Turn robot to avoid obstacle"""
    def __init__(self, name, robot_node):
        super().__init__(name)
        self.robot_node = robot_node

    def tick(self):
        self.robot_node.turn_away_from_obstacle()
        return "SUCCESS"

class AIBehaviorTreeAgent(Node):
    def __init__(self):
        super().__init__('ai_behavior_tree_agent')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Robot state
        self.laser_ranges = []

        # Create behavior tree
        self.create_behavior_tree()

        # Timer for behavior execution
        self.timer = self.create_timer(0.1, self.execute_behavior)

        self.get_logger().info('AI Behavior Tree Agent initialized')

    def create_behavior_tree(self):
        """Create the behavior tree structure"""
        # Check obstacle then either move forward or turn
        obstacle_sequence = SequenceNode("check_and_move", [
            CheckObstacleNode("check_obstacle", self),
            TurnNode("turn_away", self)
        ])

        # Main behavior: if obstacle turn, otherwise move forward
        self.root = SelectorNode("main_behavior", [
            obstacle_sequence,
            MoveForwardNode("move_forward", self)
        ])

    def laser_callback(self, msg):
        self.laser_ranges = list(msg.ranges)

    def execute_behavior(self):
        """Execute the behavior tree"""
        status = self.root.tick()
        self.get_logger().info(f'Behavior tree status: {status}')

    def has_obstacle_ahead(self):
        """Check if there's an obstacle directly ahead"""
        if len(self.laser_ranges) == 0:
            return False

        # Check the front 30 degrees
        front_range_start = len(self.laser_ranges) // 2 - len(self.laser_ranges) // 12
        front_range_end = len(self.laser_ranges) // 2 + len(self.laser_ranges) // 12

        front_distances = self.laser_ranges[front_range_start:front_range_end]
        min_distance = min(front_distances) if front_distances else float('inf')

        return min_distance < 1.0

    def move_forward(self):
        """Send command to move robot forward"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def turn_away_from_obstacle(self):
        """Send command to turn robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.5
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AIBehaviorTreeAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Practical Exercises

### Exercise 1: Basic Publisher-Subscriber Communication

Create a simple publisher that publishes temperature readings and a subscriber that logs these readings:

**Temperature Publisher:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        self.publisher_ = self.create_publisher(Float32, 'temperature', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float32()
        # Simulate temperature reading between 15 and 35 degrees Celsius
        msg.data = 15.0 + random.random() * 20.0
        self.publisher_.publish(msg)
        self.get_logger().info('Temperature: %.2f°C' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    temp_publisher = TemperaturePublisher()
    rclpy.spin(temp_publisher)
    temp_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Temperature Subscriber:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class TemperatureSubscriber(Node):
    def __init__(self):
        super().__init__('temperature_subscriber')
        self.subscription = self.create_subscription(
            Float32,
            'temperature',
            self.temperature_callback,
            10)
        self.subscription  # prevent unused variable warning

    def temperature_callback(self, msg):
        temp = msg.data
        if temp > 30.0:
            self.get_logger().warn('High temperature alert: %.2f°C' % temp)
        elif temp < 20.0:
            self.get_logger().warn('Low temperature alert: %.2f°C' % temp)
        else:
            self.get_logger().info('Normal temperature: %.2f°C' % temp)

def main(args=None):
    rclpy.init(args=args)
    temp_subscriber = TemperatureSubscriber()
    rclpy.spin(temp_subscriber)
    temp_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 2: Creating a URDF Robot Model

Create a URDF file for a simple wheeled robot with differential drive:

```xml
<?xml version="1.0"?>
<robot name="differential_drive_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Caster wheel -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.175 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.175 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="caster_wheel_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="-0.2 0 -0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

## 8. Summary and Key Takeaways

In this module, we've explored the fundamental concepts of ROS 2 as the "nervous system" of robotic systems. We've covered:

1. **ROS 2 Architecture**: Understanding the middleware approach based on DDS and the core components including nodes, topics, services, and actions.

2. **Communication Patterns**: Learning how to implement publisher-subscriber, service, and action patterns in Python using the rclpy library.

3. **Python Integration**: Developing ROS 2 nodes in Python with proper parameter handling, logging, and error management.

4. **URDF Creation**: Building robot descriptions using URDF, including links, joints, visual, collision, and inertial properties.

5. **AI Integration**: Connecting AI agents to ROS 2 systems for robot control and decision making.

6. **Simulation Integration**: Understanding how to integrate robots with simulation environments using launch files and robot state publishers.

The concepts learned in this module form the foundation for all subsequent modules in the Physical AI & Humanoid Robotics course. The ability to create nodes that communicate effectively through ROS 2 is essential for building complex robotic systems that integrate perception, planning, and control components.

## References

DDS (Data Distribution Service). (2015). *OMG DDS Specification*. Object Management Group.

Kuffner, J. (2013). *The Open Source Robot Operating System (ROS)*. In: Siciliano, B., Khatib, O. (eds) Springer Handbook of Robotics. Springer, Cham. https://doi.org/10.1007/978-3-319-32552-1_43

Quigley, M., Gerkey, B., & Smart, W. D. (2009). Programming robots with ROS: A practical introduction to the Robot Operating System. *IEEE Intelligent Systems*, 24(1), 68-75.

ROS Documentation. (2023). *ROS 2 Documentation*. Retrieved from https://docs.ros.org/en/humble/

Schoellig, A., & D'Andrea, R. (2012). Iterative learning control for trajectory tracking of multi-vehicle systems. *Proceedings of the IEEE Conference on Decision and Control*, 6612-6617.

Tully, S. (2017). *Programming Robots with ROS: A Practical Introduction to the Robot Operating System*. O'Reilly Media.

## Glossary

- **DDS (Data Distribution Service)**: A middleware protocol that provides a publisher-subscriber communication model for distributed systems.
- **Node**: A process that performs computation in ROS 2.
- **Topic**: A named bus over which nodes exchange messages using publisher-subscriber pattern.
- **Service**: Synchronous request/response communication between nodes.
- **Action**: Asynchronous goal-oriented communication with feedback and status updates.
- **URDF (Unified Robot Description Format)**: An XML-based format for describing robot structure.
- **rclpy**: Python client library for ROS 2.
- **Robot State Publisher**: A node that publishes joint state information to TF transforms.
- **TF (Transforms)**: A system for tracking coordinate frames in time.