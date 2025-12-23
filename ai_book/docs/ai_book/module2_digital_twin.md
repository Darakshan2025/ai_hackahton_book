---
title: Module 2 - The Digital Twin (Gazebo & Unity)
sidebar_position: 2
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Introduction

In the realm of physical AI and humanoid robotics, the concept of a "Digital Twin" represents a virtual replica of a physical system that enables real-time simulation, testing, and optimization. A digital twin bridges the gap between the physical and digital worlds by creating an accurate virtual representation that mirrors the behavior, properties, and responses of its physical counterpart. In the context of humanoid robotics, digital twins serve as essential tools for testing control algorithms, validating sensor data, training AI agents, and ensuring safe operation before deployment to real hardware.

This module explores the creation and utilization of digital twins using two complementary simulation environments: Gazebo for physics-based simulation and Unity for high-fidelity visualization. We will examine how these platforms work together to create comprehensive digital twins that accurately represent humanoid robots, their environments, and their interactions. By the end of this module, you will understand how to create realistic simulation environments, implement sensor models, and connect these simulations to AI agents for perception and control.

## 1. Digital Twin Fundamentals

### 1.1 What is a Digital Twin?

A digital twin is a virtual representation of a physical system that spans its lifecycle, is updated from real-time data, and uses simulation, machine learning, and reasoning to help decision-making (Grieves & Vickers, 2017). In robotics, a digital twin encompasses:

- **Physical Model**: Accurate representation of the robot's structure, kinematics, and dynamics
- **Behavioral Model**: Simulation of the robot's responses to various inputs and environmental conditions
- **Sensor Model**: Virtual sensors that produce data similar to real-world sensors
- **Environment Model**: Accurate representation of the robot's operating environment
- **Data Integration**: Real-time synchronization between physical and virtual systems

### 1.2 Digital Twin in Robotics Applications

Digital twins in robotics serve multiple purposes:

1. **Development and Testing**: Safe environment for testing control algorithms without risk to physical hardware
2. **Training**: Environment for training AI agents using synthetic data
3. **Validation**: Verification of algorithms before deployment to real robots
4. **Optimization**: Parameter tuning and performance optimization in simulation
5. **Safety**: Risk assessment and safety validation before real-world deployment

### 1.3 Simulation Platforms for Digital Twins

For humanoid robotics, we primarily use two complementary simulation platforms:

- **Gazebo**: Physics-based simulation engine optimized for realistic physics, collision detection, and sensor simulation
- **Unity**: High-fidelity rendering engine optimized for realistic visuals, lighting, and user interaction

Together, these platforms provide the physics accuracy of Gazebo combined with the visual fidelity of Unity, creating comprehensive digital twins suitable for advanced robotics applications.

## 2. Gazebo Physics Simulation

### 2.1 Gazebo Architecture and Components

Gazebo is a physics-based simulation engine that enables the testing of robots in realistic environments. The architecture consists of several key components:

1. **Physics Engine**: Supports multiple physics engines including ODE, Bullet, Simbody, and DART
2. **Sensor Simulation**: Realistic simulation of various sensor types (LiDAR, cameras, IMUs, etc.)
3. **Rendering Engine**: Visualization of the simulation environment
4. **Plugin System**: Extensible architecture for custom sensors, controllers, and interfaces
5. **ROS Integration**: Seamless integration with ROS/ROS 2 for robotics applications

### 2.2 Physics Simulation Fundamentals

Gazebo simulates realistic physics by modeling the fundamental forces and interactions that govern the physical world:

**Gravity Simulation**: Gazebo models gravitational forces with customizable parameters:
```
<world>
  <gravity>0 0 -9.8</gravity>
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
  </physics>
</world>
```

**Collision Detection**: Gazebo uses multiple collision detection algorithms to accurately model interactions between objects. The system supports:
- Primitive shapes (boxes, spheres, cylinders)
- Mesh-based collision objects
- Heightmap-based terrain collision

**Material Properties**: Accurate modeling of material properties including:
- Friction coefficients (static and dynamic)
- Restitution coefficients (bounciness)
- Density and mass properties

### 2.3 Creating Gazebo Worlds

A Gazebo world file is an XML file that defines the simulation environment. Here's an example world file with physics properties:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="digital_twin_world">
    <!-- Physics Engine Configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Example Object: Box with Physics Properties -->
    <model name="physics_box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.083</iyy>
            <iyz>0</iyz>
            <izz>0.083</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.5</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.3 0.3 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Example Humanoid Robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### 2.4 Sensor Simulation in Gazebo

Gazebo provides realistic simulation of various sensor types essential for humanoid robotics:

**LiDAR Simulation**: Simulates laser range finders with realistic noise models and performance characteristics:

```xml
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
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

**Depth Camera Simulation**: Provides both color and depth information for 3D perception:

```xml
<sensor name="depth_camera" type="depth">
  <camera name="depth_cam">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>image_raw:=/depth/image_raw</remapping>
      <remapping>camera_info:=/depth/camera_info</remapping>
    </ros>
  </plugin>
</sensor>
```

**IMU Simulation**: Simulates Inertial Measurement Units with realistic noise characteristics:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>~/out:=imu</remapping>
    </ros>
    <initial_orientation_as_reference>false</initial_orientation_as_reference>
  </plugin>
</sensor>
```

### 2.5 Gazebo ROS Integration

Gazebo integrates seamlessly with ROS 2 through the Gazebo ROS packages, enabling bidirectional communication between simulation and ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class GazeboRobotInterface(Node):
    def __init__(self):
        super().__init__('gazebo_robot_interface')

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers for sensor data from Gazebo
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.laser_data = None
        self.imu_data = None
        self.odom_data = None
        self.obstacle_detected = False

        self.get_logger().info('Gazebo Robot Interface initialized')

    def laser_callback(self, msg):
        """Process laser scan data from Gazebo"""
        self.laser_data = msg.ranges
        # Check for obstacles in front of robot
        if self.laser_data:
            front_ranges = self.laser_data[len(self.laser_data)//2-30:len(self.laser_data)//2+30]
            min_range = min(front_ranges) if front_ranges else float('inf')
            self.obstacle_detected = min_range < 1.0  # 1 meter threshold

    def imu_callback(self, msg):
        """Process IMU data from Gazebo"""
        self.imu_data = {
            'linear_acceleration': [msg.linear_acceleration.x,
                                   msg.linear_acceleration.y,
                                   msg.linear_acceleration.z],
            'angular_velocity': [msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z],
            'orientation': [msg.orientation.x,
                           msg.orientation.y,
                           msg.orientation.z,
                           msg.orientation.w]
        }

    def odom_callback(self, msg):
        """Process odometry data from Gazebo"""
        self.odom_data = {
            'position': [msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z],
            'velocity': [msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z]
        }

    def control_loop(self):
        """Main control loop"""
        cmd_vel = Twist()

        if self.obstacle_detected:
            # Obstacle avoidance behavior
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn right
            self.get_logger().info('Obstacle detected - turning')
        else:
            # Move forward
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = 0.0
            self.get_logger().info('Moving forward')

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboRobotInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Unity Environment Creation

### 3.1 Unity in Robotics Context

Unity is a powerful real-time 3D development platform that excels in creating high-fidelity visual environments. While Gazebo focuses on physics accuracy, Unity provides photorealistic rendering capabilities essential for:

- High-fidelity visualization for human operators
- Synthetic data generation for AI training
- Virtual reality interfaces for robot teleoperation
- Training environments with realistic lighting and materials

### 3.2 Unity Physics System

Unity's physics system is based on the NVIDIA PhysX engine, which provides:

- **Rigid Body Dynamics**: Accurate simulation of rigid body motion and collisions
- **Soft Body Physics**: Simulation of deformable objects
- **Fluid Simulation**: Advanced fluid dynamics
- **Cloth Simulation**: Realistic fabric and material simulation

Unity physics components for robotics applications:

```csharp
using UnityEngine;

public class RobotJoint : MonoBehaviour
{
    public ConfigurableJoint joint;
    public float targetPosition = 0f;
    public float stiffness = 100f;
    public float damping = 10f;

    void Start()
    {
        joint = GetComponent<ConfigurableJoint>();
        SetupJoint();
    }

    void SetupJoint()
    {
        // Configure joint limits
        SoftJointLimit limit = new SoftJointLimit();
        limit.limit = 45f; // 45 degrees
        joint.linearLimit = limit;

        // Configure spring for compliance
        JointSpring spring = new JointSpring();
        spring.spring = stiffness;
        spring.damper = damping;
        joint.spring = spring;
    }

    void Update()
    {
        // Apply target position
        JointDrive drive = new JointDrive();
        drive.positionSpring = stiffness;
        drive.positionDamper = damping;
        drive.maximumForce = 300f;
        joint.slerpDrive = drive;
    }
}
```

### 3.3 Creating High-Fidelity Environments

Unity excels at creating visually rich environments with realistic materials and lighting:

**Physically-Based Rendering (PBR)**: Unity supports PBR materials that respond to light realistically:

```csharp
using UnityEngine;

public class MaterialManager : MonoBehaviour
{
    [Header("PBR Properties")]
    public Texture2D albedoMap;
    public Texture2D normalMap;
    public Texture2D metallicMap;
    public Texture2D roughnessMap;

    [Header("Material Properties")]
    public float metallic = 0.5f;
    public float smoothness = 0.5f;

    private Renderer renderer;

    void Start()
    {
        renderer = GetComponent<Renderer>();
        UpdateMaterial();
    }

    void UpdateMaterial()
    {
        Material material = renderer.material;

        if (albedoMap != null) material.mainTexture = albedoMap;
        material.SetTexture("_BumpMap", normalMap);
        material.SetTexture("_MetallicGlossMap", metallicMap);
        material.SetTexture("_SmoothnessTexture", roughnessMap);

        material.SetFloat("_Metallic", metallic);
        material.SetFloat("_Glossiness", smoothness);
    }
}
```

**Dynamic Lighting**: Unity supports various lighting models including real-time and baked lighting:

```csharp
using UnityEngine;

public class DynamicLighting : MonoBehaviour
{
    public Light mainLight;
    public AnimationCurve intensityCurve;
    public float cycleDuration = 10f;

    private float startTime;

    void Start()
    {
        startTime = Time.time;
        mainLight = GetComponent<Light>();
    }

    void Update()
    {
        float elapsed = Time.time - startTime;
        float progress = (elapsed % cycleDuration) / cycleDuration;
        float intensity = intensityCurve.Evaluate(progress);

        mainLight.intensity = intensity;
    }
}
```

### 3.4 Unity Robotics Integration

Unity provides the Unity Robotics Hub for integration with ROS 2:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopic = "/cmd_vel";

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        // Example: Publish transform data to ROS
        PublishTransform();
    }

    void PublishTransform()
    {
        // Create and publish transform message
        var transformMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Pose();
        transformMsg.position.x = transform.position.x;
        transformMsg.position.y = transform.position.y;
        transformMsg.position.z = transform.position.z;

        transformMsg.orientation.x = transform.rotation.x;
        transformMsg.orientation.y = transform.rotation.y;
        transformMsg.orientation.z = transform.rotation.z;
        transformMsg.orientation.w = transform.rotation.w;

        ros.Publish(robotTopic, transformMsg);
    }

    // Subscribe to ROS messages
    void OnEnable()
    {
        ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Twist>(robotTopic, ReceiveTwist);
    }

    void ReceiveTwist(Twist cmd)
    {
        // Apply received velocity commands to robot
        Vector3 linearVelocity = new Vector3(cmd.linear.x, cmd.linear.y, cmd.linear.z);
        Vector3 angularVelocity = new Vector3(cmd.angular.x, cmd.angular.y, cmd.angular.z);

        // Apply to robot's rigidbody or movement system
        ApplyVelocity(linearVelocity, angularVelocity);
    }

    void ApplyVelocity(Vector3 linear, Vector3 angular)
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = linear;
            rb.angularVelocity = angular;
        }
    }
}
```

### 3.5 Sensor Simulation in Unity

Unity can simulate various sensors for robotics applications:

**Camera Simulation**: Unity cameras can simulate RGB, depth, and semantic segmentation:

```csharp
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    public Camera camera;
    public int width = 640;
    public int height = 480;
    public float updateRate = 30f;

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;
        SetupCamera();
        StartCoroutine(PublishImage());
    }

    void SetupCamera()
    {
        camera = GetComponent<Camera>();
        renderTexture = new RenderTexture(width, height, 24);
        camera.targetTexture = renderTexture;

        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    IEnumerator PublishImage()
    {
        while (true)
        {
            yield return new WaitForSeconds(1f / updateRate);
            PublishCameraImage();
        }
    }

    void PublishCameraImage()
    {
        // Copy render texture to texture2D
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Convert to ROS Image message
        ImageMsg imageMsg = new ImageMsg();
        imageMsg.header = new std_msgs.Header();
        imageMsg.header.stamp = new builtin_interfaces.Time();
        imageMsg.header.frame_id = "camera_frame";

        imageMsg.height = (uint)height;
        imageMsg.width = (uint)width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(width * 3); // 3 bytes per pixel for RGB

        // Convert texture to bytes
        byte[] imageData = texture2D.GetRawTextureData<byte>();
        imageMsg.data = imageData;

        ros.Publish("/unity_camera/image_raw", imageMsg);
    }
}
```

## 4. Sensor Simulation Implementation

### 4.1 LiDAR Simulation in Unity

Creating realistic LiDAR simulation in Unity requires raycasting techniques:

```csharp
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityLidar : MonoBehaviour
{
    [Header("Lidar Configuration")]
    public int rayCount = 360;
    public float range = 30f;
    public float angleMin = -Mathf.PI;
    public float angleMax = Mathf.PI;
    public float updateRate = 10f;

    private ROSConnection ros;
    private List<float> ranges;

    void Start()
    {
        ros = ROSConnection.instance;
        ranges = new List<float>(new float[rayCount]);
        StartCoroutine(PublishLidarData());
    }

    IEnumerator PublishLidarData()
    {
        while (true)
        {
            ScanEnvironment();
            PublishLaserScan();
            yield return new WaitForSeconds(1f / updateRate);
        }
    }

    void ScanEnvironment()
    {
        for (int i = 0; i < rayCount; i++)
        {
            float angle = Mathf.Lerp(angleMin, angleMax, (float)i / rayCount);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), out hit, range))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = range; // No obstacle detected
            }
        }
    }

    void PublishLaserScan()
    {
        LaserScanMsg scanMsg = new LaserScanMsg();
        scanMsg.header = new std_msgs.Header();
        scanMsg.header.stamp = new builtin_interfaces.Time();
        scanMsg.header.frame_id = "lidar_frame";

        scanMsg.angle_min = angleMin;
        scanMsg.angle_max = angleMax;
        scanMsg.angle_increment = (angleMax - angleMin) / rayCount;
        scanMsg.time_increment = 0;
        scanMsg.scan_time = 1f / updateRate;
        scanMsg.range_min = 0.1f;
        scanMsg.range_max = range;

        // Convert ranges to ROS format
        scanMsg.ranges = ranges.ToArray();

        ros.Publish("/unity_lidar/scan", scanMsg);
    }
}
```

### 4.2 IMU Simulation in Unity

Simulating IMU data in Unity involves tracking acceleration and angular velocity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityIMU : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float noiseLevel = 0.01f;
    public float updateRate = 100f;

    private ROSConnection ros;
    private Rigidbody attachedRigidbody;
    private Vector3 lastVelocity;
    private float lastTime;

    void Start()
    {
        ros = ROSConnection.instance;
        attachedRigidbody = GetComponent<Rigidbody>();
        lastTime = Time.time;

        StartCoroutine(PublishIMUData());
    }

    IEnumerator PublishIMUData()
    {
        while (true)
        {
            PublishIMUReading();
            yield return new WaitForSeconds(1f / updateRate);
        }
    }

    void PublishIMUReading()
    {
        ImuMsg imuMsg = new ImuMsg();
        imuMsg.header = new std_msgs.Header();
        imuMsg.header.stamp = new builtin_interfaces.Time();
        imuMsg.header.frame_id = "imu_frame";

        // Linear acceleration (with gravity compensation)
        Vector3 linearAcceleration = CalculateLinearAcceleration();
        imuMsg.linear_acceleration.x = linearAcceleration.x + Random.Range(-noiseLevel, noiseLevel);
        imuMsg.linear_acceleration.y = linearAcceleration.y + Random.Range(-noiseLevel, noiseLevel);
        imuMsg.linear_acceleration.z = linearAcceleration.z + Random.Range(-noiseLevel, noiseLevel);

        // Angular velocity (from rigidbody rotation)
        Vector3 angularVelocity = attachedRigidbody.angularVelocity;
        imuMsg.angular_velocity.x = angularVelocity.x + Random.Range(-noiseLevel, noiseLevel);
        imuMsg.angular_velocity.y = angularVelocity.y + Random.Range(-noiseLevel, noiseLevel);
        imuMsg.angular_velocity.z = angularVelocity.z + Random.Range(-noiseLevel, noiseLevel);

        // Orientation (from transform)
        Quaternion orientation = transform.rotation;
        imuMsg.orientation.x = orientation.x;
        imuMsg.orientation.y = orientation.y;
        imuMsg.orientation.z = orientation.z;
        imuMsg.orientation.w = orientation.w;

        ros.Publish("/unity_imu/data", imuMsg);
    }

    Vector3 CalculateLinearAcceleration()
    {
        float currentTime = Time.time;
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        Vector3 currentVelocity = attachedRigidbody.velocity;
        Vector3 linearAcceleration = (currentVelocity - lastVelocity) / deltaTime;
        lastVelocity = currentVelocity;

        return linearAcceleration;
    }
}
```

## 5. Integration Between Gazebo and Unity

### 5.1 Data Exchange Mechanisms

While Gazebo and Unity are separate simulation environments, they can be connected through various approaches:

**ROS 2 Bridge**: Use ROS 2 as the communication layer between both environments:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import numpy as np

class SimulationBridge(Node):
    def __init__(self):
        super().__init__('simulation_bridge')

        # Publishers to Unity
        self.unity_laser_pub = self.create_publisher(LaserScan, '/unity_scan', 10)
        self.unity_imu_pub = self.create_publisher(Imu, '/unity_imu', 10)
        self.unity_odom_pub = self.create_publisher(Odometry, '/unity_odom', 10)

        # Subscribers from Gazebo
        self.gazebo_laser_sub = self.create_subscription(
            LaserScan, '/gazebo_scan', self.laser_callback, 10)
        self.gazebo_imu_sub = self.create_subscription(
            Imu, '/gazebo_imu', self.imu_callback, 10)
        self.gazebo_odom_sub = self.create_subscription(
            Odometry, '/gazebo_odom', self.odom_callback, 10)

        # Subscribers from Unity
        self.unity_cmd_sub = self.create_subscription(
            Twist, '/unity_cmd_vel', self.unity_cmd_callback, 10)

        # Publishers to Gazebo
        self.gazebo_cmd_pub = self.create_publisher(Twist, '/gazebo_cmd_vel', 10)

        self.get_logger().info('Simulation Bridge initialized')

    def laser_callback(self, msg):
        """Forward laser data from Gazebo to Unity"""
        self.unity_laser_pub.publish(msg)
        self.get_logger().debug('Forwarded laser data to Unity')

    def imu_callback(self, msg):
        """Forward IMU data from Gazebo to Unity"""
        self.unity_imu_pub.publish(msg)
        self.get_logger().debug('Forwarded IMU data to Unity')

    def odom_callback(self, msg):
        """Forward odometry data from Gazebo to Unity"""
        self.unity_odom_pub.publish(msg)
        self.get_logger().debug('Forwarded odometry data to Unity')

    def unity_cmd_callback(self, msg):
        """Forward command from Unity to Gazebo"""
        self.gazebo_cmd_pub.publish(msg)
        self.get_logger().debug('Forwarded command from Unity to Gazebo')

def main(args=None):
    rclpy.init(args=args)
    bridge = SimulationBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.2 Synchronization Techniques

Synchronizing both simulation environments requires careful timing and state management:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
import time

class SimulationSynchronizer(Node):
    def __init__(self):
        super().__init__('simulation_synchronizer')

        # Time synchronization
        self.gazebo_time_sub = self.create_subscription(
            Time, '/gazebo/time', self.gazebo_time_callback, 10)
        self.unity_time_sub = self.create_subscription(
            Time, '/unity/time', self.unity_time_callback, 10)

        # State synchronization
        self.gazebo_state_sub = self.create_subscription(
            JointState, '/gazebo/joint_states', self.gazebo_state_callback, 10)
        self.unity_state_sub = self.create_subscription(
            JointState, '/unity/joint_states', self.unity_state_callback, 10)

        # Publishers for synchronization
        self.gazebo_sync_pub = self.create_publisher(JointState, '/gazebo/sync_states', 10)
        self.unity_sync_pub = self.create_publisher(JointState, '/unity/sync_states', 10)

        # Internal state tracking
        self.gazebo_time = None
        self.unity_time = None
        self.gazebo_states = {}
        self.unity_states = {}

        # Timer for synchronization
        self.sync_timer = self.create_timer(0.01, self.synchronization_loop)

        self.get_logger().info('Simulation Synchronizer initialized')

    def gazebo_time_callback(self, msg):
        self.gazebo_time = msg

    def unity_time_callback(self, msg):
        self.unity_time = msg

    def gazebo_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.gazebo_states[name] = msg.position[i]

    def unity_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.unity_states[name] = msg.position[i]

    def synchronization_loop(self):
        """Main synchronization loop"""
        # Check if both simulators are publishing time
        if self.gazebo_time and self.unity_time:
            # Calculate time difference
            gazebo_sec = self.gazebo_time.sec + self.gazebo_time.nanosec / 1e9
            unity_sec = self.unity_time.sec + self.unity_time.nanosec / 1e9
            time_diff = abs(gazebo_sec - unity_sec)

            # If time difference is too large, trigger synchronization
            if time_diff > 0.1:  # 100ms threshold
                self.synchronize_states()
                self.get_logger().info(f'Synchronized simulators, time diff: {time_diff:.3f}s')

    def synchronize_states(self):
        """Synchronize states between simulators"""
        # Sync Gazebo -> Unity
        gazebo_state_msg = JointState()
        gazebo_state_msg.header = Header()
        gazebo_state_msg.header.stamp = self.get_clock().now().to_msg()
        gazebo_state_msg.name = list(self.gazebo_states.keys())
        gazebo_state_msg.position = list(self.gazebo_states.values())
        self.unity_sync_pub.publish(gazebo_state_msg)

        # Sync Unity -> Gazebo
        unity_state_msg = JointState()
        unity_state_msg.header = Header()
        unity_state_msg.header.stamp = self.get_clock().now().to_msg()
        unity_state_msg.name = list(self.unity_states.keys())
        unity_state_msg.position = list(self.unity_states.values())
        self.gazebo_sync_pub.publish(unity_state_msg)

def main(args=None):
    rclpy.init(args=args)
    synchronizer = SimulationSynchronizer()

    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        pass
    finally:
        synchronizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6. AI Agent Integration with Simulated Sensors

### 6.1 Perception Systems

AI agents can use simulated sensor data for perception tasks:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import cv2
from cv_bridge import CvBridge
import math

class AI perceptionAgent(Node):
    def __init__(self):
        super().__init__('ai_perception_agent')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)

        # Subscribers for simulated sensors
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        # Timer for AI processing
        self.ai_timer = self.create_timer(0.1, self.ai_processing_loop)

        # Sensor data storage
        self.laser_data = None
        self.image_data = None
        self.imu_data = None
        self.bridge = CvBridge()

        # AI perception state
        self.object_detected = False
        self.obstacle_ahead = False
        self.robot_orientation = 0.0

        self.get_logger().info('AI Perception Agent initialized')

    def laser_callback(self, msg):
        """Process simulated laser scan data"""
        self.laser_data = np.array(msg.ranges)
        # Process for obstacle detection
        front_ranges = self.laser_data[len(self.laser_data)//2-30:len(self.laser_data)//2+30]
        self.obstacle_ahead = np.min(front_ranges) < 1.0 if len(front_ranges) > 0 else False

    def image_callback(self, msg):
        """Process simulated camera image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
            # Process for object detection (simplified example)
            self.detect_objects(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def imu_callback(self, msg):
        """Process simulated IMU data"""
        # Convert quaternion to euler angles
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        self.robot_orientation = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def detect_objects(self, image):
        """Simple object detection using color thresholding"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red objects (example)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Upper red range
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any significant contours found
        self.object_detected = any(cv2.contourArea(contour) > 500 for contour in contours)

    def ai_processing_loop(self):
        """Main AI processing loop"""
        cmd_vel = Twist()

        if self.obstacle_ahead:
            # Obstacle avoidance
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5
            status = "AVOIDING OBSTACLE"
        elif self.object_detected:
            # Move toward detected object
            cmd_vel.linear.x = 0.3
            cmd_vel.angular.z = 0.0
            status = "MOVING TO OBJECT"
        else:
            # Explore environment
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = 0.0
            status = "EXPLORING"

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish status
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AI perceptionAgent()

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

### 6.2 Advanced Perception with Deep Learning

For more sophisticated perception, we can integrate deep learning models:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class PerceptionCNN(nn.Module):
    """Simple CNN for object detection in simulation"""
    def __init__(self, num_classes=5):  # 5 different object types
        super(PerceptionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepPerceptionAgent(Node):
    def __init__(self):
        super().__init__('deep_perception_agent')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Initialize computer vision bridge
        self.bridge = CvBridge()

        # Initialize deep learning model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PerceptionCNN(num_classes=5).to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # AI state
        self.latest_image = None
        self.detected_object_class = None

        self.get_logger().info('Deep Perception Agent initialized')

    def image_callback(self, msg):
        """Process image and run deep learning inference"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Preprocess image for model
            image_tensor = self.preprocess_image(cv_image)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()

                if confidence > 0.7:  # Confidence threshold
                    self.detected_object_class = predicted_class
                    self.get_logger().info(f'Detected object class {predicted_class} with confidence {confidence:.2f}')

        except Exception as e:
            self.get_logger().error(f'Error in image processing: {e}')

    def preprocess_image(self, image):
        """Preprocess image for deep learning model"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image_tensor = self.transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def control_loop(self):
        """Main control loop based on perception"""
        cmd_vel = Twist()

        if self.detected_object_class is not None:
            # Different behaviors based on detected object
            if self.detected_object_class == 0:  # Move toward object 0
                cmd_vel.linear.x = 0.3
                cmd_vel.angular.z = 0.0
            elif self.detected_object_class == 1:  # Avoid object 1
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5
            else:  # Default behavior
                cmd_vel.linear.x = 0.2
                cmd_vel.angular.z = 0.1
        else:
            # Default exploration behavior
            cmd_vel.linear.x = 0.3
            cmd_vel.angular.z = 0.1

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    ai_agent = DeepPerceptionAgent()

    # Add timer for control loop
    ai_agent.control_timer = ai_agent.create_timer(0.1, ai_agent.control_loop)

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

### Exercise 1: Creating a Multi-Sensor Simulation Environment

Create a Gazebo world with multiple sensors and a humanoid robot:

**World file (multi_sensor_world.world):**
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_sensor_world">
    <!-- Physics -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Humanoid Robot with Multiple Sensors -->
    <model name="sensor_humanoid">
      <include>
        <uri>model://simple_humanoid</uri>
        <pose>0 0 1 0 0 0</pose>
      </include>

      <!-- LiDAR on head -->
      <link name="lidar_link">
        <pose>0.1 0 0.3 0 0 0</pose>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0</iyz>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.04</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.02</radius>
              <length>0.04</length>
            </cylinder>
          </geometry>
        </collision>
      </link>

      <joint name="head_to_lidar" type="fixed">
        <parent>head</parent>
        <child>lidar_link</child>
        <pose>0.1 0 0.1 0 0 0</pose>
      </joint>

      <!-- Camera on head -->
      <link name="camera_link">
        <pose>0.15 0 0.3 0 0 0</pose>
        <inertial>
          <mass>0.05</mass>
          <inertia>
            <ixx>0.00005</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.00005</iyy>
            <iyz>0</iyz>
            <izz>0.00005</izz>
          </inertia>
        </inertial>
      </link>

      <joint name="head_to_camera" type="fixed">
        <parent>head</parent>
        <child>camera_link</child>
        <pose>0.15 0 0.1 0 0 0</pose>
      </joint>

      <!-- LiDAR Sensor -->
      <sensor name="head_lidar" type="ray">
        <pose>0.1 0 0.3 0 0 0</pose>
        <ray>
          <scan>
            <horizontal>
              <samples>720</samples>
              <resolution>1.0</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>20.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
          <ros>
            <namespace>/humanoid_robot</namespace>
            <remapping>~/out:=scan</remapping>
          </ros>
          <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
      </sensor>

      <!-- Camera Sensor -->
      <sensor name="head_camera" type="camera">
        <pose>0.15 0 0.3 0 0 0</pose>
        <camera name="head_cam">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <ros>
            <namespace>/humanoid_robot</namespace>
            <remapping>image_raw:=/camera/image_raw</remapping>
            <remapping>camera_info:=/camera/camera_info</remapping>
          </ros>
        </plugin>
      </sensor>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_box">
      <pose>3 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.4</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.4</iyy>
            <iyz>0</iyz>
            <izz>0.4</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.5 0.2 1</ambient>
            <diffuse>0.8 0.5 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Exercise 2: Unity Environment with Interactive Elements

Create a Unity scene with interactive elements that respond to robot actions:

```csharp
using UnityEngine;
using System.Collections;

public class InteractiveObject : MonoBehaviour
{
    [Header("Interaction Properties")]
    public bool isGrabbable = true;
    public bool isMoveable = true;
    public float interactionDistance = 2f;

    [Header("Visual Feedback")]
    public Material highlightMaterial;
    public Material defaultMaterial;

    private bool isHighlighted = false;
    private Renderer objectRenderer;
    private Material originalMaterial;

    void Start()
    {
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer != null)
        {
            originalMaterial = objectRenderer.material;
        }
    }

    public void Highlight()
    {
        if (!isHighlighted && objectRenderer != null && highlightMaterial != null)
        {
            objectRenderer.material = highlightMaterial;
            isHighlighted = true;
        }
    }

    public void Unhighlight()
    {
        if (isHighlighted && objectRenderer != null)
        {
            objectRenderer.material = originalMaterial;
            isHighlighted = false;
        }
    }

    public void OnInteract()
    {
        if (isMoveable)
        {
            // Apply a random force to make the object move
            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                Vector3 randomForce = new Vector3(
                    Random.Range(-5f, 5f),
                    Random.Range(5f, 10f),
                    Random.Range(-5f, 5f)
                );
                rb.AddForce(randomForce, ForceMode.Impulse);
            }
        }

        // Play interaction effect
        StartCoroutine(InteractionEffect());
    }

    IEnumerator InteractionEffect()
    {
        // Change color temporarily
        if (objectRenderer != null)
        {
            objectRenderer.material = highlightMaterial;
            yield return new WaitForSeconds(0.5f);
            objectRenderer.material = originalMaterial;
        }
    }
}

// Robot interaction manager
public class RobotInteractionManager : MonoBehaviour
{
    public float interactionDistance = 3f;
    public LayerMask interactionLayer;

    void Update()
    {
        HandleInteraction();
    }

    void HandleInteraction()
    {
        // Raycast to detect interactive objects
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, interactionDistance, interactionLayer))
        {
            InteractiveObject interactable = hit.collider.GetComponent<InteractiveObject>();
            if (interactable != null)
            {
                // Highlight the object
                interactable.Highlight();

                // Check for interaction input (e.g., spacebar)
                if (Input.GetKeyDown(KeyCode.E))
                {
                    interactable.OnInteract();
                }
            }
        }
    }
}
```

## 8. Summary and Key Takeaways

In this module, we've explored the creation and utilization of digital twins for humanoid robotics using both Gazebo and Unity simulation environments. We've covered:

1. **Digital Twin Fundamentals**: Understanding the concept of digital twins and their importance in robotics development, testing, and validation.

2. **Gazebo Physics Simulation**: Learning to create realistic physics simulations with accurate gravity, collision detection, and material properties, along with comprehensive sensor simulation.

3. **Unity Environment Creation**: Developing high-fidelity visual environments with PBR materials, dynamic lighting, and realistic rendering capabilities.

4. **Sensor Simulation**: Implementing realistic sensor models for LiDAR, cameras, IMUs, and other robotic sensors in both simulation environments.

5. **Integration Approaches**: Understanding how to connect Gazebo and Unity environments through ROS 2 bridges and synchronization mechanisms.

6. **AI Agent Integration**: Connecting AI perception and control systems to simulated sensor data for testing and training.

The combination of Gazebo's physics accuracy and Unity's visual fidelity creates comprehensive digital twins that enable safe, efficient, and cost-effective development of humanoid robotics systems. These digital twins serve as essential tools for testing control algorithms, training AI agents, and validating robot behaviors before deployment to physical hardware.

## References

Grieves, M., & Vickers, J. (2017). Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex Systems. In *Transdisciplinary Perspectives on Complex Systems* (pp. 85-113). Springer.

Koenig, N., & Howard, A. (2004). Design and use paradigms for Gazebo, an open-source multi-robot simulator. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 3, 2149-2154.

Linden, T., O'Kane, J. M., & Shell, D. A. (2021). The Gazebo Handbook: A Complete Guide to Robotic Simulation. *arXiv preprint arXiv:2103.12547*.

Unity Technologies. (2023). *Unity Robotics Integration Documentation*. Retrieved from https://github.com/Unity-Technologies/Unity-Robotics-Hub

Zhou, K., Chen, D., Wu, J., He, Z., Wei, Y., & Sun, H. (2020). Digital twin for intelligent robots: Concepts, reference model, and case studies. *Robotics and Computer-Integrated Manufacturing*, 64, 101934.

## Glossary

- **Digital Twin**: A virtual representation of a physical system that spans its lifecycle and uses real-time data for simulation and analysis.
- **Gazebo**: A physics-based simulation engine for robotics applications with realistic physics and sensor simulation.
- **Unity**: A real-time 3D development platform used for creating high-fidelity visual environments.
- **PBR (Physically-Based Rendering)**: A shading and rendering approach that simulates light behavior in a physically accurate way.
- **ROS Bridge**: A system for connecting different simulation environments or real robots through ROS message passing.
- **Sensor Simulation**: The process of creating virtual sensors that produce data similar to real-world sensors.
- **Synchronization**: The process of keeping multiple simulation environments in temporal and state alignment.
- **Physics Engine**: Software that simulates physical phenomena like gravity, collision, and material properties.