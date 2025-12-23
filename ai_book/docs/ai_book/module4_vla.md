---
title: Module 4 - Vision-Language-Action (VLA)
sidebar_position: 4
---

# Module 4: Vision-Language-Action (VLA)

## Introduction

The Vision-Language-Action (VLA) paradigm represents the integration of three key modalities in AI-driven robotics: vision for perception, language for communication and instruction, and action for physical execution. This module explores how these modalities work together to enable humanoid robots to understand natural language commands, perceive their environment visually, and execute complex tasks in simulation environments.

The VLA framework enables robots to perform complex multi-step tasks by combining visual perception with language understanding and physical action execution. This integration allows for more natural human-robot interaction and enables robots to perform tasks based on high-level instructions rather than pre-programmed sequences.

This module covers the implementation of voice-to-action systems using OpenAI Whisper, natural language processing for ROS 2 action sequences, and the integration of vision-language systems for autonomous humanoid task execution. By the end of this module, you will understand how to create complete VLA systems that enable robots to perceive, understand, and act based on natural language instructions.

## 1. Vision-Language-Action Architecture

### 1.1 The VLA Framework

The Vision-Language-Action framework integrates three modalities to create intelligent robotic systems:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Vision       │    │   Language      │    │     Action      │
│   (Perception)  │◄──►│ (Understanding) │◄──►│  (Execution)    │
│                 │    │                 │    │                 │
│ • Camera data   │    │ • Voice input   │    │ • Motor control │
│ • LiDAR data    │    │ • Text commands │    │ • Navigation    │
│ • Sensor fusion │    │ • Intent parsing│    │ • Manipulation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   VLA Integration       │
                    │   (Multi-modal AI)      │
                    └─────────────────────────┘
```

### 1.2 Multi-Modal Integration

The VLA framework requires sophisticated integration of multiple AI models and systems. The key components include:

1. **Vision Processing**: Processing visual input to understand the environment
2. **Language Processing**: Understanding natural language commands and instructions
3. **Action Planning**: Translating high-level goals into executable robot actions
4. **Perception-Action Loop**: Continuous feedback between perception and action

### 1.3 VLA System Architecture

A complete VLA system consists of several interconnected components:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from audio_common_msgs.msg import AudioData
import numpy as np

class VLASystem(Node):
    def __init__(self):
        super().__init__('vla_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio', self.audio_callback, 10)
        self.command_sub = self.create_subscription(String, '/vla_command', self.command_callback, 10)

        # Initialize VLA components
        self.vision_processor = VisionProcessor()
        self.language_processor = LanguageProcessor()
        self.action_planner = ActionPlanner()
        self.perception_action_loop = PerceptionActionLoop()

        # System state
        self.current_environment = None
        self.current_command = None
        self.executing_action = False

        self.get_logger().info('VLA System initialized')

    def image_callback(self, msg):
        """Process visual input"""
        processed_image = self.vision_processor.process(msg)
        self.current_environment = processed_image
        self.perception_action_loop.update_perception(processed_image)

    def laser_callback(self, msg):
        """Process LiDAR input"""
        processed_scan = self.vision_processor.process_lidar(msg)
        self.perception_action_loop.update_lidar_data(processed_scan)

    def audio_callback(self, msg):
        """Process audio input"""
        if self.executing_action:
            return  # Don't interrupt current action

        # Convert audio to text using Whisper (simplified)
        text_command = self.language_processor.speech_to_text(msg)
        self.get_logger().info(f'Received command: {text_command}')
        self.process_command(text_command)

    def command_callback(self, msg):
        """Process text command"""
        self.process_command(msg.data)

    def process_command(self, command_text):
        """Process natural language command"""
        # Parse the command
        parsed_command = self.language_processor.parse_command(command_text)

        # Plan the action
        action_plan = self.action_planner.plan_action(parsed_command, self.current_environment)

        # Execute the action
        self.perception_action_loop.execute_action(action_plan)

def main(args=None):
    rclpy.init(args=args)
    vla_system = VLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2. Voice-to-Action Implementation

### 2.1 Speech Recognition with OpenAI Whisper

OpenAI Whisper provides state-of-the-art speech recognition capabilities that can be integrated into robotic systems:

```python
import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String
import numpy as np
import torch
import whisper
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')

        # Publishers
        self.text_pub = self.create_publisher(String, '/vla_text_command', 10)
        self.status_pub = self.create_publisher(String, '/voice_status', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio', self.audio_callback, 10)

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        try:
            self.whisper_model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.whisper_model = None

        # Audio processing parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024 * 4  # Process in chunks
        self.audio_buffer = b""
        self.min_audio_duration = 0.5  # Minimum audio duration in seconds

        self.get_logger().info('Voice-to-Action Node initialized')

    def audio_callback(self, msg):
        """Process incoming audio data"""
        if self.whisper_model is None:
            return

        # Append new audio data to buffer
        self.audio_buffer += bytes(msg.data)

        # Check if we have enough audio data to process
        audio_duration = len(self.audio_buffer) / (2 * self.sample_rate)  # Assuming 16-bit audio

        if audio_duration >= self.min_audio_duration:
            # Process the accumulated audio
            self.process_audio_buffer()

    def process_audio_buffer(self):
        """Process the accumulated audio buffer with Whisper"""
        if len(self.audio_buffer) == 0:
            return

        try:
            # Convert raw audio data to audio segment
            audio_segment = self.raw_to_audio_segment(self.audio_buffer)

            # Save to temporary file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_segment.export(temp_file.name, format='wav')

                # Transcribe using Whisper
                result = self.whisper_model.transcribe(temp_file.name)

                # Clean up temporary file
                os.unlink(temp_file.name)

            # Extract transcribed text
            transcribed_text = result['text'].strip()

            if transcribed_text:
                # Publish the transcribed text
                text_msg = String()
                text_msg.data = transcribed_text
                self.text_pub.publish(text_msg)

                self.get_logger().info(f'Transcribed: "{transcribed_text}"')

                # Publish status
                status_msg = String()
                status_msg.data = f"TRANSCRIBED: {transcribed_text}"
                self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

        # Clear the audio buffer after processing
        self.audio_buffer = b""

    def raw_to_audio_segment(self, raw_audio):
        """Convert raw audio bytes to AudioSegment"""
        # Create AudioSegment from raw data (assuming 16-bit, mono, 16kHz)
        audio_segment = AudioSegment(
            data=raw_audio,
            sample_width=2,  # 16-bit
            frame_rate=self.sample_rate,
            channels=1
        )
        return audio_segment

    def preprocess_audio(self, audio_data):
        """Preprocess audio for Whisper"""
        # Normalize audio
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample if necessary
        # Whisper expects 16kHz, so if our audio is already 16kHz, we're good
        return audio_array

def main(args=None):
    rclpy.init(args=args)
    node = VoiceToActionNode()

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

### 2.2 Command Parsing and Intent Recognition

After speech recognition, the system needs to parse the command and extract intent:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import re
import json

class CommandParserNode(Node):
    def __init__(self):
        super().__init__('command_parser_node')

        # Publishers
        self.action_pub = self.create_publisher(String, '/vla_action', 10)
        self.status_pub = self.create_publisher(String, '/command_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/vla_text_command', self.command_callback, 10)

        # Define command patterns and their corresponding actions
        self.command_patterns = [
            # Navigation commands
            {
                'pattern': r'go to the (\w+)',
                'action': 'navigate_to',
                'params': ['location']
            },
            {
                'pattern': r'go (forward|backward|left|right)',
                'action': 'move_direction',
                'params': ['direction']
            },
            {
                'pattern': r'move (\w+)',
                'action': 'move_direction',
                'params': ['direction']
            },
            # Object interaction commands
            {
                'pattern': r'pick up the (\w+)',
                'action': 'pick_up',
                'params': ['object']
            },
            {
                'pattern': r'grasp the (\w+)',
                'action': 'pick_up',
                'params': ['object']
            },
            {
                'pattern': r'put down the (\w+)',
                'action': 'put_down',
                'params': ['object']
            },
            {
                'pattern': r'place the (\w+) on the (\w+)',
                'action': 'place_object',
                'params': ['object', 'destination']
            },
            # General commands
            {
                'pattern': r'stop',
                'action': 'stop',
                'params': []
            },
            {
                'pattern': r'wait',
                'action': 'wait',
                'params': []
            }
        ]

        # Location mappings (for simple navigation)
        self.location_map = {
            'kitchen': {'x': 5.0, 'y': 0.0},
            'living room': {'x': 0.0, 'y': 5.0},
            'bedroom': {'x': -5.0, 'y': 0.0},
            'bathroom': {'x': 0.0, 'y': -5.0},
            'office': {'x': 3.0, 'y': 3.0}
        }

        self.get_logger().info('Command Parser Node initialized')

    def command_callback(self, msg):
        """Process incoming text command"""
        command_text = msg.data.lower().strip()
        self.get_logger().info(f'Processing command: {command_text}')

        # Parse the command
        parsed_command = self.parse_command(command_text)

        if parsed_command:
            # Publish the parsed command
            action_msg = String()
            action_msg.data = json.dumps(parsed_command)
            self.action_pub.publish(action_msg)

            self.get_logger().info(f'Parsed command: {parsed_command}')

            # Publish status
            status_msg = String()
            status_msg.data = f"PARSED: {parsed_command['action']} with params {parsed_command['params']}"
            self.status_pub.publish(status_msg)
        else:
            # Command not recognized
            status_msg = String()
            status_msg.data = f"UNRECOGNIZED: {command_text}"
            self.status_pub.publish(status_msg)
            self.get_logger().warn(f'Unrecognized command: {command_text}')

    def parse_command(self, command_text):
        """Parse natural language command and extract action and parameters"""
        for pattern_config in self.command_patterns:
            pattern = pattern_config['pattern']
            match = re.search(pattern, command_text)

            if match:
                # Extract parameters
                params = {}
                for i, param_name in enumerate(pattern_config['params']):
                    if i < len(match.groups()):
                        param_value = match.group(i + 1)

                        # Special handling for locations
                        if param_name == 'location' and param_value in self.location_map:
                            params['target_pose'] = self.location_map[param_value]
                        elif param_name == 'direction':
                            # Normalize direction commands
                            direction_map = {
                                'forward': 'forward',
                                'backward': 'backward',
                                'back': 'backward',
                                'left': 'left',
                                'right': 'right'
                            }
                            params[param_name] = direction_map.get(param_value, param_value)
                        else:
                            params[param_name] = param_value

                return {
                    'action': pattern_config['action'],
                    'params': params,
                    'raw_command': command_text
                }

        # If no pattern matches, return None
        return None

def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()

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

## 3. Natural Language to ROS 2 Action Sequences

### 3.1 Language Model Integration

Integrating large language models (LLMs) to translate natural language into executable ROS 2 action sequences:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from move_base_msgs.action import MoveBase
import openai
import json
import time

class LLMActionGeneratorNode(Node):
    def __init__(self):
        super().__init__('llm_action_generator_node')

        # Publishers
        self.action_sequence_pub = self.create_publisher(String, '/vla_action_sequence', 10)
        self.status_pub = self.create_publisher(String, '/llm_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/vla_parsed_command', self.command_callback, 10)

        # Initialize OpenAI client
        # Note: You'll need to set your OpenAI API key
        # openai.api_key = "your-api-key-here"

        # For this example, we'll use a mock implementation
        self.use_mock_llm = True
        self.get_logger().info('LLM Action Generator Node initialized')

    def command_callback(self, msg):
        """Process parsed command and generate action sequence"""
        try:
            command_data = json.loads(msg.data)
            raw_command = command_data.get('raw_command', '')

            self.get_logger().info(f'Generating actions for: {raw_command}')

            # Generate action sequence using LLM
            action_sequence = self.generate_action_sequence(raw_command)

            if action_sequence:
                # Publish the action sequence
                sequence_msg = String()
                sequence_msg.data = json.dumps(action_sequence)
                self.action_sequence_pub.publish(sequence_msg)

                self.get_logger().info(f'Generated action sequence: {action_sequence}')

                # Publish status
                status_msg = String()
                status_msg.data = f"GENERATED: {len(action_sequence)} actions"
                self.status_pub.publish(status_msg)
            else:
                self.get_logger().error(f'Failed to generate action sequence for: {raw_command}')

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error decoding JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def generate_action_sequence(self, natural_language_command):
        """Generate ROS 2 action sequence from natural language command"""
        if self.use_mock_llm:
            # Mock implementation for demonstration
            return self.mock_generate_action_sequence(natural_language_command)
        else:
            # Real implementation using OpenAI
            return self.real_generate_action_sequence(natural_language_command)

    def mock_generate_action_sequence(self, command):
        """Mock implementation of action sequence generation"""
        # This is a simplified mock - in reality, this would use an LLM
        command_lower = command.lower()

        if 'go to' in command_lower:
            location = 'kitchen'  # Simplified extraction
            return [
                {
                    'action_type': 'navigation',
                    'target_location': location,
                    'target_pose': {'x': 5.0, 'y': 0.0, 'theta': 0.0}
                }
            ]
        elif 'pick up' in command_lower:
            obj = 'red cup'  # Simplified extraction
            return [
                {
                    'action_type': 'navigation',
                    'target_location': 'table',
                    'target_pose': {'x': 1.0, 'y': 1.0, 'theta': 0.0}
                },
                {
                    'action_type': 'manipulation',
                    'action': 'grasp',
                    'object': obj
                }
            ]
        elif 'move forward' in command_lower:
            return [
                {
                    'action_type': 'motion',
                    'command': 'move_forward',
                    'distance': 1.0
                }
            ]
        elif 'turn left' in command_lower:
            return [
                {
                    'action_type': 'motion',
                    'command': 'turn',
                    'angle': 90.0,
                    'direction': 'left'
                }
            ]
        else:
            # Default action for unrecognized commands
            return [
                {
                    'action_type': 'idle',
                    'command': 'unknown_command',
                    'message': command
                }
            ]

    def real_generate_action_sequence(self, command):
        """Real implementation using OpenAI API"""
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Convert the following natural language command into a sequence of ROS 2 actions for a humanoid robot.
            The robot can perform navigation, manipulation, and basic motion actions.

            Command: "{command}"

            Return a JSON list of action objects with the following structure:
            {{
                "action_type": "navigation|manipulation|motion|perception",
                "target_location|object|command": "specific target",
                "parameters": {{...}}
            }}

            Example response for "Go to the kitchen and pick up the red cup":
            [
                {{
                    "action_type": "navigation",
                    "target_location": "kitchen",
                    "parameters": {{
                        "x": 5.0,
                        "y": 0.0,
                        "theta": 0.0
                    }}
                }},
                {{
                    "action_type": "perception",
                    "command": "detect_object",
                    "parameters": {{
                        "object_type": "cup",
                        "color": "red"
                    }}
                }},
                {{
                    "action_type": "manipulation",
                    "command": "grasp",
                    "parameters": {{
                        "object_id": "red_cup_detected"
                    }}
                }}
            ]
            """

            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or gpt-4 for better results
                messages=[
                    {"role": "system", "content": "You are an expert in robotics and ROS 2. Convert natural language commands to structured action sequences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            # Extract and parse the response
            response_text = response.choices[0].message['content'].strip()

            # Extract JSON from the response (it might be wrapped in markdown code blocks)
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                if json_match:
                    json_str = '[' + json_match.group(1) + ']'
                else:
                    json_str = response_text

            # Parse the JSON
            action_sequence = json.loads(json_str)
            return action_sequence

        except Exception as e:
            self.get_logger().error(f'Error calling OpenAI API: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    node = LLMActionGeneratorNode()

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

### 3.2 Action Sequence Execution

Executing the generated action sequences in the robot:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from rclpy.action import ActionClient
from move_base_msgs.action import MoveBase
from geometry_msgs.msg import PoseStamped
import json
import time

class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__('action_executor_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/action_status', 10)

        # Subscribers
        self.action_sequence_sub = self.create_subscription(
            String, '/vla_action_sequence', self.action_sequence_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Action clients
        self.nav_client = ActionClient(self, MoveBase, 'move_base')

        # Robot state
        self.current_action_sequence = []
        self.current_action_index = 0
        self.is_executing = False
        self.robot_pose = None
        self.last_image = None
        self.laser_data = None

        self.get_logger().info('Action Executor Node initialized')

    def action_sequence_callback(self, msg):
        """Process incoming action sequence"""
        try:
            action_sequence = json.loads(msg.data)

            if not self.is_executing:
                self.current_action_sequence = action_sequence
                self.current_action_index = 0
                self.is_executing = True

                self.get_logger().info(f'Starting execution of {len(action_sequence)} actions')
                self.execute_next_action()
            else:
                self.get_logger().warn('Currently executing actions, queueing new sequence')
                # In a real implementation, you might want to queue or interrupt
                # For this example, we'll just wait

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error decoding action sequence JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing action sequence: {e}')

    def execute_next_action(self):
        """Execute the next action in the sequence"""
        if not self.is_executing or self.current_action_index >= len(self.current_action_sequence):
            self.is_executing = False
            self.get_logger().info('Action sequence completed')

            # Publish completion status
            status_msg = String()
            status_msg.data = 'SEQUENCE_COMPLETED'
            self.status_pub.publish(status_msg)
            return

        current_action = self.current_action_sequence[self.current_action_index]
        self.get_logger().info(f'Executing action {self.current_action_index + 1}/{len(self.current_action_sequence)}: {current_action}')

        # Execute based on action type
        action_type = current_action.get('action_type', 'unknown')

        if action_type == 'navigation':
            self.execute_navigation_action(current_action)
        elif action_type == 'motion':
            self.execute_motion_action(current_action)
        elif action_type == 'manipulation':
            self.execute_manipulation_action(current_action)
        elif action_type == 'perception':
            self.execute_perception_action(current_action)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            self.complete_current_action()

    def execute_navigation_action(self, action):
        """Execute navigation action"""
        target_pose = action.get('parameters', {})
        x = target_pose.get('x', 0.0)
        y = target_pose.get('y', 0.0)
        theta = target_pose.get('theta', 0.0)

        self.get_logger().info(f'Navigating to ({x}, {y}, {theta})')

        # In a real implementation, you would send a navigation goal
        # For this example, we'll simulate the navigation
        self.simulate_navigation(x, y, theta)

    def execute_motion_action(self, action):
        """Execute motion action"""
        command = action.get('command', '')

        if command == 'move_forward':
            distance = action.get('distance', 1.0)
            self.move_forward(distance)
        elif command == 'turn':
            angle = action.get('angle', 90.0)
            direction = action.get('direction', 'left')
            self.turn(angle, direction)
        else:
            self.get_logger().warn(f'Unknown motion command: {command}')
            self.complete_current_action()
            return

    def execute_manipulation_action(self, action):
        """Execute manipulation action"""
        command = action.get('command', '')

        if command == 'grasp':
            obj = action.get('object', 'unknown')
            self.get_logger().info(f'Attempting to grasp {obj}')
            # In a real implementation, you would control the robot's manipulator
            # For this example, we'll just simulate
            time.sleep(2)  # Simulate grasping time
            self.complete_current_action()
        else:
            self.get_logger().warn(f'Unknown manipulation command: {command}')
            self.complete_current_action()

    def execute_perception_action(self, action):
        """Execute perception action"""
        command = action.get('command', '')

        if command == 'detect_object':
            obj_type = action.get('object_type', 'object')
            color = action.get('color', 'any')

            self.get_logger().info(f'Detecting {color} {obj_type}')

            # In a real implementation, you would process the camera image
            # For this example, we'll simulate detection
            if self.last_image is not None:
                # Simulate object detection
                detected = self.simulate_object_detection(obj_type, color)
                if detected:
                    self.get_logger().info(f'Detected {color} {obj_type}')
                else:
                    self.get_logger().info(f'Did not detect {color} {obj_type}')

            self.complete_current_action()
        else:
            self.get_logger().warn(f'Unknown perception command: {command}')
            self.complete_current_action()

    def simulate_navigation(self, x, y, theta):
        """Simulate navigation to target pose"""
        # In a real implementation, you would use the navigation stack
        # For this example, we'll just wait and then complete the action

        # Publish status
        status_msg = String()
        status_msg.data = f'NAVIGATING_TO: ({x}, {y}, {theta})'
        self.status_pub.publish(status_msg)

        # Simulate navigation time
        time.sleep(3)  # Simulate navigation time

        # Complete the action
        self.complete_current_action()

    def move_forward(self, distance):
        """Move robot forward by specified distance"""
        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5  # Forward speed
        cmd_vel.angular.z = 0.0

        # Calculate time needed (simplified)
        duration = distance / 0.5  # time = distance / speed

        self.get_logger().info(f'Moving forward {distance}m')

        # Publish command for the calculated duration
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_vel_pub.publish(cmd_vel)
            time.sleep(0.1)

        # Stop robot
        cmd_vel.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        self.complete_current_action()

    def turn(self, angle, direction):
        """Turn robot by specified angle"""
        cmd_vel = Twist()

        # Determine turn direction
        if direction == 'left':
            cmd_vel.angular.z = 0.5  # Positive for left turn
        else:
            cmd_vel.angular.z = -0.5  # Negative for right turn

        # Calculate time needed (simplified: 90 degrees in 3 seconds)
        duration = (abs(angle) / 90.0) * 3.0  # Scale based on angle

        self.get_logger().info(f'Turning {direction} by {angle} degrees')

        # Publish command for the calculated duration
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_vel_pub.publish(cmd_vel)
            time.sleep(0.1)

        # Stop robot
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        self.complete_current_action()

    def simulate_object_detection(self, obj_type, color):
        """Simulate object detection"""
        # In a real implementation, you would process the image to detect objects
        # For this example, we'll just return True sometimes
        import random
        return random.random() > 0.3  # 70% success rate for simulation

    def complete_current_action(self):
        """Mark current action as complete and move to next"""
        self.current_action_index += 1
        self.get_logger().info(f'Completed action {self.current_action_index - 1}')

        # Check if sequence is complete
        if self.current_action_index >= len(self.current_action_sequence):
            self.is_executing = False
            self.get_logger().info('Action sequence completed')

            # Publish completion status
            status_msg = String()
            status_msg.data = 'SEQUENCE_COMPLETED'
            self.status_pub.publish(status_msg)
        else:
            # Execute next action
            self.execute_next_action()

    def image_callback(self, msg):
        """Process incoming image"""
        self.last_image = msg

    def laser_callback(self, msg):
        """Process incoming laser scan"""
        self.laser_data = msg

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutorNode()

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

## 4. Vision Integration for Perception

### 4.1 Object Detection and Recognition

Integrating vision systems for object detection and scene understanding:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

class VisionPerceptionNode(Node):
    def __init__(self):
        super().__init__('vision_perception_node')

        # Publishers
        self.object_detection_pub = self.create_publisher(String, '/vla_detected_objects', 10)
        self.perception_status_pub = self.create_publisher(String, '/vision_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize vision model (using a pre-trained model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_object_detection_model()
        self.model.eval()

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Object class names (COCO dataset classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.get_logger().info('Vision Perception Node initialized')

    def initialize_object_detection_model(self):
        """Initialize a pre-trained object detection model"""
        # For this example, we'll use a simple model
        # In practice, you might use YOLOv5, Detectron2, or other models
        import torchvision.models as models

        # Using a ResNet model for classification as a placeholder
        # In real implementation, use a proper object detection model
        model = models.resnet18(pretrained=True)
        return model

    def image_callback(self, msg):
        """Process incoming image for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection
            detected_objects = self.detect_objects(cv_image)

            # Publish detected objects
            if detected_objects:
                objects_msg = String()
                objects_msg.data = str(detected_objects)
                self.object_detection_pub.publish(objects_msg)

                self.get_logger().info(f'Detected objects: {detected_objects}')

            # Publish status
            status_msg = String()
            status_msg.data = f'DETECTED: {len(detected_objects)} objects'
            self.perception_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        """Detect objects in the image"""
        # In a real implementation, this would use a proper object detection model
        # For this example, we'll use a simplified approach

        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For demonstration, let's use a simple color-based detection
        # In reality, you'd use a deep learning model
        detected_objects = []

        # Detect red objects (for example)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Define range for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Find contours of red objects
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                detected_objects.append({
                    'class': 'red_object',  # Simplified class
                    'confidence': 0.8,  # Simulated confidence
                    'bbox': [x, y, x+w, y+h],
                    'center': [x + w//2, y + h//2]
                })

        return detected_objects

    def deep_learning_detect_objects(self, image):
        """Use deep learning model for object detection"""
        # Convert image for model input
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Process output (simplified - in real implementation, use proper post-processing)
            _, predicted = torch.max(output, 1)

            class_id = predicted.item()
            confidence = torch.nn.functional.softmax(output[0], dim=0)[predicted].item()

            if confidence > 0.5:  # Confidence threshold
                return [{
                    'class': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                    'confidence': confidence,
                    'bbox': [0, 0, image.shape[1], image.shape[0]]  # Full image as bbox for simplicity
                }]

        return []

def main(args=None):
    rclpy.init(args=args)
    node = VisionPerceptionNode()

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

### 4.2 Scene Understanding and Spatial Reasoning

Integrating scene understanding for spatial reasoning:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import json

class SceneUnderstandingNode(Node):
    def __init__(self):
        super().__init__('scene_understanding_node')

        # Publishers
        self.spatial_map_pub = self.create_publisher(String, '/vla_spatial_map', 10)
        self.scene_description_pub = self.create_publisher(String, '/vla_scene_description', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Initialize components
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_scan = None
        self.spatial_map = {}  # Dictionary to store spatial information

        self.get_logger().info('Scene Understanding Node initialized')

    def image_callback(self, msg):
        """Process image for scene understanding"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.update_spatial_map()
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def laser_callback(self, msg):
        """Process laser scan for spatial information"""
        self.latest_scan = msg
        self.update_spatial_map()

    def update_spatial_map(self):
        """Update spatial map with current sensor data"""
        if self.latest_image is None or self.latest_scan is None:
            return

        # Process image to identify objects and their positions
        image_objects = self.process_image_for_objects(self.latest_image)

        # Process laser scan for spatial layout
        spatial_layout = self.process_laser_for_layout(self.latest_scan)

        # Combine information
        self.spatial_map = {
            'timestamp': self.get_clock().now().to_msg().sec,
            'objects': image_objects,
            'layout': spatial_layout,
            'relative_positions': self.calculate_relative_positions(image_objects, spatial_layout)
        }

        # Publish spatial map
        spatial_map_msg = String()
        spatial_map_msg.data = json.dumps(self.spatial_map)
        self.spatial_map_pub.publish(spatial_map_msg)

        # Generate scene description
        scene_description = self.generate_scene_description(self.spatial_map)
        description_msg = String()
        description_msg.data = scene_description
        self.scene_description_pub.publish(description_msg)

        self.get_logger().info(f'Updated spatial map with {len(image_objects)} objects')

    def process_image_for_objects(self, image):
        """Process image to identify objects and their screen positions"""
        # In a real implementation, this would use object detection
        # For this example, we'll simulate object detection
        height, width = image.shape[:2]

        # Simulate detected objects
        objects = [
            {
                'class': 'table',
                'confidence': 0.9,
                'screen_bbox': [width//4, height//2, width//2, height//2 + 100],  # x, y, w, h
                'center': [width//4 + width//4, height//2 + 50]  # x, y center
            },
            {
                'class': 'cup',
                'confidence': 0.85,
                'screen_bbox': [width//3, height//3, 50, 50],
                'center': [width//3 + 25, height//3 + 25]
            }
        ]

        return objects

    def process_laser_for_layout(self, scan):
        """Process laser scan to understand spatial layout"""
        if scan is None:
            return {}

        # Convert laser scan to spatial information
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        ranges = np.array(scan.ranges)

        # Filter out invalid ranges
        valid_indices = np.isfinite(ranges)
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]

        # Calculate spatial features
        layout_features = {
            'room_shape': self.estimate_room_shape(valid_angles, valid_ranges),
            'obstacles': self.find_obstacles(valid_angles, valid_ranges),
            'free_space': self.find_free_space(valid_angles, valid_ranges)
        }

        return layout_features

    def estimate_room_shape(self, angles, ranges):
        """Estimate room shape from laser data"""
        # Simplified room shape estimation
        # In reality, this would use more sophisticated algorithms
        distances = ranges[ranges < 10.0]  # Filter out very far measurements

        if len(distances) == 0:
            return {'type': 'unknown', 'dimensions': {}}

        avg_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Roughly classify based on distance variation
        if std_distance < 0.5:
            shape_type = 'circular'
        elif std_distance < 1.5:
            shape_type = 'rectangular'
        else:
            shape_type = 'irregular'

        return {
            'type': shape_type,
            'avg_distance': float(avg_distance),
            'std_distance': float(std_distance)
        }

    def find_obstacles(self, angles, ranges):
        """Find obstacles from laser data"""
        obstacle_threshold = 2.0  # Consider anything closer than 2m as potential obstacle
        obstacle_indices = ranges < obstacle_threshold

        obstacles = []
        for i in range(len(angles)):
            if obstacle_indices[i]:
                x = ranges[i] * np.cos(angles[i])
                y = ranges[i] * np.sin(angles[i])

                obstacles.append({
                    'position': {'x': float(x), 'y': float(y)},
                    'distance': float(ranges[i]),
                    'angle': float(angles[i])
                })

        return obstacles

    def find_free_space(self, angles, ranges):
        """Find free space from laser data"""
        free_threshold = 3.0  # Consider anything further than 3m as free space
        free_indices = ranges > free_threshold

        free_spaces = []
        for i in range(len(angles)):
            if free_indices[i]:
                x = ranges[i] * np.cos(angles[i])
                y = ranges[i] * np.sin(angles[i])

                free_spaces.append({
                    'position': {'x': float(x), 'y': float(y)},
                    'distance': float(ranges[i]),
                    'angle': float(angles[i])
                })

        return free_spaces

    def calculate_relative_positions(self, objects, layout):
        """Calculate relative positions between objects"""
        # In a real implementation, this would combine image and laser data
        # For this example, we'll simulate relative positioning
        relative_positions = []

        for obj in objects:
            # Simulate converting screen coordinates to real-world coordinates
            # This would require camera calibration and depth information in practice
            relative_positions.append({
                'object': obj['class'],
                'screen_center': obj['center'],
                'estimated_world_position': {
                    'x': float(obj['center'][0]) / 100.0 - 2.0,  # Rough conversion
                    'y': float(obj['center'][1]) / 100.0 - 2.0,
                    'z': 0.0
                }
            })

        return relative_positions

    def generate_scene_description(self, spatial_map):
        """Generate a natural language description of the scene"""
        objects = spatial_map.get('objects', [])
        layout = spatial_map.get('layout', {})

        description_parts = []

        # Describe objects
        if objects:
            object_classes = [obj['class'] for obj in objects]
            unique_classes = list(set(object_classes))

            if len(unique_classes) == 1:
                description_parts.append(f"I see a {unique_classes[0]}.")
            else:
                description_parts.append(f"I see a {', '.join(unique_classes[:-1])}, and a {unique_classes[-1]}.")

        # Describe layout
        room_shape = layout.get('room_shape', {}).get('type', 'unknown')
        if room_shape != 'unknown':
            description_parts.append(f"The room appears to be {room_shape}-shaped.")

        # Describe obstacles
        obstacles = layout.get('obstacles', [])
        if len(obstacles) > 0:
            description_parts.append(f"There are {len(obstacles)} obstacles detected.")

        return " ".join(description_parts)

def main(args=None):
    rclpy.init(args=args)
    node = SceneUnderstandingNode()

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

## 5. Capstone Project: Autonomous Humanoid

### 5.1 Complete VLA System Integration

Creating a complete system that integrates all VLA components:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from audio_common_msgs.msg import AudioData
import json
import threading
import time

class AutonomousHumanoidNode(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/humanoid_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio', self.audio_callback, 10)

        # Initialize VLA components
        self.voice_processor = VoiceToActionNode(self)
        self.command_parser = CommandParserNode(self)
        self.llm_generator = LLMActionGeneratorNode(self)
        self.action_executor = ActionExecutorNode(self)
        self.vision_perceptor = VisionPerceptionNode(self)
        self.scene_understander = SceneUnderstandingNode(self)

        # System state
        self.current_task = None
        self.is_active = False
        self.task_queue = []
        self.robot_state = {
            'position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'detected_objects': [],
            'environment_map': {}
        }

        # Start the main control loop in a separate thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        self.get_logger().info('Autonomous Humanoid Node initialized')

    def image_callback(self, msg):
        """Handle image input"""
        self.vision_perceptor.image_callback(msg)
        self.scene_understander.image_callback(msg)

    def laser_callback(self, msg):
        """Handle laser scan input"""
        self.scene_understander.laser_callback(msg)

    def audio_callback(self, msg):
        """Handle audio input"""
        if not self.is_active:
            return  # Don't process commands if not active

        # Process through voice-to-action pipeline
        self.voice_processor.audio_callback(msg)

    def control_loop(self):
        """Main control loop for the humanoid"""
        while rclpy.ok():
            try:
                # Update robot state
                self.update_robot_state()

                # Process any pending tasks
                self.process_task_queue()

                # Publish status
                status_msg = String()
                status_msg.data = f"ACTIVE: {len(self.task_queue)} tasks queued"
                self.status_pub.publish(status_msg)

                # Sleep to control loop frequency
                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Error in control loop: {e}')

    def update_robot_state(self):
        """Update internal robot state based on sensor inputs"""
        # This would update the robot's understanding of its environment
        # For this example, we'll just log that we're updating state
        self.get_logger().debug('Updating robot state')

    def process_task_queue(self):
        """Process tasks in the queue"""
        if self.task_queue and not self.action_executor.is_executing:
            # Get the next task
            task = self.task_queue.pop(0)

            # Convert task to action sequence and execute
            self.execute_task(task)

    def execute_task(self, task):
        """Execute a specific task"""
        self.get_logger().info(f'Executing task: {task}')

        # In a real implementation, this would convert the task to actions
        # and execute them through the action executor
        # For this example, we'll just simulate execution
        time.sleep(1)  # Simulate task execution time

    def add_task(self, task_description):
        """Add a task to the execution queue"""
        self.task_queue.append(task_description)
        self.get_logger().info(f'Added task to queue: {task_description}')

    def start_autonomous_mode(self):
        """Start autonomous operation"""
        self.is_active = True
        self.get_logger().info('Autonomous mode started')

    def stop_autonomous_mode(self):
        """Stop autonomous operation"""
        self.is_active = False
        self.get_logger().info('Autonomous mode stopped')

class VoiceToActionNode:
    """Simplified voice processing component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node

    def audio_callback(self, msg):
        """Process audio and generate text command"""
        # In a real implementation, this would use Whisper
        # For this example, we'll simulate transcription
        command_text = "go to the kitchen and pick up the red cup"

        # Publish to command parser
        command_msg = String()
        command_msg.data = command_text
        # In real implementation, publish to appropriate topic

class CommandParserNode:
    """Simplified command parsing component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node

class LLMActionGeneratorNode:
    """Simplified LLM action generation component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node

class ActionExecutorNode:
    """Simplified action execution component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.is_executing = False

class VisionPerceptionNode:
    """Simplified vision perception component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node

    def image_callback(self, msg):
        """Process image input"""
        # In real implementation, this would detect objects
        pass

class SceneUnderstandingNode:
    """Simplified scene understanding component"""
    def __init__(self, parent_node):
        self.parent_node = parent_node

    def image_callback(self, msg):
        """Process image for scene understanding"""
        pass

    def laser_callback(self, msg):
        """Process laser scan for scene understanding"""
        pass

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousHumanoidNode()

    # Start in autonomous mode
    node.start_autonomous_mode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_autonomous_mode()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.2 Multi-Step Task Execution

Implementing complex multi-step task execution:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import json
import time
from enum import Enum

class TaskState(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class MultiStepTaskExecutorNode(Node):
    def __init__(self):
        super().__init__('multi_step_task_executor_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.task_status_pub = self.create_publisher(String, '/task_status', 10)

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/vla_multi_step_task', self.task_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Task management
        self.current_task = None
        self.task_queue = []
        self.current_step = 0
        self.task_state = TaskState.PENDING
        self.step_results = []

        # Robot state
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.detected_objects = []
        self.environment_map = {}

        self.get_logger().info('Multi-Step Task Executor Node initialized')

    def task_callback(self, msg):
        """Process incoming multi-step task"""
        try:
            task_data = json.loads(msg.data)

            # Add task to queue if not currently executing
            if self.task_state != TaskState.EXECUTING:
                self.task_queue.append(task_data)
                self.get_logger().info(f'Added task to queue: {task_data.get("task_name", "unnamed")}')

                # Start execution if idle
                if self.task_state == TaskState.PENDING and not self.current_task:
                    self.start_next_task()
            else:
                self.task_queue.append(task_data)
                self.get_logger().info('Task queued, currently executing another task')

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error decoding task JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing task: {e}')

    def start_next_task(self):
        """Start executing the next task in the queue"""
        if self.task_queue:
            self.current_task = self.task_queue.pop(0)
            self.current_step = 0
            self.task_state = TaskState.EXECUTING
            self.step_results = []

            task_name = self.current_task.get('task_name', 'unnamed')
            self.get_logger().info(f'Starting task: {task_name}')

            # Publish task start status
            status_msg = String()
            status_msg.data = f'TASK_STARTED: {task_name}'
            self.task_status_pub.publish(status_msg)

            # Execute first step
            self.execute_current_step()

    def execute_current_step(self):
        """Execute the current step of the task"""
        if not self.current_task or self.current_step >= len(self.current_task.get('steps', [])):
            # Task completed
            self.complete_task()
            return

        step = self.current_task['steps'][self.current_step]
        step_type = step.get('type', 'unknown')

        self.get_logger().info(f'Executing step {self.current_step + 1}: {step_type}')

        # Execute based on step type
        if step_type == 'navigate':
            self.execute_navigation_step(step)
        elif step_type == 'detect_object':
            self.execute_detection_step(step)
        elif step_type == 'manipulate':
            self.execute_manipulation_step(step)
        elif step_type == 'wait':
            self.execute_wait_step(step)
        else:
            self.get_logger().error(f'Unknown step type: {step_type}')
            self.fail_current_step(f'Unknown step type: {step_type}')

    def execute_navigation_step(self, step):
        """Execute navigation step"""
        target = step.get('target', {})
        x = target.get('x', 0.0)
        y = target.get('y', 0.0)
        theta = target.get('theta', 0.0)

        self.get_logger().info(f'Navigating to ({x}, {y})')

        # Simulate navigation
        self.simulate_navigation(x, y, theta)

        # Record step result
        self.step_results.append({
            'step': self.current_step,
            'type': 'navigate',
            'status': 'completed',
            'result': {'x': x, 'y': y}
        })

        # Move to next step
        self.current_step += 1
        self.execute_current_step()

    def execute_detection_step(self, step):
        """Execute object detection step"""
        target_object = step.get('object', 'object')
        color = step.get('color', 'any')

        self.get_logger().info(f'Detecting {color} {target_object}')

        # Simulate detection using current sensor data
        detected = self.simulate_object_detection(target_object, color)

        if detected:
            self.get_logger().info(f'Successfully detected {color} {target_object}')
            self.step_results.append({
                'step': self.current_step,
                'type': 'detect_object',
                'status': 'completed',
                'result': {'object': target_object, 'detected': True, 'position': {'x': 1.0, 'y': 1.0}}
            })
        else:
            self.get_logger().warn(f'Failed to detect {color} {target_object}')
            self.step_results.append({
                'step': self.current_step,
                'type': 'detect_object',
                'status': 'failed',
                'result': {'object': target_object, 'detected': False}
            })

        # Move to next step
        self.current_step += 1
        self.execute_current_step()

    def execute_manipulation_step(self, step):
        """Execute manipulation step"""
        action = step.get('action', 'grasp')
        object_id = step.get('object_id', 'unknown')

        self.get_logger().info(f'Performing {action} on {object_id}')

        # Simulate manipulation
        success = self.simulate_manipulation(action, object_id)

        if success:
            self.get_logger().info(f'Successfully performed {action} on {object_id}')
            self.step_results.append({
                'step': self.current_step,
                'type': 'manipulate',
                'status': 'completed',
                'result': {'action': action, 'object': object_id, 'success': True}
            })
        else:
            self.get_logger().warn(f'Failed to perform {action} on {object_id}')
            self.step_results.append({
                'step': self.current_step,
                'type': 'manipulate',
                'status': 'failed',
                'result': {'action': action, 'object': object_id, 'success': False}
            })

        # Move to next step
        self.current_step += 1
        self.execute_current_step()

    def execute_wait_step(self, step):
        """Execute wait step"""
        duration = step.get('duration', 1.0)

        self.get_logger().info(f'Waiting for {duration} seconds')

        # Simulate wait
        time.sleep(duration)

        self.step_results.append({
            'step': self.current_step,
            'type': 'wait',
            'status': 'completed',
            'result': {'duration': duration}
        })

        # Move to next step
        self.current_step += 1
        self.execute_current_step()

    def simulate_navigation(self, x, y, theta):
        """Simulate navigation to target position"""
        # In a real implementation, this would use the navigation stack
        # For this example, we'll simulate the movement
        self.get_logger().info(f'Navigating to ({x}, {y}, {theta})')
        time.sleep(2)  # Simulate navigation time

        # Update robot pose
        self.robot_pose = {'x': x, 'y': y, 'theta': theta}

    def simulate_object_detection(self, obj_type, color):
        """Simulate object detection"""
        # In a real implementation, this would process the camera image
        # For this example, we'll simulate with some probability
        import random
        return random.random() > 0.2  # 80% success rate

    def simulate_manipulation(self, action, object_id):
        """Simulate manipulation action"""
        # In a real implementation, this would control the robot's manipulator
        # For this example, we'll simulate with high success rate
        import random
        return random.random() > 0.1  # 90% success rate

    def complete_task(self):
        """Complete the current task"""
        if self.current_task:
            task_name = self.current_task.get('task_name', 'unnamed')
            success = all(result['status'] == 'completed' for result in self.step_results)

            self.task_state = TaskState.COMPLETED if success else TaskState.FAILED

            status_msg = String()
            status_msg.data = f'TASK_COMPLETED: {task_name}, Success: {success}, Steps: {len(self.step_results)}'
            self.task_status_pub.publish(status_msg)

            self.get_logger().info(f'Task {task_name} completed with status: {"SUCCESS" if success else "FAILED"}')

            # Clear current task
            self.current_task = None
            self.current_step = 0
            self.step_results = []

            # Start next task if available
            if self.task_queue:
                self.start_next_task()

    def fail_current_step(self, reason):
        """Fail the current step"""
        self.get_logger().error(f'Step failed: {reason}')

        self.step_results.append({
            'step': self.current_step,
            'type': 'unknown',
            'status': 'failed',
            'result': {'error': reason}
        })

        self.task_state = TaskState.FAILED

        status_msg = String()
        status_msg.data = f'STEP_FAILED: {reason}'
        self.task_status_pub.publish(status_msg)

        # Clear current task
        self.current_task = None
        self.current_step = 0
        self.step_results = []

    def image_callback(self, msg):
        """Process incoming image"""
        # Update internal image data for perception
        pass

    def laser_callback(self, msg):
        """Process incoming laser scan"""
        # Update internal laser data for navigation
        pass

def main(args=None):
    rclpy.init(args=args)
    node = MultiStepTaskExecutorNode()

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

## 6. Practical Exercises

### Exercise 1: Complete VLA System Implementation

Create a complete VLA system that integrates voice, vision, and action:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from audio_common_msgs.msg import AudioData
import json
import time

class CompleteVLASystem(Node):
    def __init__(self):
        super().__init__('complete_vla_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.system_status_pub = self.create_publisher(String, '/vla_system_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio', self.audio_callback, 10)
        self.command_sub = self.create_subscription(String, '/vla_high_level_command', self.command_callback, 10)

        # System components
        self.voice_recognizer = VoiceRecognizer(self)
        self.language_interpreter = LanguageInterpreter(self)
        self.vision_processor = VisionProcessor(self)
        self.action_planner = ActionPlanner(self)
        self.executor = ActionExecutor(self)

        # System state
        self.system_active = True
        self.current_task = None
        self.perception_data = {
            'image': None,
            'laser': None,
            'objects': [],
            'layout': {}
        }

        self.get_logger().info('Complete VLA System initialized')

    def image_callback(self, msg):
        """Process image input"""
        self.perception_data['image'] = msg
        self.vision_processor.process_image(msg)

    def laser_callback(self, msg):
        """Process laser scan input"""
        self.perception_data['laser'] = msg
        # Process laser data for navigation and obstacle detection

    def audio_callback(self, msg):
        """Process audio input"""
        if self.system_active:
            recognized_text = self.voice_recognizer.recognize_speech(msg)
            if recognized_text:
                self.process_command(recognized_text)

    def command_callback(self, msg):
        """Process high-level command"""
        self.process_command(msg.data)

    def process_command(self, command_text):
        """Process a natural language command through the VLA pipeline"""
        self.get_logger().info(f'Processing command: {command_text}')

        # Step 1: Language interpretation
        interpreted_command = self.language_interpreter.interpret(command_text, self.perception_data)

        # Step 2: Action planning
        action_plan = self.action_planner.plan(interpreted_command, self.perception_data)

        # Step 3: Execution
        success = self.executor.execute(action_plan)

        # Publish status
        status_msg = String()
        status_msg.data = f'COMMAND_PROCESSED: "{command_text}", SUCCESS: {success}'
        self.system_status_pub.publish(status_msg)

        if success:
            self.get_logger().info('Command executed successfully')
        else:
            self.get_logger().error('Command execution failed')

    def start_system(self):
        """Start the VLA system"""
        self.system_active = True
        self.get_logger().info('VLA System started')

    def stop_system(self):
        """Stop the VLA system"""
        self.system_active = False
        self.get_logger().info('VLA System stopped')

class VoiceRecognizer:
    def __init__(self, parent_node):
        self.node = parent_node

    def recognize_speech(self, audio_msg):
        """Recognize speech from audio message"""
        # In a real implementation, this would use Whisper or similar
        # For this example, we'll simulate recognition
        return "go to the kitchen and pick up the red cup"

class LanguageInterpreter:
    def __init__(self, parent_node):
        self.node = parent_node

    def interpret(self, command_text, perception_data):
        """Interpret natural language command"""
        # Parse the command and create an intermediate representation
        # This would involve NLP processing and semantic understanding

        # For this example, we'll create a simple interpretation
        if "go to" in command_text:
            target_location = "kitchen"  # Extract from command
            return {
                'action': 'navigate',
                'target': target_location,
                'parameters': {}
            }
        elif "pick up" in command_text:
            target_object = "red cup"  # Extract from command
            return {
                'action': 'manipulate',
                'target': target_object,
                'operation': 'grasp',
                'parameters': {}
            }
        else:
            return {
                'action': 'unknown',
                'command': command_text
            }

class VisionProcessor:
    def __init__(self, parent_node):
        self.node = parent_node

    def process_image(self, image_msg):
        """Process image for object detection and scene understanding"""
        # In a real implementation, this would run object detection models
        # For this example, we'll simulate detection
        pass

class ActionPlanner:
    def __init__(self, parent_node):
        self.node = parent_node

    def plan(self, interpreted_command, perception_data):
        """Plan a sequence of actions based on interpreted command"""
        # Create a detailed action plan
        # This would involve path planning, manipulation planning, etc.

        if interpreted_command['action'] == 'navigate':
            return [
                {
                    'type': 'navigation',
                    'target_location': interpreted_command['target'],
                    'waypoints': [{'x': 1.0, 'y': 0.0}, {'x': 2.0, 'y': 0.0}, {'x': 2.0, 'y': 1.0}]
                }
            ]
        elif interpreted_command['action'] == 'manipulate':
            return [
                {
                    'type': 'navigation',
                    'target_location': 'object_location',
                    'waypoints': [{'x': 1.0, 'y': 1.0}]
                },
                {
                    'type': 'manipulation',
                    'operation': 'grasp',
                    'target_object': interpreted_command['target']
                }
            ]
        else:
            return []

class ActionExecutor:
    def __init__(self, parent_node):
        self.node = parent_node

    def execute(self, action_plan):
        """Execute the planned sequence of actions"""
        for action in action_plan:
            success = self.execute_single_action(action)
            if not success:
                return False
        return True

    def execute_single_action(self, action):
        """Execute a single action"""
        action_type = action['type']

        if action_type == 'navigation':
            return self.execute_navigation(action)
        elif action_type == 'manipulation':
            return self.execute_manipulation(action)
        else:
            self.node.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation(self, action):
        """Execute navigation action"""
        waypoints = action.get('waypoints', [])

        for waypoint in waypoints:
            x, y = waypoint['x'], waypoint['y']
            self.node.get_logger().info(f'Navigating to ({x}, {y})')

            # Simulate navigation
            time.sleep(1)  # Simulate movement time

        return True

    def execute_manipulation(self, action):
        """Execute manipulation action"""
        operation = action.get('operation', 'grasp')
        target_object = action.get('target_object', 'unknown')

        self.node.get_logger().info(f'Performing {operation} on {target_object}')

        # Simulate manipulation
        time.sleep(2)  # Simulate manipulation time

        return True

def main(args=None):
    rclpy.init(args=args)
    vla_system = CompleteVLASystem()

    try:
        vla_system.start_system()
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.stop_system()
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 2: Capstone Humanoid Project

Create a complete capstone project that demonstrates multi-step autonomous tasks:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from audio_common_msgs.msg import AudioData
import json
import time
import random

class CapstoneHumanoidProject(Node):
    def __init__(self):
        super().__init__('capstone_humanoid_project')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.project_status_pub = self.create_publisher(String, '/capstone_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio', self.audio_callback, 10)

        # Project state
        self.project_active = False
        self.current_phase = 0
        self.phases = [
            self.phase_1_perception,
            self.phase_2_navigation,
            self.phase_3_manipulation,
            self.phase_4_integration
        ]
        self.phase_names = [
            "Perception and Object Detection",
            "Navigation and Path Planning",
            "Manipulation and Interaction",
            "Complete VLA Integration"
        ]

        # Robot capabilities
        self.robot_capabilities = {
            'navigation': True,
            'manipulation': True,
            'perception': True,
            'speech_recognition': True
        }

        self.get_logger().info('Capstone Humanoid Project initialized')

    def image_callback(self, msg):
        """Process image for perception"""
        if self.project_active:
            self.process_perception_phase(msg)

    def laser_callback(self, msg):
        """Process laser data for navigation"""
        if self.project_active:
            self.process_navigation_phase(msg)

    def audio_callback(self, msg):
        """Process audio for command understanding"""
        if self.project_active:
            self.process_voice_command(msg)

    def start_project(self):
        """Start the capstone project"""
        self.project_active = True
        self.current_phase = 0
        self.get_logger().info('Capstone project started')

        # Begin with first phase
        self.execute_current_phase()

    def execute_current_phase(self):
        """Execute the current phase of the project"""
        if self.current_phase < len(self.phases):
            phase_name = self.phase_names[self.current_phase]
            self.get_logger().info(f'Starting phase: {phase_name}')

            # Execute the current phase
            success = self.phases[self.current_phase]()

            if success:
                self.get_logger().info(f'Phase completed successfully: {phase_name}')

                # Publish phase completion
                status_msg = String()
                status_msg.data = f'PHASE_COMPLETED: {phase_name}'
                self.project_status_pub.publish(status_msg)

                # Move to next phase
                self.current_phase += 1

                if self.current_phase < len(self.phases):
                    # Continue to next phase after a delay
                    time.sleep(2)
                    self.execute_current_phase()
                else:
                    # All phases completed
                    self.complete_project()
            else:
                self.get_logger().error(f'Phase failed: {phase_name}')
                self.fail_project()
        else:
            self.complete_project()

    def phase_1_perception(self):
        """Phase 1: Perception and Object Detection"""
        self.get_logger().info('Phase 1: Perception and Object Detection')

        # Simulate perception tasks
        self.get_logger().info('Detecting objects in environment...')

        # Simulate object detection
        detected_objects = self.simulate_object_detection()

        if detected_objects:
            self.get_logger().info(f'Detected {len(detected_objects)} objects: {detected_objects}')
            return True
        else:
            self.get_logger().warn('No objects detected')
            return False

    def phase_2_navigation(self):
        """Phase 2: Navigation and Path Planning"""
        self.get_logger().info('Phase 2: Navigation and Path Planning')

        # Simulate navigation tasks
        self.get_logger().info('Planning path to target location...')

        # Simulate navigation
        navigation_success = self.simulate_navigation()

        if navigation_success:
            self.get_logger().info('Navigation completed successfully')
            return True
        else:
            self.get_logger().error('Navigation failed')
            return False

    def phase_3_manipulation(self):
        """Phase 3: Manipulation and Interaction"""
        self.get_logger().info('Phase 3: Manipulation and Interaction')

        # Simulate manipulation tasks
        self.get_logger().info('Attempting object manipulation...')

        # Simulate manipulation
        manipulation_success = self.simulate_manipulation()

        if manipulation_success:
            self.get_logger().info('Manipulation completed successfully')
            return True
        else:
            self.get_logger().error('Manipulation failed')
            return False

    def phase_4_integration(self):
        """Phase 4: Complete VLA Integration"""
        self.get_logger().info('Phase 4: Complete VLA Integration')

        # Execute a complete multi-step task combining all capabilities
        success = self.execute_complete_task()

        if success:
            self.get_logger().info('Complete task executed successfully')
            return True
        else:
            self.get_logger().error('Complete task failed')
            return False

    def simulate_object_detection(self):
        """Simulate object detection"""
        # Simulate detecting various objects
        objects = ['table', 'chair', 'cup', 'bottle']
        detected = []

        for obj in objects:
            if random.random() > 0.3:  # 70% chance of detection
                detected.append(obj)

        return detected

    def simulate_navigation(self):
        """Simulate navigation to target"""
        # Simulate moving to a target location
        self.get_logger().info('Navigating to target location...')

        # Simulate navigation process
        for i in range(5):  # Simulate 5 steps
            self.get_logger().info(f'Navigation progress: {i+1}/5')
            time.sleep(0.5)

        return True

    def simulate_manipulation(self):
        """Simulate object manipulation"""
        # Simulate picking up an object
        self.get_logger().info('Attempting to grasp object...')

        # Simulate manipulation process
        time.sleep(2)

        # 90% success rate
        return random.random() > 0.1

    def execute_complete_task(self):
        """Execute a complete multi-step task"""
        self.get_logger().info('Executing complete multi-step task...')

        # Example task: "Go to the kitchen, find a red cup, and bring it to the table"
        steps = [
            ("Navigate to kitchen", self.navigate_to_kitchen),
            ("Detect red cup", self.detect_red_cup),
            ("Grasp red cup", self.grasp_red_cup),
            ("Navigate to table", self.navigate_to_table),
            ("Place cup on table", self.place_cup_on_table)
        ]

        for step_name, step_func in steps:
            self.get_logger().info(f'Executing: {step_name}')
            success = step_func()

            if not success:
                self.get_logger().error(f'Step failed: {step_name}')
                return False

            time.sleep(1)  # Small delay between steps

        return True

    def navigate_to_kitchen(self):
        """Navigate to kitchen"""
        self.get_logger().info('Navigating to kitchen...')
        time.sleep(2)
        return True

    def detect_red_cup(self):
        """Detect red cup"""
        self.get_logger().info('Detecting red cup...')
        time.sleep(1)
        return random.random() > 0.2  # 80% success

    def grasp_red_cup(self):
        """Grasp red cup"""
        self.get_logger().info('Grasping red cup...')
        time.sleep(2)
        return random.random() > 0.1  # 90% success

    def navigate_to_table(self):
        """Navigate to table"""
        self.get_logger().info('Navigating to table...')
        time.sleep(2)
        return True

    def place_cup_on_table(self):
        """Place cup on table"""
        self.get_logger().info('Placing cup on table...')
        time.sleep(1)
        return True

    def process_perception_phase(self, image_msg):
        """Process image during perception phase"""
        # In real implementation, this would run perception algorithms
        pass

    def process_navigation_phase(self, laser_msg):
        """Process laser data during navigation phase"""
        # In real implementation, this would use laser data for navigation
        pass

    def process_voice_command(self, audio_msg):
        """Process voice command"""
        # In real implementation, this would use speech recognition
        # For this project, we'll just acknowledge
        self.get_logger().info('Voice command received (simulated)')

    def complete_project(self):
        """Complete the capstone project successfully"""
        self.project_active = False
        self.get_logger().info('Capstone project completed successfully!')

        status_msg = String()
        status_msg.data = 'PROJECT_COMPLETED: All phases completed successfully'
        self.project_status_pub.publish(status_msg)

    def fail_project(self):
        """Fail the capstone project"""
        self.project_active = False
        self.get_logger().error('Capstone project failed!')

        status_msg = String()
        status_msg.data = 'PROJECT_FAILED: Project terminated due to error'
        self.project_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    project = CapstoneHumanoidProject()

    try:
        project.start_project()
        rclpy.spin(project)
    except KeyboardInterrupt:
        pass
    finally:
        project.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Summary and Key Takeaways

In this module, we've explored the Vision-Language-Action (VLA) paradigm for creating intelligent humanoid robots that can understand natural language commands, perceive their environment visually, and execute complex tasks. We've covered:

1. **VLA Architecture**: Understanding the integration of vision, language, and action modalities in robotic systems.

2. **Voice-to-Action Systems**: Implementing speech recognition using OpenAI Whisper and converting voice commands to robot actions.

3. **Natural Language Processing**: Using large language models to translate natural language instructions into executable ROS 2 action sequences.

4. **Vision Integration**: Implementing object detection, scene understanding, and spatial reasoning for robot perception.

5. **Multi-Step Task Execution**: Creating systems that can execute complex, multi-step tasks by combining perception, planning, and execution.

6. **Capstone Project**: Implementing a complete autonomous humanoid system that demonstrates the integration of all VLA components.

The VLA framework enables robots to perform complex tasks based on natural language instructions, making human-robot interaction more intuitive and natural. This represents a significant advancement in robotics, moving from pre-programmed behaviors to intelligent, responsive systems.

## References

Brown, T., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *International Conference on Machine Learning*.

Srivastava, N., et al. (2015). Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models. *Journal of Machine Learning Research*, 16, 1-48.

Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

Yin, P., et al. (2022). A Generalist Agent. *Transactions on Machine Learning Research*.

## Glossary

- **VLA (Vision-Language-Action)**: Framework integrating visual perception, language understanding, and physical action
- **Whisper**: OpenAI's automatic speech recognition system
- **LLM (Large Language Model)**: Advanced AI models for natural language processing
- **Multi-Modal Integration**: Combining multiple sensory inputs for robot decision making
- **Natural Language Understanding**: AI capability to interpret human language commands
- **Action Planning**: Converting high-level goals into executable robot actions
- **Perception-Action Loop**: Continuous cycle of sensing, understanding, and acting
- **Multi-Step Task Execution**: Performing complex tasks through sequential subtasks
- **Scene Understanding**: Comprehending the spatial and object layout of an environment
- **Spatial Reasoning**: Understanding and navigating physical space