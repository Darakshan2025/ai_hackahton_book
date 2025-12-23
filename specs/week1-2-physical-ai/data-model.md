# Data Model: Week 1-2 Physical AI Content

## Educational Content Structure

### Core Concepts Entity
- **Name**: Core Concepts
- **Fields**:
  - concept_name (string): Name of the core concept (embodied intelligence, Physical AI, etc.)
  - definition (string): Clear definition of the concept
  - theoretical_foundation (string): Theoretical background and origins
  - practical_application (string): Real-world applications and examples
  - learning_objectives (array): Specific learning outcomes for students
- **Relationships**: Connected to Examples and Applications entities

### Sensor Systems Entity
- **Name**: Sensor Systems
- **Fields**:
  - sensor_type (string): Type of sensor (LIDAR, camera, IMU, force/torque)
  - working_principle (string): How the sensor operates physically
  - applications (array): Specific applications in humanoid robotics
  - limitations (string): Constraints and challenges
  - integration_requirements (string): How the sensor integrates with the robot system
- **Relationships**: Connected to Robot Architecture entity

### Robot Architecture Entity
- **Name**: Robot Architecture
- **Fields**:
  - architecture_type (string): Humanoid, wheeled, manipulator, etc.
  - sensor_integration (string): How sensors are integrated into the system
  - processing_requirements (string): Computational needs for sensor data
  - control_systems (string): How the robot processes and responds to sensor data
- **Relationships**: Connected to Sensor Systems and AI Systems entities

### AI Systems Entity
- **Name**: AI Systems
- **Fields**:
  - system_type (string): Digital AI, Physical AI, embodied AI
  - interaction_model (string): How the system interacts with the environment
  - processing_approach (string): Computational approach used
  - embodiment_level (string): Degree of physical integration
- **Relationships**: Connected to Core Concepts and Robot Architecture entities

### Examples and Applications Entity
- **Name**: Examples and Applications
- **Fields**:
  - example_type (string): Practical, theoretical, or comparative example
  - description (string): Detailed explanation of the example
  - educational_value (string): What students learn from this example
  - complexity_level (string): Beginner, intermediate, or advanced
  - real_world_correlation (string): Connection to actual robotics applications
- **Relationships**: Connected to Core Concepts and Sensor Systems entities

## Validation Rules

1. **Content Completeness**: Each core concept must have a definition, theoretical foundation, and practical application
2. **Technical Accuracy**: All sensor descriptions must be technically accurate and verifiable
3. **Educational Appropriateness**: Content must be suitable for undergraduate-graduate STEM students
4. **Citation Requirement**: All claims must be supported by appropriate sources
5. **Diagram Inclusion**: Each major concept must include visual representation

## State Transitions

### Content Development State
- **Draft**: Initial content creation
- **Reviewed**: Content reviewed for technical accuracy
- **Validated**: Content validated for educational effectiveness
- **Published**: Content ready for student consumption