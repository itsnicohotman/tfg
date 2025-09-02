from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Command:
    command: str
    coordinates: Optional[Tuple[float, float, float]] = None


@dataclass
class ControlMessage:
    grasp: bool  # True for grasping, False for releasing
    referenceX: Optional[float]
    referenceY: Optional[float]
    referenceZ: Optional[float]


@dataclass
class SensorData:
    timestamp: float  # Timestamp of the current simulation moment
    timestep: float  # Simulation's timestep
    sensorValues: (
        dict  # Dictionary with sensor values: key=sensor's name, value=sensor's value
    )
    dronePosition: list[float, float, float]  # Drone's position


@dataclass
class RobotSensorData:
    timestamp: float  # Timestamp of the current simulation moment
    timestep: float  # Simulation's timestep
    sensorValues: (
        dict  # Dictionary with sensor values: key=sensor's name, value=sensor's value
    )
    manipulatorPosition: list[float, float, float]  # Manipulator's position
    dronePosition: list[float, float, float]  # Drone's position
    jointsPosition: list[
        float, float, float, float, float, float, float, float
    ]  # Joints position (qpos)
    jointsVelocity: list[
        float, float, float, float, float, float, float, float
    ]  # Joints velocity (qvel)
    # droneAcceleration: list[float, float, float]  # Drone's linear acceleration
    # armsHooks: list  # List with anchor points on the arms' side
    # droneHooks: list  # List with anchor points on the drone' side
    qCombinationCase: int


@dataclass
class ControlAction:
    droneThrusters: dict  # Dictionary with drone's control values: key=thruster name, value=action value


@dataclass
class RobotControlAction:
    jointMotors: dict  # Dictionary with manipulator's control values: key=motor name, value=action value
