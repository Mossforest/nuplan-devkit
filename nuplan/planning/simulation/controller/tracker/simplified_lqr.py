import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

@dataclass
class VehicleParameters:
    """Simple vehicle parameters class"""
    wheel_base: float  # [m] Distance between front and rear axle
    
class LateralStateIndex(IntEnum):
    """Index mapping for the lateral dynamics state vector."""
    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.

@dataclass
class SimplifiedTrajectory:
    """Simplified trajectory class with only essential attributes"""
    times: List[float]  # List of timestamps [s]
    positions: List[Tuple[float, float]]  # List of (x,y) positions [m]
    headings: List[float]  # List of headings [rad]
    velocities: List[float]  # List of velocities [m/s]
    curvatures: List[float]  # List of curvatures [1/m]


def angle_diff(angle1: float, angle2: float, period: float) -> float:
    """计算两个角度之间的最小差值"""
    diff = (angle1 - angle2) % period
    if diff > period / 2:
        diff -= period
    return diff

class SimplifiedLQRTracker:
    """
    Simplified LQR tracker that only implements the core trajectory tracking functionality.
    """
    def __init__(
        self,
        q_longitudinal: npt.NDArray[np.float64],  # Weights for longitudinal LQR
        r_longitudinal: npt.NDArray[np.float64],  # Input weights for longitudinal LQR
        q_lateral: npt.NDArray[np.float64],       # Weights for lateral LQR
        r_lateral: npt.NDArray[np.float64],       # Input weights for lateral LQR
        discretization_time: float,               # [s] Time interval for discretization
        tracking_horizon: int,                    # Number of steps to look ahead
        stopping_velocity: float,                 # [m/s] Velocity threshold for stopping
        stopping_proportional_gain: float,        # P gain for stopping controller
        vehicle: VehicleParameters,              # Vehicle parameters
    ):
        """Initialize the LQR tracker with control parameters"""
        # Longitudinal LQR Parameters
        assert len(q_longitudinal) == 1, "q_longitudinal should have 1 element (velocity)."
        assert len(r_longitudinal) == 1, "r_longitudinal should have 1 element (acceleration)."
        self._q_longitudinal = np.diag(q_longitudinal)
        self._r_longitudinal = np.diag(r_longitudinal)

        # Lateral LQR Parameters
        assert len(q_lateral) == 3, "q_lateral should have 3 elements (lateral_error, heading_error, steering_angle)."
        assert len(r_lateral) == 1, "r_lateral should have 1 element (steering_rate)."
        self._q_lateral = np.diag(q_lateral)
        self._r_lateral = np.diag(r_lateral)

        # Common parameters
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base
        self._stopping_velocity = stopping_velocity
        self._stopping_proportional_gain = stopping_proportional_gain

    def track_trajectory(
        self,
        current_time: float,  # Current time [s]
        initial_state: Tuple[float, float, float, float, float],  # (x, y, heading, velocity, steering_angle)
        trajectory: SimplifiedTrajectory,  # Reference trajectory to track
    ) -> Tuple[float, float]:  # Returns (acceleration [m/s^2], steering_rate [rad/s])
        """
        Track a given trajectory and compute control commands.
        
        Args:
            current_time: Current simulation time in seconds
            initial_state: Tuple of (x, y, heading, velocity, steering_angle)
            trajectory: Simplified trajectory to track
            
        Returns:
            Tuple of (acceleration command, steering_rate command)
        """
        # Extract initial state
        x, y, heading, velocity, steering_angle = initial_state
        
        # Find closest point on trajectory and compute errors
        closest_idx = self._find_closest_point(x, y, trajectory.positions)
        ref_x, ref_y = trajectory.positions[closest_idx]
        ref_heading = trajectory.headings[closest_idx]
        ref_velocity = trajectory.velocities[closest_idx]
        ref_curvature = trajectory.curvatures[closest_idx]
        
        # Compute lateral error (signed distance to reference path)
        dx = x - ref_x
        dy = y - ref_y
        lateral_error = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)
        heading_error = self._angle_diff(heading, ref_heading)
        
        # Initial state vectors
        initial_lateral_state = np.array([lateral_error, heading_error, steering_angle])
        
        # Check if stopping
        should_stop = ref_velocity <= self._stopping_velocity and velocity <= self._stopping_velocity
        
        if should_stop:
            # Use simple proportional controller for stopping
            accel_cmd = -self._stopping_proportional_gain * (velocity - ref_velocity)
            steering_rate_cmd = 0.0
        else:
            # Use LQR controllers
            # Longitudinal control
            accel_cmd = self._longitudinal_lqr_controller(velocity, ref_velocity)
            
            # Generate velocity profile for lateral control
            velocity_profile = np.ones(self._tracking_horizon) * velocity + \
                             np.arange(self._tracking_horizon) * self._discretization_time * accel_cmd
            
            # Generate curvature profile
            curvature_profile = np.ones(self._tracking_horizon) * ref_curvature
            
            # Lateral control
            steering_rate_cmd = self._lateral_lqr_controller(
                initial_lateral_state, velocity_profile, curvature_profile
            )
        
        return accel_cmd, steering_rate_cmd

    def _find_closest_point(self, x: float, y: float, reference_points: List[Tuple[float, float]]) -> int:
        """Find the index of the closest point on the reference trajectory"""
        distances = [(x - rx)**2 + (y - ry)**2 for rx, ry in reference_points]
        return np.argmin(distances)

    def _angle_diff(self, angle1: float, angle2: float) -> float:
        """Compute the shortest angle difference"""
        diff = (angle1 - angle2) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        return diff

    def _longitudinal_lqr_controller(self, initial_velocity: float, reference_velocity: float) -> float:
        """Compute acceleration command using LQR"""
        # 确保矩阵维度正确
        A = np.array([[1.0]])  # 2D array, shape (1,1)
        B = np.array([[self._tracking_horizon * self._discretization_time]])  # 2D array, shape (1,1)
        g = np.array([[0.0]])  # 2D array, shape (1,1)
        
        # 将标量转换为向量
        initial_state = np.array([[initial_velocity]])  # 2D array, shape (1,1)
        reference_state = np.array([[reference_velocity]])  # 2D array, shape (1,1)
        
        # 使用与原始 LQR 相同的计算逻辑
        state_error_zero_input = A @ initial_state + g - reference_state
        
        accel_cmd = -np.linalg.inv(B.T @ self._q_longitudinal @ B + self._r_longitudinal) @ \
                    B.T @ self._q_longitudinal @ state_error_zero_input
        
        return float(accel_cmd.item())

    def _lateral_lqr_controller(
        self,
        initial_lateral_state: npt.NDArray[np.float64],
        velocity_profile: npt.NDArray[np.float64],
        curvature_profile: npt.NDArray[np.float64],
    ) -> float:
        """Compute steering rate command using LQR"""
        n_states = len(LateralStateIndex)
        I = np.eye(n_states)
        
        A = I.copy()
        B = np.zeros((n_states, 1))
        g = np.zeros(n_states)
        
        # Build up matrices considering the full horizon
        for velocity, curvature in zip(velocity_profile, curvature_profile):
            state_matrix = np.eye(n_states)
            state_matrix[LateralStateIndex.LATERAL_ERROR, LateralStateIndex.HEADING_ERROR] = \
                velocity * self._discretization_time
            state_matrix[LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE] = \
                velocity * self._discretization_time / self._wheel_base
            
            input_matrix = np.zeros((n_states, 1))
            input_matrix[LateralStateIndex.STEERING_ANGLE] = self._discretization_time
            
            affine_term = np.zeros(n_states)
            affine_term[LateralStateIndex.HEADING_ERROR] = -velocity * curvature * self._discretization_time
            
            A = state_matrix @ A
            B = state_matrix @ B + input_matrix
            g = state_matrix @ g + affine_term
        
        # 计算状态误差（添加参考状态）
        reference_state = np.zeros(n_states)  # 目标是零误差状态
        state_error_zero_input = A @ initial_lateral_state + g - reference_state
        
        # 处理角度差异
        for angle_idx in [LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE]:
            state_error_zero_input[angle_idx] = angle_diff(
                state_error_zero_input[angle_idx], 0.0, 2 * np.pi
            )
        
        # 计算控制输入
        steering_rate_cmd = -np.linalg.inv(B.T @ self._q_lateral @ B + self._r_lateral) @ \
                        B.T @ self._q_lateral @ state_error_zero_input
        
        return float(steering_rate_cmd)
