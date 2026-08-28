#!/usr/bin/env python3
"""
Ball Trajectory Smoothing via Kalman Filtering (Phase 3, sections 3.5-3.6)

Constant-velocity Kalman filter over the image-plane ball position. Tracks
state S_t = [x, y, vx, vy]. On frames where the detector finds the ball, the
filter is updated with the measurement (position_source='detected'). On
frames where the detector misses the ball, the filter's own motion-model
prediction is used instead (position_source='predicted'), up to
max_predict_gap consecutive missed frames -- after that the track is
considered lost and no position is reported until the ball is re-detected.

Pure numpy, no video/model dependency, so it is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    position_source: str  # 'detected' or 'predicted'


class BallKalmanFilter:
    """Constant-velocity 2D Kalman filter for ball position tracking.

    State vector: [x, y, vx, vy]^T
    Measurement:  [x, y]^T (detector box center)
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 4.0,
                 max_predict_gap: int = 15):
        self.max_predict_gap = max_predict_gap
        self._initialized = False
        self._missed_streak = 0

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1000.0  # large initial uncertainty
        self.x = np.zeros((4, 1))

    def reset(self):
        self._initialized = False
        self._missed_streak = 0
        self.P = np.eye(4) * 1000.0
        self.x = np.zeros((4, 1))

    def _predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(self, measurement: Optional[tuple[float, float]]) -> Optional[BallState]:
        """Advance the filter by one frame.

        Parameters
        ----------
        measurement : (x, y) detector ball center for this frame, or None if
            the ball was not detected this frame.

        Returns
        -------
        BallState with the current estimate and its source, or None if the
        ball has not yet been seen, or has been missing for longer than
        max_predict_gap frames (track considered lost).
        """
        if measurement is None:
            if not self._initialized:
                return None
            self._missed_streak += 1
            if self._missed_streak > self.max_predict_gap:
                return None
            self._predict()
            return BallState(float(self.x[0, 0]), float(self.x[1, 0]),
                             float(self.x[2, 0]), float(self.x[3, 0]), "predicted")

        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        if not self._initialized:
            self.x[0, 0], self.x[1, 0] = measurement[0], measurement[1]
            self.x[2, 0], self.x[3, 0] = 0.0, 0.0
            self.P = np.eye(4) * 100.0
            self._initialized = True
            self._missed_streak = 0
            return BallState(measurement[0], measurement[1], 0.0, 0.0, "detected")

        self._predict()
        self._update(z)
        self._missed_streak = 0
        return BallState(float(self.x[0, 0]), float(self.x[1, 0]),
                         float(self.x[2, 0]), float(self.x[3, 0]), "detected")
