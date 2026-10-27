"""State machine used by the driver monitoring application.

The neural network detects objects in individual frames.  This module turns
those detections into events that need a notion of time: sleeping, yawning and
accumulated fatigue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriverState(str, Enum):
    OK = "OK"
    DISTRACTED = "ROSTO NAO DETECTADO"
    DROWSY = "SONOLENCIA"
    ASLEEP = "PERIGO: MOTORISTA DORMINDO"


@dataclass(frozen=True)
class FrameObservation:
    face: bool
    left_eye: bool
    right_eye: bool
    mouth_open: bool


@dataclass(frozen=True)
class MonitorUpdate:
    state: DriverState
    fatigue: float
    yawn_count: int
    new_yawn: bool = False
    fatigue_level_changed: bool = False
    sleep_alarm: bool = False
    camera_warning: bool = False
    critical_alarm: bool = False
    eyes_closed_seconds: float = 0.0


class FatigueMonitor:
    """Temporal filter for YOLO detections.

    Eye detections are allowed to disappear briefly because blinking and a
    single bad inference are normal.  A yawn is counted only after the mouth
    has stayed open for ``yawn_seconds`` and must close before another yawn can
    be counted.
    """

    def __init__(
        self,
        *,
        sleep_seconds: float = 2.0,
        drowsy_seconds: float = 0.8,
        yawn_seconds: float = 0.9,
        mouth_dropout_seconds: float = 0.25,
        no_face_seconds: float = 2.5,
        fatigue_per_yawn: float = 18.0,
        fatigue_decay_per_minute: float = 1.0,
        alarm_repeat_seconds: float = 2.0,
    ) -> None:
        if not 0 < drowsy_seconds < sleep_seconds:
            raise ValueError("drowsy_seconds must be between 0 and sleep_seconds")
        if yawn_seconds <= 0 or mouth_dropout_seconds < 0 or no_face_seconds <= 0:
            raise ValueError("time thresholds must be positive")

        self.sleep_seconds = sleep_seconds
        self.drowsy_seconds = drowsy_seconds
        self.yawn_seconds = yawn_seconds
        self.mouth_dropout_seconds = mouth_dropout_seconds
        self.no_face_seconds = no_face_seconds
        self.fatigue_per_yawn = fatigue_per_yawn
        self.decay_per_second = fatigue_decay_per_minute / 60.0
        self.alarm_repeat_seconds = alarm_repeat_seconds

        self.fatigue = 0.0
        self.yawn_count = 0
        self._last_time: float | None = None
        self._eyes_closed_since: float | None = None
        self._face_missing_since: float | None = None
        self._mouth_open_since: float | None = None
        self._mouth_last_seen: float | None = None
        self._mouth_missing_since: float | None = None
        self._yawn_latched = False
        self._last_sleep_alarm = float("-inf")
        self._last_fatigue_level = 0
        self._critical_alarm = False

    @staticmethod
    def fatigue_level(fatigue: float) -> int:
        if fatigue >= 85:
            return 3
        if fatigue >= 65:
            return 2
        if fatigue >= 40:
            return 1
        return 0

    def reset_fatigue(self) -> None:
        self.fatigue = 0.0
        self.yawn_count = 0
        self._last_fatigue_level = 0
        self._critical_alarm = False

    def update(self, observation: FrameObservation, now: float) -> MonitorUpdate:
        if self._last_time is None:
            dt = 0.0
        else:
            dt = max(0.0, min(now - self._last_time, 1.0))
        self._last_time = now

        # A very slow decay prevents yawns from an hour ago from dominating,
        # while still ensuring fatigue mostly accumulates during a trip.
        if not self._critical_alarm:
            self.fatigue = max(0.0, self.fatigue - self.decay_per_second * dt)

        if observation.face:
            self._face_missing_since = None
        elif self._face_missing_since is None:
            self._face_missing_since = now

        both_eyes_closed = (
            observation.face
            and not observation.left_eye
            and not observation.right_eye
        )
        if both_eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
        else:
            self._eyes_closed_since = None

        new_yawn = False
        mouth_detected = observation.face and observation.mouth_open
        if mouth_detected:
            # Se a deteccao voltou apenas depois da tolerancia, esta e uma
            # nova abertura da boca e deve liberar a contagem anterior.
            if (
                self._mouth_missing_since is not None
                and now - self._mouth_missing_since > self.mouth_dropout_seconds
            ):
                self._mouth_open_since = None
                self._yawn_latched = False
            self._mouth_missing_since = None
            self._mouth_last_seen = now
            if self._mouth_open_since is None:
                self._mouth_open_since = now
        elif self._mouth_missing_since is None:
            self._mouth_missing_since = now

        in_dropout_grace = (
            observation.face
            and self._mouth_missing_since is not None
            and now - self._mouth_missing_since <= self.mouth_dropout_seconds
        )
        if self._mouth_open_since is not None and (mouth_detected or in_dropout_grace):
            if (
                not self._yawn_latched
                and now - self._mouth_open_since >= self.yawn_seconds
            ):
                self._yawn_latched = True
                self.yawn_count += 1
                self.fatigue = min(100.0, self.fatigue + self.fatigue_per_yawn)
                if self.fatigue >= 100.0:
                    # Once the critical limit is reached it remains active
                    # until the driver explicitly confirms a rest with R.
                    self.fatigue = 100.0
                    self._critical_alarm = True
                new_yawn = True
        else:
            self._mouth_open_since = None
            self._mouth_last_seen = None
            self._mouth_missing_since = None
            self._yawn_latched = False

        no_face_duration = (
            now - self._face_missing_since
            if self._face_missing_since is not None
            else 0.0
        )
        eyes_closed_duration = (
            now - self._eyes_closed_since
            if self._eyes_closed_since is not None
            else 0.0
        )

        if eyes_closed_duration >= self.sleep_seconds:
            state = DriverState.ASLEEP
        elif eyes_closed_duration >= self.drowsy_seconds:
            state = DriverState.DROWSY
        elif no_face_duration >= self.no_face_seconds:
            state = DriverState.DISTRACTED
        else:
            state = DriverState.OK

        sleep_alarm = False
        if state is DriverState.ASLEEP and now - self._last_sleep_alarm >= self.alarm_repeat_seconds:
            sleep_alarm = True
            self._last_sleep_alarm = now

        level = self.fatigue_level(self.fatigue)
        level_changed = level > self._last_fatigue_level
        self._last_fatigue_level = level

        return MonitorUpdate(
            state=state,
            fatigue=self.fatigue,
            yawn_count=self.yawn_count,
            new_yawn=new_yawn,
            fatigue_level_changed=level_changed,
            sleep_alarm=sleep_alarm,
            camera_warning=(state is DriverState.DISTRACTED),
            critical_alarm=self._critical_alarm,
            eyes_closed_seconds=eyes_closed_duration,
        )
