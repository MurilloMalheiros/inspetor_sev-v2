import queue
import unittest

from fatigue_monitor import DriverState, FatigueMonitor, FrameObservation, MonitorUpdate
from inspector_motorista import EyeStateFilter, FaceBoxFilter, SoundPlayer, alarm_decision, combined_blink_score


FACE_EYES = FrameObservation(True, True, True, False)
FACE_CLOSED_EYES = FrameObservation(True, False, False, False)
FACE_YAWNING = FrameObservation(True, True, True, True)


class FatigueMonitorTests(unittest.TestCase):
    def test_generated_alarm_tone_is_a_valid_cached_wav(self):
        first = SoundPlayer._tone_wav(1200, 100)
        second = SoundPlayer._tone_wav(1200, 100)
        self.assertTrue(first.startswith(b"RIFF"))
        self.assertIn(b"WAVE", first[:16])
        self.assertIs(first, second)

    def test_sound_alert_is_queued_for_the_persistent_mixer(self):
        player = SoundPlayer.__new__(SoundPlayer)
        player.enabled = True
        player._continuous = False
        player._items = queue.Queue(maxsize=4)
        player.play("startup")
        self.assertEqual(player._items.get_nowait(), "startup")

    def test_alarm_is_continuous_when_driver_falls_asleep(self):
        update = MonitorUpdate(
            state=DriverState.ASLEEP,
            fatigue=0,
            yawn_count=0,
            sleep_alarm=True,
        )
        self.assertEqual(alarm_decision(update), (True, None))

    def test_alarm_is_continuous_at_critical_fatigue(self):
        update = MonitorUpdate(
            state=DriverState.OK,
            fatigue=100,
            yawn_count=5,
            critical_alarm=True,
        )
        self.assertEqual(alarm_decision(update), (True, None))

    def test_new_yawn_and_fatigue_level_have_the_expected_sounds(self):
        yawn = MonitorUpdate(
            state=DriverState.OK,
            fatigue=20,
            yawn_count=1,
            new_yawn=True,
        )
        warning = MonitorUpdate(
            state=DriverState.OK,
            fatigue=65,
            yawn_count=4,
            new_yawn=True,
            fatigue_level_changed=True,
        )
        self.assertEqual(alarm_decision(yawn), (False, "yawn"))
        self.assertEqual(alarm_decision(warning), (False, "fatigue2"))

    def test_short_blink_does_not_trigger_alarm(self):
        monitor = FatigueMonitor(drowsy_seconds=0.8, sleep_seconds=2.0)
        monitor.update(FACE_EYES, 0.0)
        update = monitor.update(FACE_CLOSED_EYES, 0.1)
        update = monitor.update(FACE_CLOSED_EYES, 0.7)
        self.assertEqual(update.state, DriverState.OK)
        update = monitor.update(FACE_EYES, 0.75)
        self.assertEqual(update.state, DriverState.OK)

    def test_continuous_closed_eyes_triggers_sleep_alarm(self):
        monitor = FatigueMonitor(drowsy_seconds=0.8, sleep_seconds=2.0)
        monitor.update(FACE_CLOSED_EYES, 0.0)
        self.assertEqual(monitor.update(FACE_CLOSED_EYES, 1.0).state, DriverState.DROWSY)
        update = monitor.update(FACE_CLOSED_EYES, 2.1)
        self.assertEqual(update.state, DriverState.ASLEEP)
        self.assertTrue(update.sleep_alarm)

    def test_yawn_is_counted_once_until_mouth_closes(self):
        monitor = FatigueMonitor(yawn_seconds=0.9, fatigue_per_yawn=20, fatigue_decay_per_minute=0)
        monitor.update(FACE_YAWNING, 0.0)
        first = monitor.update(FACE_YAWNING, 1.0)
        second = monitor.update(FACE_YAWNING, 2.0)
        self.assertTrue(first.new_yawn)
        self.assertFalse(second.new_yawn)
        self.assertEqual(second.yawn_count, 1)
        self.assertEqual(second.fatigue, 20)

        monitor.update(FACE_EYES, 2.1)
        monitor.update(FACE_YAWNING, 3.0)
        third = monitor.update(FACE_YAWNING, 4.0)
        self.assertEqual(third.yawn_count, 2)
        self.assertEqual(third.fatigue, 40)

    def test_brief_mouth_detection_dropout_does_not_lose_yawn(self):
        monitor = FatigueMonitor(
            yawn_seconds=0.9,
            mouth_dropout_seconds=0.25,
            fatigue_decay_per_minute=0,
        )
        monitor.update(FACE_YAWNING, 0.0)
        monitor.update(FACE_YAWNING, 0.8)
        update = monitor.update(FACE_EYES, 0.95)
        self.assertTrue(update.new_yawn)
        self.assertEqual(update.yawn_count, 1)

    def test_long_mouth_detection_dropout_cancels_yawn(self):
        monitor = FatigueMonitor(yawn_seconds=0.9, mouth_dropout_seconds=0.25)
        monitor.update(FACE_YAWNING, 0.0)
        update = monitor.update(FACE_EYES, 0.5)
        self.assertFalse(update.new_yawn)
        update = monitor.update(FACE_YAWNING, 1.0)
        self.assertFalse(update.new_yawn)

    def test_missing_face_does_not_look_like_sleep(self):
        monitor = FatigueMonitor(no_face_seconds=2.5)
        no_face = FrameObservation(False, False, False, False)
        monitor.update(no_face, 0.0)
        update = monitor.update(no_face, 3.0)
        self.assertEqual(update.state, DriverState.DISTRACTED)
        self.assertFalse(update.sleep_alarm)

    def test_critical_fatigue_stays_at_100_until_reset(self):
        monitor = FatigueMonitor(
            yawn_seconds=0.1,
            fatigue_per_yawn=100,
            fatigue_decay_per_minute=60,
        )
        monitor.update(FACE_YAWNING, 0.0)
        update = monitor.update(FACE_YAWNING, 0.2)
        self.assertTrue(update.critical_alarm)
        self.assertEqual(update.fatigue, 100)

        update = monitor.update(FACE_EYES, 20.0)
        self.assertTrue(update.critical_alarm)
        self.assertEqual(update.fatigue, 100)

        monitor.reset_fatigue()
        update = monitor.update(FACE_EYES, 21.0)
        self.assertFalse(update.critical_alarm)
        self.assertEqual(update.fatigue, 0)

    def test_eye_filter_ignores_a_brief_bad_reading(self):
        eye = EyeStateFilter(0.55)
        self.assertTrue(eye.update(0.0, 0.00))
        self.assertTrue(eye.update(1.0, 0.03))
        self.assertTrue(eye.update(1.0, 0.06))
        self.assertTrue(eye.update(0.0, 0.09))

    def test_eye_filter_changes_only_after_stable_readings(self):
        eye = EyeStateFilter(0.55)
        for index in range(15):
            is_open = eye.update(1.0, index * 0.04)
        self.assertFalse(is_open)
        for index in range(15, 30):
            is_open = eye.update(0.0, index * 0.04)
        self.assertTrue(is_open)

    def test_combined_eye_signal_tolerates_one_weak_eye(self):
        self.assertGreater(combined_blink_score(0.10, 0.90), 0.55)
        self.assertGreater(combined_blink_score(0.95, 0.15), 0.55)
        self.assertLess(combined_blink_score(0.10, 0.20), 0.55)

    def test_face_box_filter_moves_without_jumping_to_raw_box(self):
        box_filter = FaceBoxFilter(smoothing=0.2)
        self.assertEqual(box_filter.update((0.2, 0.2, 0.6, 0.7)), (0.2, 0.2, 0.6, 0.7))
        smoothed = box_filter.update((0.4, 0.1, 0.9, 0.8))
        self.assertAlmostEqual(smoothed[0], 0.24)
        self.assertAlmostEqual(smoothed[2], 0.66)


if __name__ == "__main__":
    unittest.main()
