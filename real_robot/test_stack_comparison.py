"""Verify success-BC uses recorded commands, not stationary/measured-arm targets."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from real_robot.data import LEFT_COLUMNS
from real_robot.prepare_stack_comparison import cache_success


class SuccessfulRolloutTests(unittest.TestCase):
    def test_commands_are_zero_order_held_and_proprio_is_follower(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            take = root / "take"
            take.mkdir()
            for filename, times, values in (
                    ("follower_joint.csv", np.arange(0, 10, .01), lambda t: .9),
                    ("policy_commands.csv", np.arange(.5, 9.8, .08), lambda t: .1 if t < 5 else .4)):
                with (take / filename).open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["timestamp"] + LEFT_COLUMNS)
                    writer.writerows([[t] + [values(t)]*8 for t in times])
            for camera in ("chest", "left_wrist"):
                writer = cv2.VideoWriter(str(take / f"{camera}_rgb_test.avi"),
                                         cv2.VideoWriter_fourcc(*"XVID"), 15, (64, 64))
                for _ in range(150):
                    writer.write(np.zeros((64, 64, 3), np.uint8))
                writer.release()
            (take / "take_meta.json").write_text(json.dumps({
                "ended_at_epoch_s": 10., "evaluation": {"policy_ended_at_epoch_s": 9.9}}))
            info = cache_success(take, root / "cache")
            self.assertGreater(info["samples"], 0)
            actions = np.load(root / "cache/take/actions.npy")
            proprio = np.load(root / "cache/take/proprio.npy")
            self.assertEqual(actions.shape[1:], (30, 8))
            self.assertTrue(np.isin(actions, np.array([.1, .4], np.float32)).all())
            np.testing.assert_allclose(proprio, .9)


if __name__ == "__main__":
    unittest.main()
