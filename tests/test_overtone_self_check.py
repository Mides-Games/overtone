"""The engine's own verdict on a timing, without a map (roadmap 29.5)."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from engine_support import (Analysis, TimingPoint, _click_track, analyze_audio,
                            setUpModule, tearDownModule)  # noqa: F401
import overtone as ov


def _attack_analysis(lines, duration=60.0, bpm=120.0, start=0.25) -> Analysis:
    """Clicks every beat at ``bpm`` from ``start``, under the given red lines
    (offset_ms, bpm): no engine run, so the grid is exactly what is asked."""
    times = np.arange(start, duration - 0.2, 60.0 / bpm)
    points = [TimingPoint(float(ms), float(b), 0.95, n) for n, (ms, b) in enumerate(lines)]
    return Analysis(
        source="test", duration=duration, beats=times, local_bpms=np.full(times.size, bpm),
        points=points, hop_length=512, sample_rate=44100, subdivision=1.0,
        global_bpm=bpm, stability=0.95, engine="precision",
        attack_times=times, attack_weights=np.ones(times.size))


class SelfCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "click.wav"
            _click_track(path, 120.0, duration=60.0)
            cls.analysis = analyze_audio(str(path))

    def test_a_click_track_on_its_grid_is_sure(self):
        result = ov.self_check(self.analysis)
        self.assertEqual(result["verdict"], "sure", result)
        self.assertIsNone(result["look"])
        self.assertFalse(result["calibrated"])
        line = result["sections"][0]
        self.assertLess(line["p95_ms"], 1.0)
        self.assertLess(line["drift_span_ms"], 1.0)

    def test_the_grid_half_a_beat_out_is_looked_at(self):
        point = self.analysis.points[0]
        moved = replace(self.analysis, points=[replace(point, offset_ms=point.offset_ms + 250.0)])
        result = ov.self_check(moved)
        self.assertEqual(result["verdict"], "look")
        self.assertEqual(result["look"]["reason"], "offbeat")
        self.assertEqual(result["look"]["index"], 0)
        self.assertAlmostEqual(result["look"]["start_s"], (point.offset_ms + 250.0) / 1000.0, places=3)

    def test_a_bpm_three_thousandths_out_drifts(self):
        point = self.analysis.points[0]
        # The offset centred so the error crosses zero mid-song, as a fit would leave it.
        fast = replace(self.analysis, points=[replace(point, bpm=point.bpm * 1.003,
                                                      offset_ms=point.offset_ms + 90.0)])
        result = ov.self_check(fast)
        self.assertEqual(result["verdict"], "look")
        self.assertIn("drift", result["sections"][0]["reasons"])
        self.assertGreater(abs(result["sections"][0]["drift_ms_per_min"]), 100.0)

    def test_the_wrong_red_line_is_the_one_named(self):
        good = _attack_analysis([(250.0, 120.0), (30250.0, 120.0)])
        self.assertEqual(ov.self_check(good)["verdict"], "sure")
        bad = _attack_analysis([(250.0, 120.0), (30250.0, 120.36)])
        result = ov.self_check(bad)
        self.assertEqual(result["verdict"], "look")
        self.assertEqual(result["look"]["index"], 1)
        self.assertEqual((result["look"]["start_s"], result["look"]["end_s"]), (30.25, 60.0))
        self.assertEqual(result["sections"][0]["reasons"], [])

    def test_a_refusal_and_an_empty_analysis_are_looked_at(self):
        refused = ov.self_check(None)
        self.assertEqual((refused["verdict"], refused["look"]["reason"]), ("look", "refused"))
        empty = ov.self_check(_attack_analysis([]))
        self.assertEqual((empty["verdict"], empty["look"]["reason"]), ("look", "no_red_lines"))
        self.assertEqual(empty["sections"], [])

    def test_a_line_without_attacks_or_from_the_fallback_tracker_is_looked_at(self):
        bare = replace(_attack_analysis([(250.0, 120.0)]), attack_times=np.zeros(0),
                       attack_weights=np.zeros(0))
        self.assertEqual(ov.self_check(bare)["look"]["reason"], "few_attacks")
        legacy = replace(_attack_analysis([(250.0, 120.0)]), engine="legacy")
        self.assertEqual(ov.self_check(legacy)["look"]["reason"], "fallback_tracker")

    def test_it_moves_nothing(self):
        before = [(p.offset_ms, p.bpm) for p in self.analysis.points]
        ov.self_check(self.analysis)
        self.assertEqual([(p.offset_ms, p.bpm) for p in self.analysis.points], before)


if __name__ == "__main__":
    unittest.main()
