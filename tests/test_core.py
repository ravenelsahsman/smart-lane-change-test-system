from __future__ import annotations

import math
import unittest

from src.lane_change_test_system.core import (
    DEFAULT_PAYLOAD,
    ChassisLoadParams,
    ControlParams,
    evaluate_feedback,
    generate_control_command,
    parse_payload,
    run_lane_change_test,
    sample_points_text,
    curvature_rate_threshold,
)


class CoreFormulaTests(unittest.TestCase):
    def test_curvature_rate_threshold_matches_plan_formula(self) -> None:
        params = ChassisLoadParams(
            rear_track_width=1.58,
            vehicle_mass=1420.0,
            center_of_mass_height=0.52,
            rear_roll_stiffness_distribution=0.55,
            load_transfer_damping=260000.0,
        )
        expected = 260000.0 * 1.58 / (1420.0 * 0.52 * 0.55 * 12.0**3)
        self.assertAlmostEqual(curvature_rate_threshold(12.0, params), expected)

    def test_feedback_update_matches_plan_formula(self) -> None:
        payload = DEFAULT_PAYLOAD | {"lateralError": 0.5, "yawRateError": 0.4}
        test_input = parse_payload(payload | {"pointsText": sample_points_text()})
        result = evaluate_feedback(0.5, 0.4, test_input.spatial_range, test_input.feedback)
        expected_f = math.exp(-1.4 * 0.5**2 - 2.2 * 0.4**2)
        expected_l = min(12.0, 5.0 + 2.0 * (0.82 - expected_f))
        self.assertAlmostEqual(result.f_opt, expected_f)
        self.assertAlmostEqual(result.new_spatial_range, expected_l)
        self.assertTrue(result.updated)

    def test_control_command_matches_bicycle_and_pwm_formula(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.4), (3.0, 0.9)]
        curvatures = [0.02, 0.03, 0.04, 0.05]
        params = ControlParams(
            wheelbase=2.72,
            pwm_gain=0.9,
            duty_center=7.5,
            lookahead_distance=1.0,
        )
        command = generate_control_command(points, curvatures, params)
        expected_delta = math.atan(2.72 * command.target_curvature)
        self.assertAlmostEqual(command.steering_angle_rad, expected_delta)
        self.assertAlmostEqual(command.pwm_duty, 0.9 * expected_delta + 7.5)

    def test_full_pipeline_returns_all_project_book_outputs(self) -> None:
        payload = DEFAULT_PAYLOAD | {"pointsText": sample_points_text()}
        result = run_lane_change_test(parse_payload(payload))
        self.assertEqual(len(result.raw_points), len(result.target_points))
        self.assertEqual(len(result.raw_curvature), len(result.raw_points))
        self.assertEqual(len(result.raw_curvature_rate), len(result.raw_points) - 1)
        self.assertGreater(result.gamma_max, 0.0)
        self.assertIsNotNone(result.control_command)
        self.assertIsNotNone(result.feedback)

    def test_reconstructed_windows_keep_endpoint_continuity(self) -> None:
        payload = DEFAULT_PAYLOAD | {"pointsText": sample_points_text()}
        result = run_lane_change_test(parse_payload(payload))
        rebuilt_reports = [report for report in result.reversal_reports if report.reconstructed]
        self.assertGreater(len(rebuilt_reports), 0)
        for report in rebuilt_reports:
            raw_end = result.raw_points[report.end_index]
            target_end = result.target_points[report.end_index]
            self.assertAlmostEqual(raw_end[0], target_end[0], places=9)
            self.assertAlmostEqual(raw_end[1], target_end[1], places=9)

        raw_rate_peak = max(abs(value) for value in result.raw_curvature_rate)
        target_rate_peak = max(abs(value) for value in result.target_curvature_rate)
        self.assertLess(target_rate_peak, raw_rate_peak * 5.0)


if __name__ == "__main__":
    unittest.main()
