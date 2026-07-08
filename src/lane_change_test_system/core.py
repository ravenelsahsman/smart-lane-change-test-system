from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


Point = tuple[float, float]
EPSILON = 1e-12


@dataclass(frozen=True)
class ChassisLoadParams:
    rear_track_width: float
    vehicle_mass: float
    center_of_mass_height: float
    rear_roll_stiffness_distribution: float
    load_transfer_damping: float


@dataclass(frozen=True)
class ControlParams:
    wheelbase: float
    pwm_gain: float
    duty_center: float
    lookahead_distance: float


@dataclass(frozen=True)
class FeedbackParams:
    lambda_lateral: float
    lambda_yaw_rate: float
    confidence_threshold: float
    expansion_coefficient: float
    max_spatial_range: float


@dataclass(frozen=True)
class TestInput:
    points: list[Point]
    speed: float
    spatial_range: float
    delta_t: float
    chassis: ChassisLoadParams
    control: ControlParams
    feedback: FeedbackParams
    lateral_error: float
    yaw_rate_error: float


@dataclass
class ReversalReport:
    index: int
    start_index: int
    end_index: int
    gamma_raw: float
    gamma_max: float
    reconstructed: bool
    candidate_rate_count: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "startIndex": self.start_index,
            "endIndex": self.end_index,
            "gammaRaw": self.gamma_raw,
            "gammaRawAbs": abs(self.gamma_raw),
            "gammaMax": self.gamma_max,
            "reconstructed": self.reconstructed,
            "candidateRateCount": self.candidate_rate_count,
        }


@dataclass(frozen=True)
class ControlCommand:
    target_curvature: float
    steering_angle_rad: float
    pwm_duty: float

    def to_dict(self) -> dict:
        return {
            "targetCurvature": self.target_curvature,
            "steeringAngleRad": self.steering_angle_rad,
            "steeringAngleDeg": math.degrees(self.steering_angle_rad),
            "pwmDuty": self.pwm_duty,
        }


@dataclass(frozen=True)
class FeedbackResult:
    f_opt: float
    old_spatial_range: float
    new_spatial_range: float
    updated: bool

    def to_dict(self) -> dict:
        return {
            "fOpt": self.f_opt,
            "oldSpatialRange": self.old_spatial_range,
            "newSpatialRange": self.new_spatial_range,
            "updated": self.updated,
        }


@dataclass
class TestResult:
    raw_points: list[Point]
    target_points: list[Point]
    raw_curvature: list[float]
    raw_curvature_rate: list[float]
    target_curvature: list[float]
    target_curvature_rate: list[float]
    gamma_max: float
    threshold_lookup: list[dict]
    reversal_reports: list[ReversalReport]
    control_command: ControlCommand
    feedback: FeedbackResult

    def to_dict(self) -> dict:
        return {
            "rawPoints": [{"x": x, "y": y} for x, y in self.raw_points],
            "targetPoints": [{"x": x, "y": y} for x, y in self.target_points],
            "rawCurvature": self.raw_curvature,
            "rawCurvatureRate": self.raw_curvature_rate,
            "targetCurvature": self.target_curvature,
            "targetCurvatureRate": self.target_curvature_rate,
            "gammaMax": self.gamma_max,
            "thresholdLookup": self.threshold_lookup,
            "reversalReports": [report.to_dict() for report in self.reversal_reports],
            "controlCommand": self.control_command.to_dict(),
            "feedback": self.feedback.to_dict(),
        }


DEFAULT_PAYLOAD = {
    "speed": 12.0,
    "spatialRange": 5.0,
    "deltaT": 1.0,
    "rearTrackWidth": 1.58,
    "vehicleMass": 1420.0,
    "centerOfMassHeight": 0.52,
    "rearRollStiffnessDistribution": 0.55,
    "loadTransferDamping": 2000.0,
    "wheelbase": 2.72,
    "pwmGain": 0.9,
    "dutyCenter": 7.5,
    "lookaheadDistance": 8.0,
    "lambdaLateral": 1.4,
    "lambdaYawRate": 2.2,
    "confidenceThreshold": 0.82,
    "expansionCoefficient": 2.0,
    "maxSpatialRange": 12.0,
    "lateralError": 0.42,
    "yawRateError": 0.36,
}


def sample_points() -> list[Point]:
    """Only a UI bootstrap input; the algorithm itself consumes T_raw points."""
    points: list[Point] = []
    for i in range(51):
        x = i * 1.2
        y = 3.5 * math.sin(2.0 * math.pi * (x + 0.6) / 40.0)
        points.append((x, y))
    return points


def sample_points_text() -> str:
    return "\n".join(f"{x:.6f},{y:.6f}" for x, y in sample_points())


def parse_points_text(text: str) -> list[Point]:
    points: list[Point] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.replace("，", ",").replace(";", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"第 {line_no} 行轨迹点必须为 x,y 格式")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(f"第 {line_no} 行轨迹点包含非数字内容") from exc
    if len(points) < 3:
        raise ValueError("预选变道轨迹点集至少需要 3 个点以执行三点差分")
    return points


def parse_payload(payload: dict) -> TestInput:
    points_text = str(payload.get("pointsText", "")).strip()
    points = parse_points_text(points_text)
    speed = _positive(payload, "speed", "纵向行驶速度 v")
    spatial_range = _positive(payload, "spatialRange", "预设空间范围 L_old")
    delta_t = _positive(payload, "deltaT", "三点差分 Δt")
    chassis = ChassisLoadParams(
        rear_track_width=_positive(payload, "rearTrackWidth", "后轴轮距 t_w"),
        vehicle_mass=_positive(payload, "vehicleMass", "整车质量 m"),
        center_of_mass_height=_positive(payload, "centerOfMassHeight", "质心高度 h_g"),
        rear_roll_stiffness_distribution=_positive(
            payload, "rearRollStiffnessDistribution", "后轴侧倾刚度分配系数 D_r"
        ),
        load_transfer_damping=_positive(payload, "loadTransferDamping", "载荷转移临界阻尼 K_damp"),
    )
    control = ControlParams(
        wheelbase=_positive(payload, "wheelbase", "车辆轴距 L"),
        pwm_gain=float(payload.get("pwmGain", 0.0)),
        duty_center=float(payload.get("dutyCenter", 0.0)),
        lookahead_distance=_positive(payload, "lookaheadDistance", "当前控制周期前瞻距离"),
    )
    feedback = FeedbackParams(
        lambda_lateral=_non_negative(payload, "lambdaLateral", "位移惩罚权重 λ1"),
        lambda_yaw_rate=_non_negative(payload, "lambdaYawRate", "偏航角惩罚权重 λ2"),
        confidence_threshold=_positive(payload, "confidenceThreshold", "置信度阈值 F_th"),
        expansion_coefficient=_non_negative(payload, "expansionCoefficient", "空间膨胀系数 ρ"),
        max_spatial_range=_positive(payload, "maxSpatialRange", "最大预设空间范围 L_max"),
    )
    if feedback.confidence_threshold > 1.0:
        raise ValueError("置信度阈值 F_th 应位于 0 到 1 之间")
    return TestInput(
        points=points,
        speed=speed,
        spatial_range=spatial_range,
        delta_t=delta_t,
        chassis=chassis,
        control=control,
        feedback=feedback,
        lateral_error=float(payload.get("lateralError", 0.0)),
        yaw_rate_error=float(payload.get("yawRateError", 0.0)),
    )


def run_lane_change_test(test_input: TestInput) -> TestResult:
    raw_points = list(test_input.points)
    raw_curvature = curvature_sequence(raw_points, test_input.delta_t)
    raw_rates = curvature_rate_sequence(raw_points, raw_curvature)
    gamma_max = curvature_rate_threshold(test_input.speed, test_input.chassis)
    threshold_lookup = build_threshold_lookup(test_input.chassis, test_input.speed)

    reports = detect_reversal_reports(
        raw_points,
        raw_curvature,
        raw_rates,
        spatial_range=test_input.spatial_range,
        gamma_max=gamma_max,
    )

    target_points = list(raw_points)
    for report in reports:
        if abs(report.gamma_raw) > gamma_max:
            current_curvature = curvature_sequence(target_points, test_input.delta_t)
            current_rates = curvature_rate_sequence(target_points, current_curvature)
            target_points = reconstruct_window(
                target_points,
                current_curvature,
                current_rates,
                report.start_index,
                report.end_index,
                gamma_max,
            )
            report.reconstructed = True

    target_curvature = curvature_sequence(target_points, test_input.delta_t)
    target_rates = curvature_rate_sequence(target_points, target_curvature)
    command = generate_control_command(target_points, target_curvature, test_input.control)
    feedback = evaluate_feedback(
        test_input.lateral_error,
        test_input.yaw_rate_error,
        test_input.spatial_range,
        test_input.feedback,
    )

    return TestResult(
        raw_points=raw_points,
        target_points=target_points,
        raw_curvature=raw_curvature,
        raw_curvature_rate=raw_rates,
        target_curvature=target_curvature,
        target_curvature_rate=target_rates,
        gamma_max=gamma_max,
        threshold_lookup=threshold_lookup,
        reversal_reports=reports,
        control_command=command,
        feedback=feedback,
    )


def curvature_sequence(points: Sequence[Point], delta_t: float = 1.0) -> list[float]:
    if len(points) < 3:
        raise ValueError("三点差分需要至少 3 个轨迹点")
    if delta_t <= 0:
        raise ValueError("Δt 必须大于 0")

    curvatures = [0.0 for _ in points]
    for i in range(1, len(points) - 1):
        x_prev, y_prev = points[i - 1]
        x_curr, y_curr = points[i]
        x_next, y_next = points[i + 1]
        x_prime = (x_next - x_prev) / (2.0 * delta_t)
        y_prime = (y_next - y_prev) / (2.0 * delta_t)
        x_double_prime = (x_next - 2.0 * x_curr + x_prev) / (delta_t**2)
        y_double_prime = (y_next - 2.0 * y_curr + y_prev) / (delta_t**2)
        denominator = (x_prime**2 + y_prime**2) ** 1.5
        if denominator <= EPSILON:
            curvatures[i] = 0.0
        else:
            curvatures[i] = (x_prime * y_double_prime - y_prime * x_double_prime) / denominator

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures


def curvature_rate_sequence(points: Sequence[Point], curvatures: Sequence[float]) -> list[float]:
    if len(points) != len(curvatures):
        raise ValueError("轨迹点集与曲率序列长度必须一致")

    rates: list[float] = []
    for i in range(len(points) - 1):
        ds = distance(points[i], points[i + 1])
        if ds <= EPSILON:
            raise ValueError(f"第 {i} 与第 {i + 1} 个轨迹点重合，无法计算物理弧长差值")
        rates.append((curvatures[i + 1] - curvatures[i]) / ds)
    return rates


def detect_reversal_reports(
    points: Sequence[Point],
    curvatures: Sequence[float],
    rates: Sequence[float],
    spatial_range: float,
    gamma_max: float,
) -> list[ReversalReport]:
    reports: list[ReversalReport] = []
    for i in range(len(curvatures) - 1):
        if curvatures[i] * curvatures[i + 1] < 0.0:
            start, end = spatial_window_indices(points, i, spatial_range)
            rate_start = min(start, len(rates) - 1)
            rate_end_exclusive = max(rate_start + 1, min(end, len(rates)))
            candidate_rates = list(rates[rate_start:rate_end_exclusive])
            gamma_raw = max(candidate_rates, key=lambda value: abs(value))
            reports.append(
                ReversalReport(
                    index=i,
                    start_index=start,
                    end_index=end,
                    gamma_raw=gamma_raw,
                    gamma_max=gamma_max,
                    reconstructed=False,
                    candidate_rate_count=len(candidate_rates),
                )
            )
    return reports


def spatial_window_indices(points: Sequence[Point], center_index: int, spatial_range: float) -> tuple[int, int]:
    if spatial_range <= 0:
        raise ValueError("预设空间范围必须大于 0")
    arcs = cumulative_arc_lengths(points)
    center_s = arcs[center_index]
    left_s = center_s - spatial_range
    right_s = center_s + spatial_range
    start = center_index
    while start > 0 and arcs[start - 1] >= left_s:
        start -= 1
    end = center_index
    while end < len(points) - 1 and arcs[end + 1] <= right_s:
        end += 1
    return start, end


def curvature_rate_threshold(speed: float, params: ChassisLoadParams) -> float:
    if speed <= 0:
        raise ValueError("纵向行驶速度 v 必须大于 0")
    denominator = (
        params.vehicle_mass
        * params.center_of_mass_height
        * params.rear_roll_stiffness_distribution
        * speed**3
    )
    return params.load_transfer_damping * params.rear_track_width / denominator


def build_threshold_lookup(params: ChassisLoadParams, current_speed: float) -> list[dict]:
    upper = max(20.0, math.ceil(current_speed + 8.0))
    speeds = [max(0.5, value) for value in range(1, int(upper) + 1)]
    return [
        {"speed": float(speed), "gammaMax": curvature_rate_threshold(float(speed), params)}
        for speed in speeds
    ]


def reconstruct_window(
    points: Sequence[Point],
    curvatures: Sequence[float],
    rates: Sequence[float],
    start_index: int,
    end_index: int,
    gamma_max: float,
) -> list[Point]:
    if start_index >= end_index:
        return list(points)

    safe_rates = list(rates)
    rate_end = min(end_index, len(rates))
    for rate_index in range(start_index, rate_end):
        if abs(safe_rates[rate_index]) > gamma_max:
            safe_rates[rate_index] = math.copysign(gamma_max, safe_rates[rate_index])

    safe_curvatures = list(curvatures)
    for point_index in range(start_index + 1, end_index + 1):
        ds = distance(points[point_index - 1], points[point_index])
        safe_curvatures[point_index] = (
            safe_curvatures[point_index - 1] + safe_rates[point_index - 1] * ds
        )

    reconstructed = list(points)
    x, y = points[start_index]
    theta = initial_heading(points, start_index)
    reconstructed[start_index] = (x, y)
    for point_index in range(start_index + 1, end_index + 1):
        ds = distance(points[point_index - 1], points[point_index])
        theta += safe_curvatures[point_index - 1] * ds
        x += math.cos(theta) * ds
        y += math.sin(theta) * ds
        reconstructed[point_index] = (x, y)
    return reconstructed


def generate_control_command(
    target_points: Sequence[Point],
    target_curvatures: Sequence[float],
    params: ControlParams,
) -> ControlCommand:
    target_curvature = interpolate_curvature_at_distance(
        target_points, target_curvatures, params.lookahead_distance
    )
    steering_angle = math.atan(params.wheelbase * target_curvature)
    duty = params.pwm_gain * steering_angle + params.duty_center
    return ControlCommand(target_curvature, steering_angle, duty)


def evaluate_feedback(
    lateral_error: float,
    yaw_rate_error: float,
    old_spatial_range: float,
    params: FeedbackParams,
) -> FeedbackResult:
    f_opt = math.exp(
        -params.lambda_lateral * lateral_error**2 - params.lambda_yaw_rate * yaw_rate_error**2
    )
    if f_opt < params.confidence_threshold:
        new_range = min(
            params.max_spatial_range,
            old_spatial_range + params.expansion_coefficient * (params.confidence_threshold - f_opt),
        )
        return FeedbackResult(f_opt, old_spatial_range, new_range, True)
    return FeedbackResult(f_opt, old_spatial_range, old_spatial_range, False)


def interpolate_curvature_at_distance(
    points: Sequence[Point],
    curvatures: Sequence[float],
    lookahead_distance: float,
) -> float:
    if lookahead_distance <= 0:
        raise ValueError("当前控制周期前瞻距离必须大于 0")
    arcs = cumulative_arc_lengths(points)
    if lookahead_distance <= arcs[0]:
        return curvatures[0]
    if lookahead_distance >= arcs[-1]:
        return curvatures[-1]
    for i in range(len(arcs) - 1):
        if arcs[i] <= lookahead_distance <= arcs[i + 1]:
            span = arcs[i + 1] - arcs[i]
            if span <= EPSILON:
                return curvatures[i]
            alpha = (lookahead_distance - arcs[i]) / span
            return curvatures[i] * (1.0 - alpha) + curvatures[i + 1] * alpha
    return curvatures[-1]


def cumulative_arc_lengths(points: Sequence[Point]) -> list[float]:
    arcs = [0.0]
    for i in range(len(points) - 1):
        arcs.append(arcs[-1] + distance(points[i], points[i + 1]))
    return arcs


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def initial_heading(points: Sequence[Point], start_index: int) -> float:
    if start_index < len(points) - 1:
        p0, p1 = points[start_index], points[start_index + 1]
    else:
        p0, p1 = points[start_index - 1], points[start_index]
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def _positive(payload: dict, key: str, label: str) -> float:
    value = float(payload.get(key, 0.0))
    if value <= 0:
        raise ValueError(f"{label} 必须大于 0")
    return value


def _non_negative(payload: dict, key: str, label: str) -> float:
    value = float(payload.get(key, 0.0))
    if value < 0:
        raise ValueError(f"{label} 不能小于 0")
    return value
