# 计划书公式与代码追溯

本文件用于说明测试系统每个计算模块与计划书内容的对应关系。代码不实现计划书未给出的深度学习网络结构，而是接收其输出的预选变道轨迹点集 `T_raw`。

| 计划书内容 | 公式/判定 | 代码位置 |
| --- | --- | --- |
| 获取预设轨迹生成网络输出 | `T_raw={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}` | `parse_points_text`, `parse_payload` |
| 三点差分一阶导 | `x_i'≈(x_{i+1}-x_{i-1})/(2Δt)`, `y_i'≈(y_{i+1}-y_{i-1})/(2Δt)` | `curvature_sequence` |
| 三点差分二阶导 | `x_i''≈(x_{i+1}-2x_i+x_{i-1})/Δt^2`, `y_i''≈(y_{i+1}-2y_i+y_{i-1})/Δt^2` | `curvature_sequence` |
| 几何曲率序列 | `κ_i=(x_i'y_i''-y_i'x_i'')/(x_i'^2+y_i'^2)^(3/2)` | `curvature_sequence` |
| 曲率空间变化率 | `κ_i'=(κ_{i+1}-κ_i)/Δs_i` | `curvature_rate_sequence` |
| 曲率极性反转条件 | `κ_i * κ_{i+1} < 0`，取第一轨迹点为反转点 | `detect_reversal_reports` |
| 反转点邻域峰值 | 在预设空间范围内取绝对值最大的曲率空间变化率为 `γ_raw` | `detect_reversal_reports`, `spatial_window_indices` |
| 后轴动态载荷转移 | `ΔF_z=(m*h_g/t_w)*D_r*a_y` | 作为阈值公式来源记录 |
| 曲率空间变化率阈值 | `γ_max=(K_damp*t_w)/(m*h_g*D_r*v^3)` | `curvature_rate_threshold`, `build_threshold_lookup` |
| 局部非线性钳位 | 当 `abs(γ_raw)>γ_max`，将超限曲率空间变化率截断到 `±γ_max` | `reconstruct_window` |
| 安全曲率积分 | `κ_safe(s)=κ_safe(0)+∫_0^s κ_rate(l)dl` | `reconstruct_window` |
| 航向角积分 | `θ(s)=θ_0+∫_0^s κ_safe(l)dl` | `reconstruct_window` |
| 空间坐标累加 | `x(s)=x_0+∫_0^s cos(θ(l))dl`, `y(s)=y_0+∫_0^s sin(θ(l))dl` | `reconstruct_window` |
| 局部点集替换与边界连续 | 将重构后的局部平面点集替换预设空间范围内原始点，并消除替换边界的数据断层 | `reconstruct_window`, `apply_endpoint_continuity_correction` |
| 目标前轮转向角 | `δ=arctan(L*κ)` | `generate_control_command` |
| PWM 占空比 | `Duty=K_pwm*δ+Duty_center` | `generate_control_command` |
| 反馈优化参数 | `F_opt=exp(-λ1*e_y^2-λ2*e_ω^2)` | `evaluate_feedback` |
| 预设空间范围更新 | `L_new=min(L_max,L_old+ρ*(F_th-F_opt))`，当 `F_opt < F_th` 时更新 | `evaluate_feedback` |

## 实现边界

- 计划书未给出深度学习网络结构和权重，因此测试系统只验证其输出点集之后的轨迹提取、阈值、重构、控制与反馈链路。
- 计划书未给出“根据当前纵向行驶速度动态计算前瞻距离”的解析公式，因此测试系统将当前控制周期前瞻距离作为输入参数，用于执行计划书中明确给出的曲率插值、转向映射与 PWM 公式。
- 曲率序列端点没有三点邻域。为保持 `K_raw={κ_1,...,κ_N}` 的序列长度，端点复制最近的可三点差分内点曲率，不引入新的曲率公式。
