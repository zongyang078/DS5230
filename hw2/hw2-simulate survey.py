import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1) Helper functions
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def round_humanlike_hours(x, rng, p_integer=0.82):
    """
    更像真人填写的时间：大多数填整数，少量填 0.5
    p_integer 越高，整数比例越高
    """
    x = np.asarray(x, dtype=float)
    u = rng.random(len(x))
    out = np.empty_like(x)

    # integer
    idx_int = u < p_integer
    out[idx_int] = np.round(x[idx_int])

    # half-hour
    idx_half = ~idx_int
    out[idx_half] = np.round(x[idx_half] * 2) / 2

    return out

def likert_centered_from_latent(z, rng, center_bias=0.70, noise=0.65):
    """
    生成更像真实问卷的 Likert（3最多，2/4次之，1/5很少）
    同时保留与 latent z 的弱相关，保证 correlation/regression 仍可做。
    """
    z = np.asarray(z, dtype=float)
    z = z + rng.normal(0, noise, size=len(z))

    # 标准化，便于阈值切分
    z_std = (z - z.mean()) / (z.std() + 1e-9)

    # 初始 1..5（阈值设置为：2/3/4 多，1/5 少）
    base = np.digitize(z_std, [-1.15, -0.35, 0.35, 1.15]) + 1  # -> 1..5

    # 中庸偏好：把很多 2/4 拉到 3；把大部分 1/5 拉回 2/4
    out = base.copy()
    u = rng.random(len(z))

    # 2/4 -> 3
    mask_24 = (base == 2) | (base == 4)
    out[mask_24 & (u < center_bias)] = 3

    # 1/5 -> 2/4（极端更少出现）
    mask_15 = (base == 1) | (base == 5)
    # 85% 概率不让极端保留
    pull = mask_15 & (u < 0.85)
    out[pull] = np.where(base[pull] == 1, 2, 4)

    return out

def format_timestamp(dt_obj):
    # Google Forms 常见格式：M/D/YYYY HH:MM:SS
    return f"{dt_obj.month}/{dt_obj.day}/{dt_obj.year} {dt_obj.hour:02d}:{dt_obj.minute:02d}:{dt_obj.second:02d}"

def sample_work_hours_timestamps(n, seed=11):
    """
    生成 2026/1/20-2026/1/21 两天内，09:00-18:00 的时间戳（美西正常工作时间）
    """
    rng = np.random.default_rng(seed)

    start_day = datetime(2026, 1, 20)
    end_day = datetime(2026, 1, 21)

    # 两天各自窗口：09:00:00 - 18:00:00
    day_starts = [start_day.replace(hour=9, minute=0, second=0),
                  end_day.replace(hour=9, minute=0, second=0)]
    day_ends   = [start_day.replace(hour=18, minute=0, second=0),
                  end_day.replace(hour=18, minute=0, second=0)]

    # 分配到两天：按 55% / 45%（看起来像第一天多一点也合理）
    day_choice = rng.choice([0, 1], size=n, p=[0.55, 0.45])

    timestamps = []
    for d in day_choice:
        window_seconds = int((day_ends[d] - day_starts[d]).total_seconds())
        offset = rng.integers(0, window_seconds + 1)
        ts = day_starts[d] + timedelta(seconds=int(offset))
        timestamps.append(ts)

    # 排序让它更像真实提交顺序（从早到晚）
    timestamps.sort()
    return [format_timestamp(x) for x in timestamps]

def likert_to_label(v):
    labels = {
        1: "1 – Very low",
        2: "2 – Low",
        3: "3 – Moderate",
        4: "4 – High",
        5: "5 – Very high",
    }
    return labels[int(v)]

# 2) Main generator (human-like, tuned to your real data shape)
def generate_hw2_humanlike(n=60, seed=11):
    rng = np.random.default_rng(seed)

    # ---- Categorical distributions (close to your current sample) ----
    chronotype = rng.choice(
        ["Morning person (sleep early, wake early)",
         "Night owl (sleep late, wake late)",
         "Mixed / No clear preference"],
        size=n,
        p=[0.40, 0.40, 0.20]
    )

    # ---- Sleep hours: mostly 6-8, very few extremes; mostly integers ----
    base_sleep = np.where(chronotype == "Morning person (sleep early, wake early)", 7.3,
                  np.where(chronotype == "Night owl (sleep late, wake late)", 6.6, 7.0))
    sleep = base_sleep + rng.normal(0, 0.65, size=n)
    sleep = clamp(sleep, 5.0, 8.5)  # match your observed min/max range tendency (5-8-ish)
    sleep = round_humanlike_hours(sleep, rng, p_integer=0.88)  # 更偏整数（像你真实数据）
    sleep = clamp(sleep, 5.0, 8.5)

    # ---- Regular sleep: Yes dominant (your data 6/7 Yes) ----
    # Slightly less regular for night owls, slightly more regular when sleep around 7-8
    p_reg = 0.80 + 0.06 * (sleep >= 7.0) - 0.10 * (chronotype == "Night owl (sleep late, wake late)")
    p_reg = clamp(p_reg, 0.55, 0.92)
    regular_sleep = np.where(rng.random(n) < p_reg, "Yes", "No")

    # ---- Study hours: mostly 2-4, some 4.5/5/6; mostly integers, some halves ----
    # Your current sample: mean ~3.4, min 2, max 6
    study = 3.2 + 0.18 * (7.0 - sleep) + rng.normal(0, 0.85, size=n)
    # irregular sleepers have a bit more variance / slightly higher cram
    study += np.where(regular_sleep == "No", rng.normal(0.35, 0.35, size=n), rng.normal(0.05, 0.25, size=n))
    study = clamp(study, 1.5, 6.5)
    study = round_humanlike_hours(study, rng, p_integer=0.78)  # 学习时间更可能出现 0.5
    study = clamp(study, 1.5, 6.5)

    # ---- Latent signals (keep relationships modest; then Likert will be "centered") ----
    # Efficiency: +sleep, +study (weak/moderate)
    eff_latent = 0.55 * sleep + 0.22 * study + rng.normal(0, 0.75, size=n)

    # Stress: -sleep, +study, +irregular
    stress_latent = -0.65 * sleep + 0.28 * study + np.where(regular_sleep == "No", 0.90, 0.0) + rng.normal(0, 0.85, size=n)

    # ---- Likert (3 most common, 2/4 next, 1/5 rare) ----
    efficiency = likert_centered_from_latent(eff_latent, rng, center_bias=0.72, noise=0.62)
    stress = likert_centered_from_latent(stress_latent, rng, center_bias=0.74, noise=0.66)

    # ---- Timestamp within Jan 20-21, 2026, 09:00-18:00 ----
    timestamps = sample_work_hours_timestamps(n, seed=seed + 100)

    # ---- Assemble with EXACT Google Forms-like column names ----
    df_out = pd.DataFrame({
        "Timestamp": timestamps,
        "Q1. Average Sleep Duration\nOn average, how many hours do you sleep per night during the past week?": sleep,
        "Q2. Average Study Time\nOn average, how many hours do you spend studying or doing coursework per day?": study,
        "Q3. Sleep Schedule Regularity\nDo you consider your sleep schedule to be regular?": regular_sleep,
        "Q4. Chronotype (Sleep Preference)\nWhich best describes your sleep preference?": chronotype,
        "Q5. Perceived Academic Efficiency\nOn a scale from 1 to 5, how would you rate your academic efficiency during the past week?":
            [likert_to_label(v) for v in efficiency],
        "Q6. Perceived Stress Level\nOn a scale from 1 to 5, how stressed did you feel during the past week?":
            [likert_to_label(v) for v in stress],
    })

    return df_out

# 3) Generate and save
if __name__ == "__main__":
    df_sim = generate_hw2_humanlike(n=60, seed=11)

    # Quick sanity checks (optional prints)
    print(df_sim.head(8))
    print("\nSleep hours value counts (top):")
    print(df_sim.iloc[:, 1].value_counts().sort_index())
    print("\nStudy hours value counts (top):")
    print(df_sim.iloc[:, 2].value_counts().sort_index())
    print("\nEfficiency Likert counts:")
    print(df_sim.iloc[:, 5].value_counts())
    print("\nStress Likert counts:")
    print(df_sim.iloc[:, 6].value_counts())

    out_path = "hw2_simulated_humanlike_60_with_timestamp.csv"
    df_sim.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")