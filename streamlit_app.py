import random
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


st.set_page_config(
    page_title="가상 반도체 박막 공정 시뮬레이터",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* 넓은 모니터에서도 콘텐츠가 지나치게 벌어지지 않도록 중앙 정렬 */
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 1480px;
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1.5rem;
    }

    /* 표와 차트 사이 여백을 조금 줄임 */
    div[data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }

    /* 2열 대시보드의 컬럼 간격 */
    div[data-testid="stHorizontalBlock"] {
        gap: 1.25rem;
    }

    /* 중간 크기 화면 */
    @media (max-width: 1500px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: 1280px;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
    }

    /* 노트북·태블릿 */
    @media (max-width: 1100px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WAFERS_PER_LOT = 25
TARGET_THICKNESS = 100.0
LOWER_SPEC = 98.0
UPPER_SPEC = 102.0
UNIFORMITY_LIMIT = 3.0

APP_VERSION = "v2.1.0"
LAST_UPDATED = "2026-08-05"

# 아래 수치는 실제 생산 Recipe가 아니라 교육용 비교 모델입니다.
# 녹는점과 밀도는 참고 정보로만 표시하고 계산에는 사용하지 않습니다.
MATERIALS = {
    "Al": {
        "name_ko": "알루미늄",
        "base_rate_nm_s": 1.00,
        "power_w": 500.0,
        "ar_flow_sccm": 30.0,
        "pressure_mtorr": 5.0,
        "bulk_resistivity_uohm_cm": 2.65,
        "density_g_cm3": 2.70,
        "melting_point_c": 660.3,
        "wear_sensitivity": 0.008,
        "description": "배선·전극용 금속 박막의 기초 Case",
    },
    "Cu": {
        "name_ko": "구리",
        "base_rate_nm_s": 0.90,
        "power_w": 500.0,
        "ar_flow_sccm": 30.0,
        "pressure_mtorr": 5.0,
        "bulk_resistivity_uohm_cm": 1.68,
        "density_g_cm3": 8.96,
        "melting_point_c": 1084.6,
        "wear_sensitivity": 0.009,
        "description": "낮은 비저항을 갖는 금속 배선 비교 Case",
    },
    "Ti": {
        "name_ko": "타이타늄",
        "base_rate_nm_s": 0.70,
        "power_w": 600.0,
        "ar_flow_sccm": 30.0,
        "pressure_mtorr": 5.0,
        "bulk_resistivity_uohm_cm": 42.0,
        "density_g_cm3": 4.51,
        "melting_point_c": 1668.0,
        "wear_sensitivity": 0.011,
        "description": "접착층·배리어층 재료 비교 Case",
    },
    "Ta": {
        "name_ko": "탄탈럼",
        "base_rate_nm_s": 0.55,
        "power_w": 650.0,
        "ar_flow_sccm": 30.0,
        "pressure_mtorr": 5.0,
        "bulk_resistivity_uohm_cm": 13.5,
        "density_g_cm3": 16.69,
        "melting_point_c": 3017.0,
        "wear_sensitivity": 0.013,
        "description": "배리어층 재료 비교 Case",
    },
}

# 박막에서는 표면·입계 산란 등에 의해 벌크보다 비저항이 커질 수 있으므로
# 교육용 단순화 계수를 적용합니다. 실제 측정값을 의미하지 않습니다.
THIN_FILM_RESISTIVITY_FACTOR = 1.25

CAUSES = {
    "power_drift": "전원 공급계통 출력 드리프트",
    "flow_drop": "Ar 유량 제어 이상",
    "pressure_rise": "챔버 압력 제어 이상",
    "target_wear": "타겟 국부 소모",
}

CAUSE_EXPLANATIONS = {
    "power_drift": (
        "Actual Power가 Lot 진행에 따라 변하고, "
        "증착률·두께·면저항이 함께 변합니다."
    ),
    "flow_drop": (
        "Actual Ar Flow와 Pressure가 함께 낮아지고, "
        "증착률과 두께가 감소하면서 면저항이 높아집니다."
    ),
    "pressure_rise": (
        "Actual Pressure가 상승하면서 증착률과 두께가 감소하고 "
        "균일도와 면저항이 악화됩니다."
    ),
    "target_wear": (
        "표시되는 주요 제어값은 Setpoint 부근이지만 "
        "증착률이 서서히 낮아지고 균일도와 면저항이 악화됩니다."
    ),
}


DIFFICULTIES = {
    "easy": {
        "label": "쉬움",
        "description": "원인 1개, 강한 신호, 낮은 노이즈",
        "severity_range": (0.95, 1.35),
        "noise_range": (0.70, 0.95),
        "nuisance_probability": 0.03,
        "partial_probability": 0.00,
        "composite_probability": 0.00,
        "sensor_bias_probability": 0.02,
        "transient_probability": 0.03,
        "recipe_shift_probability": 0.03,
        "compensation_range": (0.00, 0.10),
        "overlap_strength": 0.04,
        "intermittent_probability": 0.00,
        "ai_threshold": 0.55,
    },
    "normal": {
        "label": "보통",
        "description": "신호 중첩, 보통 노이즈, 일시적 변동",
        "severity_range": (0.65, 1.15),
        "noise_range": (0.90, 1.25),
        "nuisance_probability": 0.12,
        "partial_probability": 0.05,
        "composite_probability": 0.00,
        "sensor_bias_probability": 0.08,
        "transient_probability": 0.10,
        "recipe_shift_probability": 0.08,
        "compensation_range": (0.00, 0.25),
        "overlap_strength": 0.12,
        "intermittent_probability": 0.05,
        "ai_threshold": 0.58,
    },
    "hard": {
        "label": "어려움",
        "description": "약한 이상, 센서 편향, 일부 Wafer 이상",
        "severity_range": (0.40, 0.95),
        "noise_range": (1.10, 1.65),
        "nuisance_probability": 0.28,
        "partial_probability": 0.22,
        "composite_probability": 0.15,
        "sensor_bias_probability": 0.22,
        "transient_probability": 0.22,
        "recipe_shift_probability": 0.20,
        "compensation_range": (0.15, 0.55),
        "overlap_strength": 0.28,
        "intermittent_probability": 0.20,
        "ai_threshold": 0.62,
    },
    "expert": {
        "label": "전문가",
        "description": "복합 이상, 간헐 현상, 보상 제어, 높은 노이즈",
        "severity_range": (0.25, 0.80),
        "noise_range": (1.35, 2.10),
        "nuisance_probability": 0.45,
        "partial_probability": 0.42,
        "composite_probability": 0.48,
        "sensor_bias_probability": 0.38,
        "transient_probability": 0.35,
        "recipe_shift_probability": 0.32,
        "compensation_range": (0.30, 0.78),
        "overlap_strength": 0.45,
        "intermittent_probability": 0.40,
        "ai_threshold": 0.65,
    },
}

DIFFICULTY_ORDER = [
    "easy",
    "normal",
    "hard",
    "expert",
]

DIFFICULTY_LABELS = {
    key: value["label"]
    for key, value in DIFFICULTIES.items()
}

AI_LABELS = {
    "normal": "정상",
    **CAUSES,
}

AI_CLASS_ORDER = [
    "normal",
    "power_drift",
    "flow_drop",
    "pressure_rise",
    "target_wear",
]

AI_FEATURE_COLUMNS = [
    "material_Al",
    "material_Cu",
    "material_Ti",
    "material_Ta",
    "difficulty_easy",
    "difficulty_normal",
    "difficulty_hard",
    "difficulty_expert",
    "lot_number",
    "mean_thickness_nm",
    "std_thickness_nm",
    "mean_uniformity_pct",
    "mean_sheet_resistance_ohm_sq",
    "oos_ratio",
    "power_dev_set_pct",
    "flow_dev_set_pct",
    "pressure_dev_set_pct",
    "rate_dev_set_pct",
    "thickness_delta_baseline",
    "std_delta_baseline",
    "uniformity_delta_baseline",
    "resistance_delta_baseline_pct",
    "power_delta_baseline_pct",
    "flow_delta_baseline_pct",
    "pressure_delta_baseline_pct",
    "rate_delta_baseline_pct",
    "thickness_delta_previous",
    "uniformity_delta_previous",
    "resistance_delta_previous_pct",
]

AI_FEATURE_LABELS = {
    "material_Al": "재료: Al",
    "material_Cu": "재료: Cu",
    "material_Ti": "재료: Ti",
    "material_Ta": "재료: Ta",
    "difficulty_easy": "난이도: 쉬움",
    "difficulty_normal": "난이도: 보통",
    "difficulty_hard": "난이도: 어려움",
    "difficulty_expert": "난이도: 전문가",
    "lot_number": "Lot 번호",
    "mean_thickness_nm": "평균 두께",
    "std_thickness_nm": "두께 표준편차",
    "mean_uniformity_pct": "평균 균일도",
    "mean_sheet_resistance_ohm_sq": "예상 면저항",
    "oos_ratio": "두께 이탈 비율",
    "power_dev_set_pct": "Power Setpoint 편차",
    "flow_dev_set_pct": "Ar Flow Setpoint 편차",
    "pressure_dev_set_pct": "Pressure Setpoint 편차",
    "rate_dev_set_pct": "증착률 기준 편차",
    "thickness_delta_baseline": "정상 기준 대비 두께 변화",
    "std_delta_baseline": "정상 기준 대비 산포 변화",
    "uniformity_delta_baseline": "정상 기준 대비 균일도 변화",
    "resistance_delta_baseline_pct": "정상 기준 대비 면저항 변화",
    "power_delta_baseline_pct": "정상 기준 대비 Power 변화",
    "flow_delta_baseline_pct": "정상 기준 대비 Ar Flow 변화",
    "pressure_delta_baseline_pct": "정상 기준 대비 Pressure 변화",
    "rate_delta_baseline_pct": "정상 기준 대비 증착률 변화",
    "thickness_delta_previous": "직전 Lot 대비 두께 변화",
    "uniformity_delta_previous": "직전 Lot 대비 균일도 변화",
    "resistance_delta_previous_pct": "직전 Lot 대비 면저항 변화",
}

RAW_COLUMNS = [
    "material",
    "lot",
    "wafer",
    "power_set_w",
    "power_actual_w",
    "ar_flow_set_sccm",
    "ar_flow_actual_sccm",
    "pressure_set_mtorr",
    "pressure_actual_mtorr",
    "time_set_s",
    "time_actual_s",
    "target_usage_h",
    "deposition_rate_nm_s",
    "thickness_nm",
    "uniformity_pct",
    "estimated_sheet_resistance_ohm_sq",
]

SUMMARY_COLUMNS = [
    "material",
    "lot",
    "mean_thickness_nm",
    "std_thickness_nm",
    "mean_uniformity_pct",
    "mean_sheet_resistance_ohm_sq",
    "oos_count",
    "mean_power_w",
    "mean_ar_flow_sccm",
    "mean_pressure_mtorr",
    "mean_rate_nm_s",
]


def empty_raw():
    return pd.DataFrame(columns=RAW_COLUMNS)


def empty_summary():
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def get_recipe(material):
    props = MATERIALS[material]
    deposition_time = TARGET_THICKNESS / props["base_rate_nm_s"]

    return {
        "power_w": props["power_w"],
        "ar_flow_sccm": props["ar_flow_sccm"],
        "pressure_mtorr": props["pressure_mtorr"],
        "deposition_time_s": deposition_time,
        "base_rate_nm_s": props["base_rate_nm_s"],
    }


def reference_sheet_resistance(material):
    resistivity = MATERIALS[material]["bulk_resistivity_uohm_cm"]
    return (
        10
        * resistivity
        * THIN_FILM_RESISTIVITY_FACTOR
        / TARGET_THICKNESS
    )



def build_case_profile(
    rng,
    difficulty,
    fault_start_lot,
    event_min_lot,
    event_max_lot,
    has_fault=True,
):
    config = DIFFICULTIES[difficulty]

    primary_cause = (
        str(rng.choice(list(CAUSES.keys())))
        if has_fault
        else "normal"
    )

    secondary_cause = None
    if (
        has_fault
        and rng.random()
        < config["composite_probability"]
    ):
        secondary_candidates = [
            cause
            for cause in CAUSES
            if cause != primary_cause
        ]
        secondary_cause = str(
            rng.choice(secondary_candidates)
        )

    severity = float(
        rng.uniform(
            *config["severity_range"]
        )
    )
    secondary_severity = (
        float(
            rng.uniform(
                *config["severity_range"]
            )
        )
        if secondary_cause
        else 0.0
    )

    profile = {
        "difficulty": difficulty,
        "primary_cause": primary_cause,
        "secondary_cause": secondary_cause,
        "fault_start_lot": (
            int(fault_start_lot)
            if has_fault
            else None
        ),
        "secondary_start_lot": (
            int(
                fault_start_lot
                + rng.integers(0, 3)
            )
            if secondary_cause
            else None
        ),
        "fault_sign": int(
            rng.choice([-1, 1])
        ),
        "severity": severity,
        "secondary_severity": (
            secondary_severity
        ),
        "noise_scale": float(
            rng.uniform(
                *config["noise_range"]
            )
        ),
        "compensation": float(
            rng.uniform(
                *config[
                    "compensation_range"
                ]
            )
        ),
        "overlap_strength": float(
            config["overlap_strength"]
        ),
        "partial_fraction": 1.0,
        "intermittent": bool(
            rng.random()
            < config[
                "intermittent_probability"
            ]
        ),
        "intermittent_period": int(
            rng.integers(2, 4)
        ),
        "intermittent_phase": int(
            rng.integers(0, 3)
        ),
        "sensor_variable": None,
        "sensor_bias": 0.0,
        "sensor_start_lot": None,
        "transient_lot": None,
        "transient_type": None,
        "transient_magnitude": 0.0,
        "recipe_shift_lot": None,
        "recipe_power_shift_pct": 0.0,
        "measurement_spike_lot": None,
        "measurement_spike_fraction": 0.0,
        "measurement_spike_nm": 0.0,
    }

    if (
        rng.random()
        < config["partial_probability"]
    ):
        profile["partial_fraction"] = float(
            rng.uniform(0.20, 0.70)
        )

    if (
        rng.random()
        < config[
            "sensor_bias_probability"
        ]
    ):
        sensor_variable = str(
            rng.choice(
                [
                    "power",
                    "flow",
                    "pressure",
                ]
            )
        )
        sensor_scales = {
            "power": 0.006,
            "flow": 0.010,
            "pressure": 0.018,
        }
        profile["sensor_variable"] = (
            sensor_variable
        )
        profile["sensor_start_lot"] = int(
            rng.integers(
                event_min_lot,
                event_max_lot + 1,
            )
        )
        profile["sensor_bias"] = float(
            rng.choice([-1, 1])
            * sensor_scales[
                sensor_variable
            ]
            * rng.uniform(0.50, 1.50)
        )

    if (
        rng.random()
        < config[
            "transient_probability"
        ]
    ):
        profile["transient_lot"] = int(
            rng.integers(
                event_min_lot,
                event_max_lot + 1,
            )
        )
        profile["transient_type"] = str(
            rng.choice(
                [
                    "power",
                    "flow",
                    "pressure",
                    "time",
                ]
            )
        )
        profile[
            "transient_magnitude"
        ] = float(
            rng.choice([-1, 1])
            * rng.uniform(0.003, 0.012)
        )

    if (
        rng.random()
        < config[
            "recipe_shift_probability"
        ]
    ):
        profile["recipe_shift_lot"] = int(
            rng.integers(
                event_min_lot,
                event_max_lot + 1,
            )
        )
        profile[
            "recipe_power_shift_pct"
        ] = float(
            rng.choice([-1, 1])
            * rng.uniform(0.005, 0.020)
        )

    if (
        rng.random()
        < config[
            "nuisance_probability"
        ]
    ):
        profile[
            "measurement_spike_lot"
        ] = int(
            rng.integers(
                event_min_lot,
                event_max_lot + 1,
            )
        )
        profile[
            "measurement_spike_fraction"
        ] = float(
            rng.uniform(0.08, 0.35)
        )
        profile[
            "measurement_spike_nm"
        ] = float(
            rng.choice([-1, 1])
            * rng.uniform(0.40, 1.50)
        )

    return profile


def apply_cause_effect(
    cause,
    severity,
    progress,
    sign,
    overlap_strength,
    compensation,
    mask,
    base_rate,
    power_setpoint,
    flow_setpoint,
    pressure_setpoint,
    true_power,
    true_flow,
    true_pressure,
    hidden_rate_effect,
    hidden_uniformity_effect,
    hidden_resistivity_effect,
):
    effective_progress = (
        max(progress, 0)
        * severity
    )
    observable_fraction = (
        1.0
        - compensation
    )

    if cause == "power_drift":
        true_power[mask] += (
            sign
            * power_setpoint
            * 0.010
            * effective_progress
            * observable_fraction
        )
        hidden_rate_effect[mask] += (
            sign
            * base_rate
            * 0.004
            * effective_progress
            * (
                0.40
                + 0.60
                * compensation
            )
        )
        true_pressure[mask] += (
            sign
            * pressure_setpoint
            * 0.006
            * effective_progress
            * overlap_strength
        )
        hidden_uniformity_effect[
            mask
        ] += (
            0.08
            * effective_progress
            * overlap_strength
        )

    elif cause == "flow_drop":
        true_flow[mask] -= (
            0.85
            * effective_progress
            * observable_fraction
        )
        true_pressure[mask] -= (
            0.06
            * effective_progress
            * observable_fraction
        )
        hidden_rate_effect[mask] -= (
            base_rate
            * 0.004
            * effective_progress
            * (
                0.40
                + 0.60
                * compensation
            )
        )
        true_power[mask] += (
            power_setpoint
            * 0.0025
            * effective_progress
            * overlap_strength
        )
        hidden_uniformity_effect[
            mask
        ] += (
            0.10
            * effective_progress
            * overlap_strength
        )

    elif cause == "pressure_rise":
        true_pressure[mask] += (
            0.25
            * effective_progress
            * observable_fraction
        )
        hidden_rate_effect[mask] -= (
            base_rate
            * 0.005
            * effective_progress
        )
        hidden_uniformity_effect[
            mask
        ] += (
            0.34
            * effective_progress
        )
        hidden_resistivity_effect[
            mask
        ] += (
            0.015
            * effective_progress
        )
        true_flow[mask] -= (
            0.15
            * effective_progress
            * overlap_strength
        )

    elif cause == "target_wear":
        hidden_rate_effect[mask] -= (
            base_rate
            * 0.0075
            * effective_progress
        )
        hidden_uniformity_effect[
            mask
        ] += (
            0.24
            * effective_progress
        )
        hidden_resistivity_effect[
            mask
        ] += (
            0.012
            * effective_progress
        )
        true_power[mask] += (
            power_setpoint
            * 0.004
            * effective_progress
            * overlap_strength
        )
        true_pressure[mask] += (
            pressure_setpoint
            * 0.004
            * effective_progress
            * overlap_strength
        )


def generate_lot_measurements(
    rng,
    material,
    lot,
    profile,
):
    material_props = MATERIALS[material]
    recipe = get_recipe(material)
    wafer_count = WAFERS_PER_LOT

    power_set = np.full(
        wafer_count,
        recipe["power_w"],
        dtype=float,
    )
    flow_set = np.full(
        wafer_count,
        recipe["ar_flow_sccm"],
        dtype=float,
    )
    pressure_set = np.full(
        wafer_count,
        recipe["pressure_mtorr"],
        dtype=float,
    )
    time_set = np.full(
        wafer_count,
        recipe["deposition_time_s"],
        dtype=float,
    )

    recipe_shift_lot = profile.get(
        "recipe_shift_lot"
    )
    if (
        recipe_shift_lot is not None
        and lot >= recipe_shift_lot
    ):
        power_shift = profile.get(
            "recipe_power_shift_pct",
            0.0,
        )
        power_set *= (
            1.0
            + power_shift
        )
        time_set *= (
            1.0
            - 0.45
            * power_shift
        )

    noise_scale = profile.get(
        "noise_scale",
        1.0,
    )

    true_power = rng.normal(
        power_set,
        power_set
        * 0.0028
        * noise_scale,
    )
    true_flow = rng.normal(
        flow_set,
        0.08
        * noise_scale,
    )
    true_pressure = rng.normal(
        pressure_set,
        0.03
        * noise_scale,
    )
    true_time = rng.normal(
        time_set,
        time_set
        * 0.0006
        * noise_scale,
    )

    hidden_rate_effect = np.zeros(
        wafer_count,
        dtype=float,
    )
    hidden_uniformity_effect = np.zeros(
        wafer_count,
        dtype=float,
    )
    hidden_resistivity_effect = np.zeros(
        wafer_count,
        dtype=float,
    )

    primary_cause = profile.get(
        "primary_cause",
        "normal",
    )
    fault_start_lot = profile.get(
        "fault_start_lot"
    )

    primary_active = (
        primary_cause != "normal"
        and fault_start_lot is not None
        and lot >= fault_start_lot
    )

    if (
        primary_active
        and profile.get(
            "intermittent",
            False,
        )
    ):
        primary_active = (
            (
                lot
                - fault_start_lot
                + profile.get(
                    "intermittent_phase",
                    0,
                )
            )
            % profile.get(
                "intermittent_period",
                2,
            )
            != 1
        )

    primary_mask = np.ones(
        wafer_count,
        dtype=bool,
    )

    partial_fraction = profile.get(
        "partial_fraction",
        1.0,
    )

    if (
        primary_active
        and partial_fraction < 1.0
    ):
        primary_mask[:] = False
        affected_count = max(
            1,
            int(
                round(
                    wafer_count
                    * partial_fraction
                )
            ),
        )
        affected_indices = rng.choice(
            wafer_count,
            size=affected_count,
            replace=False,
        )
        primary_mask[
            affected_indices
        ] = True

    if primary_active:
        apply_cause_effect(
            cause=primary_cause,
            severity=profile.get(
                "severity",
                1.0,
            ),
            progress=(
                lot
                - fault_start_lot
                + 1
            ),
            sign=profile.get(
                "fault_sign",
                1,
            ),
            overlap_strength=profile.get(
                "overlap_strength",
                0.0,
            ),
            compensation=profile.get(
                "compensation",
                0.0,
            ),
            mask=primary_mask,
            base_rate=recipe[
                "base_rate_nm_s"
            ],
            power_setpoint=float(
                power_set[0]
            ),
            flow_setpoint=float(
                flow_set[0]
            ),
            pressure_setpoint=float(
                pressure_set[0]
            ),
            true_power=true_power,
            true_flow=true_flow,
            true_pressure=true_pressure,
            hidden_rate_effect=(
                hidden_rate_effect
            ),
            hidden_uniformity_effect=(
                hidden_uniformity_effect
            ),
            hidden_resistivity_effect=(
                hidden_resistivity_effect
            ),
        )

    secondary_cause = profile.get(
        "secondary_cause"
    )
    secondary_start_lot = profile.get(
        "secondary_start_lot"
    )

    secondary_active = (
        secondary_cause is not None
        and secondary_start_lot is not None
        and lot >= secondary_start_lot
    )

    if secondary_active:
        secondary_mask = np.ones(
            wafer_count,
            dtype=bool,
        )

        if partial_fraction < 1.0:
            secondary_mask[:] = False
            secondary_fraction = min(
                1.0,
                partial_fraction
                + 0.10,
            )
            affected_count = max(
                1,
                int(
                    round(
                        wafer_count
                        * secondary_fraction
                    )
                ),
            )
            affected_indices = rng.choice(
                wafer_count,
                size=affected_count,
                replace=False,
            )
            secondary_mask[
                affected_indices
            ] = True

        apply_cause_effect(
            cause=secondary_cause,
            severity=profile.get(
                "secondary_severity",
                0.0,
            ),
            progress=(
                lot
                - secondary_start_lot
                + 1
            ),
            sign=(
                -1
                * profile.get(
                    "fault_sign",
                    1,
                )
            ),
            overlap_strength=profile.get(
                "overlap_strength",
                0.0,
            ),
            compensation=profile.get(
                "compensation",
                0.0,
            ),
            mask=secondary_mask,
            base_rate=recipe[
                "base_rate_nm_s"
            ],
            power_setpoint=float(
                power_set[0]
            ),
            flow_setpoint=float(
                flow_set[0]
            ),
            pressure_setpoint=float(
                pressure_set[0]
            ),
            true_power=true_power,
            true_flow=true_flow,
            true_pressure=true_pressure,
            hidden_rate_effect=(
                hidden_rate_effect
            ),
            hidden_uniformity_effect=(
                hidden_uniformity_effect
            ),
            hidden_resistivity_effect=(
                hidden_resistivity_effect
            ),
        )

    transient_lot = profile.get(
        "transient_lot"
    )

    if (
        transient_lot is not None
        and lot == transient_lot
    ):
        transient_type = profile.get(
            "transient_type"
        )
        transient_magnitude = profile.get(
            "transient_magnitude",
            0.0,
        )

        if transient_type == "power":
            true_power *= (
                1.0
                + transient_magnitude
            )
        elif transient_type == "flow":
            true_flow *= (
                1.0
                + transient_magnitude
            )
        elif transient_type == "pressure":
            true_pressure *= (
                1.0
                + transient_magnitude
            )
        elif transient_type == "time":
            true_time *= (
                1.0
                + transient_magnitude
            )

    wafer_numbers = np.arange(
        1,
        wafer_count + 1,
    )
    processed_wafer_count = (
        (lot - 1)
        * wafer_count
        + wafer_numbers
    )
    target_usage = (
        80
        + processed_wafer_count
        * 0.02
    )

    relative_rate = (
        1.0
        + 0.0014
        * (
            true_power
            - power_set
        )
        + 0.0080
        * (
            true_flow
            - flow_set
        )
        - 0.0200
        * (
            true_pressure
            - pressure_set
        )
    )

    deposition_rate = (
        recipe["base_rate_nm_s"]
        * relative_rate
        - recipe["base_rate_nm_s"]
        * 0.0004
        * (
            target_usage
            - 80
        )
        + hidden_rate_effect
        + rng.normal(
            0,
            recipe["base_rate_nm_s"]
            * 0.0025
            * noise_scale,
            wafer_count,
        )
    )

    deposition_rate = np.maximum(
        deposition_rate,
        recipe["base_rate_nm_s"]
        * 0.25,
    )

    thickness = (
        deposition_rate
        * true_time
        + rng.normal(
            0,
            0.16
            * noise_scale,
            wafer_count,
        )
    )

    uniformity = (
        1.15
        + 0.55
        * np.abs(
            true_pressure
            - pressure_set
        )
        + hidden_uniformity_effect
        + rng.normal(
            0,
            0.10
            * noise_scale,
            wafer_count,
        )
    )

    uniformity = np.maximum(
        uniformity,
        0.10,
    )

    film_factor = (
        THIN_FILM_RESISTIVITY_FACTOR
        * (
            1.0
            + hidden_resistivity_effect
        )
        * (
            1.0
            + 0.015
            * np.maximum(
                uniformity
                - 1.15,
                0,
            )
        )
        * rng.normal(
            1.0,
            0.008
            * noise_scale,
            wafer_count,
        )
    )

    sheet_resistance = (
        10
        * material_props[
            "bulk_resistivity_uohm_cm"
        ]
        * film_factor
        / np.maximum(
            thickness,
            1.0,
        )
    )

    measurement_spike_lot = profile.get(
        "measurement_spike_lot"
    )

    if (
        measurement_spike_lot
        is not None
        and lot
        == measurement_spike_lot
    ):
        spike_count = max(
            1,
            int(
                round(
                    wafer_count
                    * profile.get(
                        "measurement_spike_fraction",
                        0.0,
                    )
                )
            ),
        )
        spike_indices = rng.choice(
            wafer_count,
            size=spike_count,
            replace=False,
        )
        thickness[
            spike_indices
        ] += (
            profile.get(
                "measurement_spike_nm",
                0.0,
            )
            * rng.uniform(
                0.70,
                1.30,
                spike_count,
            )
        )
        sheet_resistance[
            spike_indices
        ] = (
            10
            * material_props[
                "bulk_resistivity_uohm_cm"
            ]
            * film_factor[
                spike_indices
            ]
            / np.maximum(
                thickness[
                    spike_indices
                ],
                1.0,
            )
        )

    reported_power = true_power.copy()
    reported_flow = true_flow.copy()
    reported_pressure = (
        true_pressure.copy()
    )

    sensor_variable = profile.get(
        "sensor_variable"
    )
    sensor_start_lot = profile.get(
        "sensor_start_lot"
    )

    if (
        sensor_variable is not None
        and sensor_start_lot is not None
        and lot >= sensor_start_lot
    ):
        sensor_bias = profile.get(
            "sensor_bias",
            0.0,
        )

        if sensor_variable == "power":
            reported_power *= (
                1.0
                + sensor_bias
            )
        elif sensor_variable == "flow":
            reported_flow *= (
                1.0
                + sensor_bias
            )
        elif sensor_variable == "pressure":
            reported_pressure *= (
                1.0
                + sensor_bias
            )

    return {
        "power_set": power_set,
        "flow_set": flow_set,
        "pressure_set": pressure_set,
        "time_set": time_set,
        "power_actual": reported_power,
        "flow_actual": reported_flow,
        "pressure_actual": (
            reported_pressure
        ),
        "time_actual": true_time,
        "target_usage": target_usage,
        "deposition_rate": deposition_rate,
        "thickness": thickness,
        "uniformity": uniformity,
        "sheet_resistance": (
            sheet_resistance
        ),
        "primary_active": bool(
            primary_active
        ),
        "secondary_active": bool(
            secondary_active
        ),
    }


def lot_measurements_to_summary(
    case_id,
    material,
    difficulty,
    lot,
    measurements,
    target,
    primary_cause,
    secondary_cause,
    fault_start_lot,
    severity,
):
    thickness = measurements[
        "thickness"
    ]
    uniformity = measurements[
        "uniformity"
    ]
    sheet_resistance = measurements[
        "sheet_resistance"
    ]

    return {
        "case_id": case_id,
        "material": material,
        "difficulty": difficulty,
        "lot": lot,
        "mean_thickness_nm": float(
            np.mean(thickness)
        ),
        "std_thickness_nm": float(
            np.std(
                thickness,
                ddof=1,
            )
        ),
        "mean_uniformity_pct": float(
            np.mean(uniformity)
        ),
        "mean_sheet_resistance_ohm_sq": float(
            np.mean(
                sheet_resistance
            )
        ),
        "oos_count": int(
            np.sum(
                (thickness < LOWER_SPEC)
                | (thickness > UPPER_SPEC)
            )
        ),
        "mean_power_w": float(
            np.mean(
                measurements[
                    "power_actual"
                ]
            )
        ),
        "mean_ar_flow_sccm": float(
            np.mean(
                measurements[
                    "flow_actual"
                ]
            )
        ),
        "mean_pressure_mtorr": float(
            np.mean(
                measurements[
                    "pressure_actual"
                ]
            )
        ),
        "mean_rate_nm_s": float(
            np.mean(
                measurements[
                    "deposition_rate"
                ]
            )
        ),
        "target": target,
        "actual_case_cause": (
            primary_cause
        ),
        "secondary_cause": (
            secondary_cause
            or ""
        ),
        "actual_fault_start_lot": (
            fault_start_lot
            if fault_start_lot
            is not None
            else 0
        ),
        "severity": severity,
    }


def initialize_case(
    material="Al",
    difficulty="normal",
    seed=None,
):
    if seed is None:
        seed = random.SystemRandom().randint(
            1,
            999_999_999,
        )

    st.session_state.active_material = material
    st.session_state.active_difficulty = (
        difficulty
    )
    st.session_state.case_seed = int(seed)
    st.session_state.rng = (
        np.random.default_rng(
            int(seed)
        )
    )
    st.session_state.current_lot = 0
    st.session_state.cause_key = None
    st.session_state.secondary_cause_key = (
        None
    )
    st.session_state.fault_start_lot = None
    st.session_state.case_profile = None
    st.session_state.raw_data = empty_raw()
    st.session_state.last_result = None
    st.session_state.case_ready = True



def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "case_ready" not in st.session_state:
        st.session_state.case_ready = False

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "ai_bundle" not in st.session_state:
        st.session_state.ai_bundle = None

    if "active_difficulty" not in st.session_state:
        st.session_state.active_difficulty = "normal"

    if "case_profile" not in st.session_state:
        st.session_state.case_profile = None

    if "secondary_cause_key" not in st.session_state:
        st.session_state.secondary_cause_key = None

    if not st.session_state.case_ready:
        initialize_case()


def summarize(df):
    if df.empty:
        return empty_summary()

    summary = (
        df.groupby(["material", "lot"])
        .agg(
            mean_thickness_nm=("thickness_nm", "mean"),
            std_thickness_nm=("thickness_nm", "std"),
            mean_uniformity_pct=("uniformity_pct", "mean"),
            mean_sheet_resistance_ohm_sq=(
                "estimated_sheet_resistance_ohm_sq",
                "mean",
            ),
            mean_power_w=("power_actual_w", "mean"),
            mean_ar_flow_sccm=("ar_flow_actual_sccm", "mean"),
            mean_pressure_mtorr=("pressure_actual_mtorr", "mean"),
            mean_rate_nm_s=("deposition_rate_nm_s", "mean"),
        )
        .reset_index()
    )

    oos = (
        df.assign(
            failed=(
                (df["thickness_nm"] < LOWER_SPEC)
                | (df["thickness_nm"] > UPPER_SPEC)
            )
        )
        .groupby(["material", "lot"])["failed"]
        .sum()
        .reset_index(name="oos_count")
    )

    summary = summary.merge(
        oos,
        on=["material", "lot"],
    )

    numeric_columns = [
        column
        for column in SUMMARY_COLUMNS
        if column not in ("material", "lot", "oos_count")
    ]
    summary[numeric_columns] = summary[numeric_columns].round(4)

    return summary[SUMMARY_COLUMNS]


def inject_fault():
    if (
        st.session_state.current_lot
        < 2
    ):
        return (
            False,
            "정상 기준을 만들기 위해 "
            "Lot을 2개 먼저 생산하세요.",
        )

    if (
        st.session_state.cause_key
        is not None
    ):
        return (
            False,
            "이번 Case에는 이미 "
            "이상이 예약되어 있습니다.",
        )

    rng = st.session_state.rng
    difficulty = (
        st.session_state.active_difficulty
    )

    fault_start_lot = (
        st.session_state.current_lot
        + int(
            rng.integers(1, 3)
        )
    )

    event_min_lot = (
        st.session_state.current_lot
        + 1
    )
    event_max_lot = (
        st.session_state.current_lot
        + 10
    )

    profile = build_case_profile(
        rng=rng,
        difficulty=difficulty,
        fault_start_lot=(
            fault_start_lot
        ),
        event_min_lot=(
            event_min_lot
        ),
        event_max_lot=(
            event_max_lot
        ),
        has_fault=True,
    )

    st.session_state.case_profile = (
        profile
    )
    st.session_state.cause_key = (
        profile["primary_cause"]
    )
    st.session_state.secondary_cause_key = (
        profile["secondary_cause"]
    )
    st.session_state.fault_start_lot = (
        profile["fault_start_lot"]
    )

    return (
        True,
        "랜덤 이상이 예약되었습니다. "
        "난이도에 따라 노이즈·일시 변동·"
        "센서 편향·복합 이상이 포함될 수 있습니다.",
    )


def produce_lot():
    material = (
        st.session_state.active_material
    )
    difficulty = (
        st.session_state.active_difficulty
    )
    rng = st.session_state.rng

    st.session_state.current_lot += 1
    lot = st.session_state.current_lot

    profile = (
        st.session_state.case_profile
    )

    if profile is None:
        config = DIFFICULTIES[
            difficulty
        ]
        profile = {
            "difficulty": difficulty,
            "primary_cause": "normal",
            "secondary_cause": None,
            "fault_start_lot": None,
            "secondary_start_lot": None,
            "fault_sign": 1,
            "severity": 0.0,
            "secondary_severity": 0.0,
            "noise_scale": float(
                np.mean(
                    config[
                        "noise_range"
                    ]
                )
            ),
            "compensation": 0.0,
            "overlap_strength": 0.0,
            "partial_fraction": 1.0,
            "intermittent": False,
            "intermittent_period": 2,
            "intermittent_phase": 0,
            "sensor_variable": None,
            "sensor_bias": 0.0,
            "sensor_start_lot": None,
            "transient_lot": None,
            "transient_type": None,
            "transient_magnitude": 0.0,
            "recipe_shift_lot": None,
            "recipe_power_shift_pct": 0.0,
            "measurement_spike_lot": None,
            "measurement_spike_fraction": 0.0,
            "measurement_spike_nm": 0.0,
        }

    measurements = (
        generate_lot_measurements(
            rng=rng,
            material=material,
            lot=lot,
            profile=profile,
        )
    )

    rows = []

    for wafer_index in range(
        WAFERS_PER_LOT
    ):
        rows.append({
            "material": material,
            "lot": lot,
            "wafer": wafer_index + 1,
            "power_set_w": round(
                float(
                    measurements[
                        "power_set"
                    ][wafer_index]
                ),
                2,
            ),
            "power_actual_w": round(
                float(
                    measurements[
                        "power_actual"
                    ][wafer_index]
                ),
                2,
            ),
            "ar_flow_set_sccm": round(
                float(
                    measurements[
                        "flow_set"
                    ][wafer_index]
                ),
                2,
            ),
            "ar_flow_actual_sccm": round(
                float(
                    measurements[
                        "flow_actual"
                    ][wafer_index]
                ),
                2,
            ),
            "pressure_set_mtorr": round(
                float(
                    measurements[
                        "pressure_set"
                    ][wafer_index]
                ),
                3,
            ),
            "pressure_actual_mtorr": round(
                float(
                    measurements[
                        "pressure_actual"
                    ][wafer_index]
                ),
                3,
            ),
            "time_set_s": round(
                float(
                    measurements[
                        "time_set"
                    ][wafer_index]
                ),
                2,
            ),
            "time_actual_s": round(
                float(
                    measurements[
                        "time_actual"
                    ][wafer_index]
                ),
                2,
            ),
            "target_usage_h": round(
                float(
                    measurements[
                        "target_usage"
                    ][wafer_index]
                ),
                2,
            ),
            "deposition_rate_nm_s": round(
                float(
                    measurements[
                        "deposition_rate"
                    ][wafer_index]
                ),
                4,
            ),
            "thickness_nm": round(
                float(
                    measurements[
                        "thickness"
                    ][wafer_index]
                ),
                2,
            ),
            "uniformity_pct": round(
                float(
                    measurements[
                        "uniformity"
                    ][wafer_index]
                ),
                2,
            ),
            "estimated_sheet_resistance_ohm_sq": round(
                float(
                    measurements[
                        "sheet_resistance"
                    ][wafer_index]
                ),
                4,
            ),
        })

    new_df = pd.DataFrame(rows)

    st.session_state.raw_data = pd.concat(
        [
            st.session_state.raw_data,
            new_df,
        ],
        ignore_index=True,
    )

    return new_df


def simulate_training_case(
    case_id,
    rng,
    difficulty,
):
    material = str(
        rng.choice(
            list(MATERIALS.keys())
        )
    )

    lot_count = int(
        rng.integers(8, 14)
    )
    has_fault = bool(
        rng.random() >= 0.15
    )

    if has_fault:
        fault_start_lot = int(
            rng.integers(
                3,
                max(
                    4,
                    lot_count - 1,
                ),
            )
        )
    else:
        fault_start_lot = None

    profile = build_case_profile(
        rng=rng,
        difficulty=difficulty,
        fault_start_lot=(
            fault_start_lot
            if fault_start_lot
            is not None
            else 3
        ),
        event_min_lot=3,
        event_max_lot=(
            lot_count - 1
        ),
        has_fault=has_fault,
    )

    summary_rows = []

    for lot in range(
        1,
        lot_count + 1,
    ):
        measurements = (
            generate_lot_measurements(
                rng=rng,
                material=material,
                lot=lot,
                profile=profile,
            )
        )

        target = (
            profile[
                "primary_cause"
            ]
            if (
                has_fault
                and fault_start_lot
                is not None
                and lot
                >= fault_start_lot
            )
            else "normal"
        )

        summary_rows.append(
            lot_measurements_to_summary(
                case_id=case_id,
                material=material,
                difficulty=difficulty,
                lot=lot,
                measurements=(
                    measurements
                ),
                target=target,
                primary_cause=profile[
                    "primary_cause"
                ],
                secondary_cause=profile[
                    "secondary_cause"
                ],
                fault_start_lot=(
                    fault_start_lot
                ),
                severity=profile[
                    "severity"
                ],
            )
        )

    return pd.DataFrame(
        summary_rows
    )


def build_ai_features(
    summary,
    material,
    difficulty="normal",
):
    if summary.empty:
        return pd.DataFrame(
            columns=AI_FEATURE_COLUMNS
        )

    work = (
        summary
        .sort_values("lot")
        .reset_index(drop=True)
        .copy()
    )
    recipe = get_recipe(material)

    baseline_rows = work.iloc[
        : min(2, len(work))
    ]

    baseline = {
        "thickness": float(
            baseline_rows[
                "mean_thickness_nm"
            ].mean()
        ),
        "std": float(
            baseline_rows[
                "std_thickness_nm"
            ].mean()
        ),
        "uniformity": float(
            baseline_rows[
                "mean_uniformity_pct"
            ].mean()
        ),
        "resistance": float(
            baseline_rows[
                "mean_sheet_resistance_ohm_sq"
            ].mean()
        ),
        "power": float(
            baseline_rows[
                "mean_power_w"
            ].mean()
        ),
        "flow": float(
            baseline_rows[
                "mean_ar_flow_sccm"
            ].mean()
        ),
        "pressure": float(
            baseline_rows[
                "mean_pressure_mtorr"
            ].mean()
        ),
        "rate": float(
            baseline_rows[
                "mean_rate_nm_s"
            ].mean()
        ),
    }

    rows = []

    for index, row in work.iterrows():
        previous = (
            work.iloc[index - 1]
            if index > 0
            else row
        )

        feature_row = {
            "material_Al": int(
                material == "Al"
            ),
            "material_Cu": int(
                material == "Cu"
            ),
            "material_Ti": int(
                material == "Ti"
            ),
            "material_Ta": int(
                material == "Ta"
            ),
            "difficulty_easy": int(
                difficulty == "easy"
            ),
            "difficulty_normal": int(
                difficulty == "normal"
            ),
            "difficulty_hard": int(
                difficulty == "hard"
            ),
            "difficulty_expert": int(
                difficulty == "expert"
            ),
            "lot_number": float(
                row["lot"]
            ),
            "mean_thickness_nm": float(
                row[
                    "mean_thickness_nm"
                ]
            ),
            "std_thickness_nm": float(
                row[
                    "std_thickness_nm"
                ]
            ),
            "mean_uniformity_pct": float(
                row[
                    "mean_uniformity_pct"
                ]
            ),
            "mean_sheet_resistance_ohm_sq": float(
                row[
                    "mean_sheet_resistance_ohm_sq"
                ]
            ),
            "oos_ratio": float(
                row["oos_count"]
                / WAFERS_PER_LOT
            ),
            "power_dev_set_pct": float(
                (
                    row["mean_power_w"]
                    - recipe["power_w"]
                )
                / recipe["power_w"]
                * 100
            ),
            "flow_dev_set_pct": float(
                (
                    row[
                        "mean_ar_flow_sccm"
                    ]
                    - recipe[
                        "ar_flow_sccm"
                    ]
                )
                / recipe[
                    "ar_flow_sccm"
                ]
                * 100
            ),
            "pressure_dev_set_pct": float(
                (
                    row[
                        "mean_pressure_mtorr"
                    ]
                    - recipe[
                        "pressure_mtorr"
                    ]
                )
                / recipe[
                    "pressure_mtorr"
                ]
                * 100
            ),
            "rate_dev_set_pct": float(
                (
                    row["mean_rate_nm_s"]
                    - recipe[
                        "base_rate_nm_s"
                    ]
                )
                / recipe[
                    "base_rate_nm_s"
                ]
                * 100
            ),
            "thickness_delta_baseline": float(
                row[
                    "mean_thickness_nm"
                ]
                - baseline["thickness"]
            ),
            "std_delta_baseline": float(
                row[
                    "std_thickness_nm"
                ]
                - baseline["std"]
            ),
            "uniformity_delta_baseline": float(
                row[
                    "mean_uniformity_pct"
                ]
                - baseline[
                    "uniformity"
                ]
            ),
            "resistance_delta_baseline_pct": float(
                (
                    row[
                        "mean_sheet_resistance_ohm_sq"
                    ]
                    - baseline[
                        "resistance"
                    ]
                )
                / max(
                    abs(
                        baseline[
                            "resistance"
                        ]
                    ),
                    1e-9,
                )
                * 100
            ),
            "power_delta_baseline_pct": float(
                (
                    row["mean_power_w"]
                    - baseline["power"]
                )
                / recipe["power_w"]
                * 100
            ),
            "flow_delta_baseline_pct": float(
                (
                    row[
                        "mean_ar_flow_sccm"
                    ]
                    - baseline["flow"]
                )
                / recipe[
                    "ar_flow_sccm"
                ]
                * 100
            ),
            "pressure_delta_baseline_pct": float(
                (
                    row[
                        "mean_pressure_mtorr"
                    ]
                    - baseline[
                        "pressure"
                    ]
                )
                / recipe[
                    "pressure_mtorr"
                ]
                * 100
            ),
            "rate_delta_baseline_pct": float(
                (
                    row["mean_rate_nm_s"]
                    - baseline["rate"]
                )
                / recipe[
                    "base_rate_nm_s"
                ]
                * 100
            ),
            "thickness_delta_previous": float(
                row[
                    "mean_thickness_nm"
                ]
                - previous[
                    "mean_thickness_nm"
                ]
            ),
            "uniformity_delta_previous": float(
                row[
                    "mean_uniformity_pct"
                ]
                - previous[
                    "mean_uniformity_pct"
                ]
            ),
            "resistance_delta_previous_pct": float(
                (
                    row[
                        "mean_sheet_resistance_ohm_sq"
                    ]
                    - previous[
                        "mean_sheet_resistance_ohm_sq"
                    ]
                )
                / max(
                    abs(
                        previous[
                            "mean_sheet_resistance_ohm_sq"
                        ]
                    ),
                    1e-9,
                )
                * 100
            ),
        }

        rows.append(feature_row)

    return pd.DataFrame(
        rows,
        columns=AI_FEATURE_COLUMNS,
    )


@st.cache_data(
    show_spinner=False,
)
def generate_ai_training_dataset(
    case_count,
    dataset_seed,
    training_scope,
):
    rng = np.random.default_rng(
        int(dataset_seed)
    )

    all_rows = []

    if training_scope == "mixed":
        difficulty_choices = (
            DIFFICULTY_ORDER
        )
        difficulty_probabilities = [
            0.20,
            0.35,
            0.30,
            0.15,
        ]
    else:
        difficulty_choices = [
            training_scope
        ]
        difficulty_probabilities = [
            1.0
        ]

    for case_id in range(
        1,
        int(case_count) + 1,
    ):
        difficulty = str(
            rng.choice(
                difficulty_choices,
                p=difficulty_probabilities,
            )
        )

        case_summary = (
            simulate_training_case(
                case_id=case_id,
                rng=rng,
                difficulty=difficulty,
            )
        )

        material = str(
            case_summary[
                "material"
            ].iloc[0]
        )

        features = build_ai_features(
            summary=case_summary,
            material=material,
            difficulty=difficulty,
        )

        features.insert(
            0,
            "case_id",
            case_id,
        )
        features["material"] = material
        features["difficulty"] = (
            difficulty
        )
        features["lot"] = (
            case_summary[
                "lot"
            ].to_numpy()
        )
        features["target"] = (
            case_summary[
                "target"
            ].to_numpy()
        )
        features[
            "actual_case_cause"
        ] = case_summary[
            "actual_case_cause"
        ].to_numpy()
        features[
            "secondary_cause"
        ] = case_summary[
            "secondary_cause"
        ].to_numpy()
        features[
            "actual_fault_start_lot"
        ] = case_summary[
            "actual_fault_start_lot"
        ].to_numpy()
        features["severity"] = (
            case_summary[
                "severity"
            ].to_numpy()
        )

        all_rows.append(features)

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


@st.cache_resource(
    show_spinner=False,
)
def train_ai_model(
    case_count,
    dataset_seed,
    training_scope,
):
    dataset = (
        generate_ai_training_dataset(
            int(case_count),
            int(dataset_seed),
            str(training_scope),
        )
    )

    x = dataset[
        AI_FEATURE_COLUMNS
    ]
    y = dataset["target"]
    groups = dataset["case_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_index, test_index = next(
        splitter.split(
            x,
            y,
            groups=groups,
        )
    )

    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight=(
            "balanced_subsample"
        ),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    prediction = model.predict(
        x_test
    )
    probability = model.predict_proba(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction,
    )
    macro_f1 = f1_score(
        y_test,
        prediction,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        prediction,
        labels=AI_CLASS_ORDER,
    )

    report = classification_report(
        y_test,
        prediction,
        labels=AI_CLASS_ORDER,
        target_names=[
            AI_LABELS[label]
            for label in AI_CLASS_ORDER
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "분류",
                "precision": "정밀도",
                "recall": "재현율",
                "f1-score": "F1",
                "support": "표본 수",
            }
        )
    )

    importance_df = (
        pd.DataFrame({
            "feature": (
                AI_FEATURE_COLUMNS
            ),
            "importance": (
                model.feature_importances_
            ),
        })
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df[
        "feature_label"
    ] = importance_df[
        "feature"
    ].map(AI_FEATURE_LABELS)

    test_dataset = (
        dataset.iloc[
            test_index
        ]
        .copy()
        .reset_index(drop=True)
    )
    test_dataset[
        "prediction"
    ] = prediction

    probability_df = pd.DataFrame(
        probability,
        columns=list(
            model.classes_
        ),
    )

    for class_name in (
        model.classes_
    ):
        test_dataset[
            f"probability_{class_name}"
        ] = probability_df[
            class_name
        ].to_numpy()

    difficulty_rows = []

    for difficulty in DIFFICULTY_ORDER:
        difficulty_subset = (
            test_dataset[
                test_dataset[
                    "difficulty"
                ]
                == difficulty
            ]
        )

        if difficulty_subset.empty:
            continue

        row_accuracy = accuracy_score(
            difficulty_subset[
                "target"
            ],
            difficulty_subset[
                "prediction"
            ],
        )
        row_macro_f1 = f1_score(
            difficulty_subset[
                "target"
            ],
            difficulty_subset[
                "prediction"
            ],
            average="macro",
            zero_division=0,
        )

        case_records = []

        for (
            case_id,
            case_data,
        ) in difficulty_subset.groupby(
            "case_id"
        ):
            case_data = (
                case_data
                .sort_values("lot")
                .reset_index(drop=True)
            )

            actual_start = int(
                case_data[
                    "actual_fault_start_lot"
                ].iloc[0]
            )
            primary_cause = str(
                case_data[
                    "actual_case_cause"
                ].iloc[0]
            )
            secondary_cause = str(
                case_data[
                    "secondary_cause"
                ].iloc[0]
            )

            normal_probability = (
                case_data[
                    "probability_normal"
                ]
                if "probability_normal"
                in case_data.columns
                else pd.Series(
                    np.zeros(
                        len(case_data)
                    )
                )
            )

            case_data = case_data.copy()
            case_data[
                "abnormal_probability"
            ] = (
                1.0
                - normal_probability.to_numpy()
            )

            threshold = DIFFICULTIES[
                difficulty
            ]["ai_threshold"]

            predicted_start = None

            for row_index in range(
                len(case_data)
            ):
                current_probability = float(
                    case_data.loc[
                        row_index,
                        "abnormal_probability",
                    ]
                )
                next_probability = (
                    float(
                        case_data.loc[
                            row_index + 1,
                            "abnormal_probability",
                        ]
                    )
                    if row_index + 1
                    < len(case_data)
                    else current_probability
                )

                if (
                    current_probability
                    >= threshold
                    and (
                        next_probability
                        >= threshold
                        - 0.05
                        or current_probability
                        >= threshold
                        + 0.15
                    )
                ):
                    predicted_start = int(
                        case_data.loc[
                            row_index,
                            "lot",
                        ]
                    )
                    break

            ranked_causes = []

            if predicted_start is not None:
                after_detection = (
                    case_data[
                        case_data["lot"]
                        >= predicted_start
                    ]
                )

                average_probabilities = {
                    cause: float(
                        after_detection[
                            f"probability_{cause}"
                        ].mean()
                    )
                    for cause in CAUSES
                    if (
                        f"probability_{cause}"
                        in after_detection.columns
                    )
                }

                ranked_causes = sorted(
                    average_probabilities,
                    key=average_probabilities.get,
                    reverse=True,
                )

            predicted_primary = (
                ranked_causes[0]
                if ranked_causes
                else None
            )

            top_two = ranked_causes[:2]

            case_records.append({
                "is_fault_case": (
                    actual_start > 0
                ),
                "actual_start": (
                    actual_start
                ),
                "predicted_start": (
                    predicted_start
                ),
                "primary_cause": (
                    primary_cause
                ),
                "secondary_cause": (
                    secondary_cause
                ),
                "predicted_primary": (
                    predicted_primary
                ),
                "top_two": top_two,
            })

        case_frame = pd.DataFrame(
            case_records
        )

        fault_cases = case_frame[
            case_frame[
                "is_fault_case"
            ]
        ]
        normal_cases = case_frame[
            ~case_frame[
                "is_fault_case"
            ]
        ]

        exact_accuracy = (
            float(
                (
                    fault_cases[
                        "predicted_start"
                    ]
                    == fault_cases[
                        "actual_start"
                    ]
                ).mean()
            )
            if not fault_cases.empty
            else np.nan
        )

        predicted_start_numeric = (
            pd.to_numeric(
                fault_cases[
                    "predicted_start"
                ],
                errors="coerce",
            )
        )

        plus_minus_one_accuracy = (
            float(
                (
                    (
                        predicted_start_numeric
                        - fault_cases[
                            "actual_start"
                        ]
                    ).abs()
                    <= 1
                ).mean()
            )
            if not fault_cases.empty
            else np.nan
        )

        cause_accuracy = (
            float(
                (
                    fault_cases[
                        "predicted_primary"
                    ]
                    == fault_cases[
                        "primary_cause"
                    ]
                ).mean()
            )
            if not fault_cases.empty
            else np.nan
        )

        top_two_hits = []

        for _, row in fault_cases.iterrows():
            top_two = row["top_two"]
            top_two_hits.append(
                (
                    row[
                        "primary_cause"
                    ]
                    in top_two
                )
                or (
                    bool(
                        row[
                            "secondary_cause"
                        ]
                    )
                    and row[
                        "secondary_cause"
                    ]
                    in top_two
                )
            )

        top_two_accuracy = (
            float(
                np.mean(
                    top_two_hits
                )
            )
            if top_two_hits
            else np.nan
        )

        false_alarm_rate = (
            float(
                normal_cases[
                    "predicted_start"
                ].notna().mean()
            )
            if not normal_cases.empty
            else np.nan
        )

        difficulty_rows.append({
            "난이도": DIFFICULTY_LABELS[
                difficulty
            ],
            "Lot 분류 정확도": (
                row_accuracy
            ),
            "Macro F1": row_macro_f1,
            "이상 Lot 정확 일치": (
                exact_accuracy
            ),
            "±1 Lot 이내": (
                plus_minus_one_accuracy
            ),
            "원인 정확도": (
                cause_accuracy
            ),
            "Top-2 원인 적중률": (
                top_two_accuracy
            ),
            "정상 Case 오탐률": (
                false_alarm_rate
            ),
            "시험 Case": int(
                case_frame.shape[0]
            ),
        })

    difficulty_metrics = pd.DataFrame(
        difficulty_rows
    )

    train_case_count = int(
        dataset.iloc[
            train_index
        ]["case_id"].nunique()
    )
    test_case_count = int(
        dataset.iloc[
            test_index
        ]["case_id"].nunique()
    )

    return {
        "model": model,
        "dataset": dataset,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix,
        "classification_report": (
            report_df
        ),
        "feature_importance": (
            importance_df
        ),
        "difficulty_metrics": (
            difficulty_metrics
        ),
        "train_rows": int(
            len(train_index)
        ),
        "test_rows": int(
            len(test_index)
        ),
        "train_cases": (
            train_case_count
        ),
        "test_cases": (
            test_case_count
        ),
        "case_count": int(
            case_count
        ),
        "dataset_seed": int(
            dataset_seed
        ),
        "training_scope": str(
            training_scope
        ),
    }


def ai_diagnose_current_case(
    ai_bundle,
    summary,
    material,
    difficulty,
):
    if (
        ai_bundle is None
        or summary.empty
    ):
        return None

    features = build_ai_features(
        summary=summary,
        material=material,
        difficulty=difficulty,
    )

    model = ai_bundle["model"]
    probability = model.predict_proba(
        features[
            AI_FEATURE_COLUMNS
        ]
    )

    probability_df = pd.DataFrame(
        probability,
        columns=list(
            model.classes_
        ),
    )

    normal_probability = (
        probability_df["normal"]
        if "normal"
        in probability_df.columns
        else pd.Series(
            np.zeros(
                len(features)
            )
        )
    )

    abnormal_probability = (
        1.0
        - normal_probability
    )

    cause_columns = [
        cause
        for cause in CAUSES
        if cause
        in probability_df.columns
    ]

    lot_predictions = []

    for index, row in summary.reset_index(
        drop=True
    ).iterrows():
        cause_probabilities = (
            probability_df.loc[
                index,
                cause_columns,
            ]
        )

        ranked_causes = (
            cause_probabilities
            .sort_values(
                ascending=False
            )
        )

        predicted_cause_key = str(
            ranked_causes.index[0]
        )
        predicted_cause_probability = float(
            ranked_causes.iloc[0]
        )
        abnormal_prob = float(
            abnormal_probability.iloc[
                index
            ]
        )

        threshold = DIFFICULTIES[
            difficulty
        ]["ai_threshold"]

        predicted_state = (
            predicted_cause_key
            if abnormal_prob
            >= threshold
            else "normal"
        )

        lot_predictions.append({
            "Lot": int(row["lot"]),
            "AI 판정": AI_LABELS[
                predicted_state
            ],
            "이상 확률": abnormal_prob,
            "1순위 원인": CAUSES[
                predicted_cause_key
            ],
            "1순위 확률": (
                predicted_cause_probability
            ),
            "2순위 원인": (
                CAUSES[
                    str(
                        ranked_causes.index[1]
                    )
                ]
                if len(
                    ranked_causes
                ) > 1
                else ""
            ),
            "2순위 확률": (
                float(
                    ranked_causes.iloc[1]
                )
                if len(
                    ranked_causes
                ) > 1
                else 0.0
            ),
        })

    prediction_table = pd.DataFrame(
        lot_predictions
    )

    threshold = DIFFICULTIES[
        difficulty
    ]["ai_threshold"]

    first_fault_index = None

    for index in range(
        len(prediction_table)
    ):
        current_probability = float(
            prediction_table.loc[
                index,
                "이상 확률",
            ]
        )

        next_probability = (
            float(
                prediction_table.loc[
                    index + 1,
                    "이상 확률",
                ]
            )
            if index + 1
            < len(prediction_table)
            else current_probability
        )

        if (
            current_probability
            >= threshold
            and (
                next_probability
                >= threshold
                - 0.05
                or current_probability
                >= threshold
                + 0.15
            )
        ):
            first_fault_index = index
            break

    if first_fault_index is None:
        return {
            "fault_lot": None,
            "cause_key": None,
            "cause_label": "판단 유보",
            "confidence": float(
                abnormal_probability.max()
            ),
            "confidence_label": (
                "낮음"
            ),
            "top_causes": [],
            "composite_possible": False,
            "prediction_table": (
                prediction_table
            ),
        }

    fault_lot = int(
        prediction_table.loc[
            first_fault_index,
            "Lot",
        ]
    )

    aggregate_probability = (
        probability_df.loc[
            first_fault_index:,
            cause_columns,
        ].mean()
    )

    ranked_aggregate = (
        aggregate_probability
        .sort_values(
            ascending=False
        )
    )

    top_causes = []

    for cause_key, probability_value in (
        ranked_aggregate.iloc[:2].items()
    ):
        top_causes.append({
            "cause_key": str(
                cause_key
            ),
            "cause_label": CAUSES[
                str(cause_key)
            ],
            "probability": float(
                probability_value
            ),
        })

    top_probability = (
        top_causes[0][
            "probability"
        ]
        if top_causes
        else 0.0
    )
    second_probability = (
        top_causes[1][
            "probability"
        ]
        if len(top_causes) > 1
        else 0.0
    )
    probability_gap = (
        top_probability
        - second_probability
    )

    if (
        top_probability >= 0.75
        and probability_gap >= 0.20
    ):
        confidence_label = "높음"
    elif top_probability >= 0.50:
        confidence_label = "보통"
    else:
        confidence_label = "낮음"

    composite_possible = (
        second_probability >= 0.25
        and probability_gap <= 0.25
    )

    cause_label = (
        top_causes[0][
            "cause_label"
        ]
        if top_causes
        else "판단 유보"
    )

    return {
        "fault_lot": fault_lot,
        "cause_key": (
            top_causes[0][
                "cause_key"
            ]
            if top_causes
            else None
        ),
        "cause_label": cause_label,
        "confidence": (
            top_probability
        ),
        "confidence_label": (
            confidence_label
        ),
        "top_causes": top_causes,
        "composite_possible": (
            composite_possible
        ),
        "prediction_table": (
            prediction_table
        ),
    }


def automatic_abnormal_lot(summary):
    if len(summary) < 3:
        return None

    baseline = summary.iloc[:2]

    thickness_center = baseline[
        "mean_thickness_nm"
    ].mean()
    thickness_std = max(
        baseline["mean_thickness_nm"].std(ddof=1),
        0.12,
    )

    power_center = baseline[
        "mean_power_w"
    ].mean()
    flow_center = baseline[
        "mean_ar_flow_sccm"
    ].mean()
    pressure_center = baseline[
        "mean_pressure_mtorr"
    ].mean()
    rate_center = baseline[
        "mean_rate_nm_s"
    ].mean()
    resistance_center = baseline[
        "mean_sheet_resistance_ohm_sq"
    ].mean()

    for _, row in summary.iloc[2:].iterrows():
        signals = [
            abs(
                row["mean_thickness_nm"]
                - thickness_center
            )
            > max(3 * thickness_std, 0.8),

            abs(
                row["mean_power_w"]
                - power_center
            )
            > max(
                MATERIALS[
                    st.session_state.active_material
                ]["power_w"]
                * 0.009,
                4.5,
            ),

            abs(
                row["mean_ar_flow_sccm"]
                - flow_center
            )
            > 0.7,

            abs(
                row["mean_pressure_mtorr"]
                - pressure_center
            )
            > 0.20,

            abs(
                row["mean_rate_nm_s"]
                - rate_center
            )
            > max(rate_center * 0.006, 0.003),

            row["mean_uniformity_pct"] > 2.0,

            abs(
                row["mean_sheet_resistance_ohm_sq"]
                - resistance_center
            )
            > max(resistance_center * 0.06, 0.01),
        ]

        if sum(signals) >= 2:
            return int(row["lot"])

    return None


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


ensure_state()

st.title(
    "가상 반도체 박막 공정 이상 진단 시뮬레이터"
)
st.caption(
    f"현재 버전 {APP_VERSION} · 최종 업데이트 {LAST_UPDATED} · "
    "난이도 선택, 현실적 합성 데이터, AI Top-2 진단"
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("업데이트 내역", expanded=False):
        st.markdown(
            """
            - **v2.1.0**: 난이도 선택, 신호 중첩·센서 편향·일시 변동·부분 Wafer·복합 이상, 난이도별 AI 평가
            - **v2.0.0**: 가상 Case 자동 생성, Random Forest 학습, 사람·통계·AI 진단 비교
            - **v1.2.2**: Lot 증가 시 두께 축 자동 확장, X축 눈금 가로 표시 및 간격 최적화
            - **v1.2.1**: 본문 최대 폭 제한, 중앙 정렬, 넓은 모니터 레이아웃 개선
            - **v1.2.0**: 분석 화면 2열 대시보드 구성, 그래프 크기 축소, Raw Data 분리
            - **v1.1.1**: 공정 변수 그래프 툴팁 명칭 개선, 버전 정보 추가
            - **v1.1**: Al·Cu·Ti·Ta 선택, 재료별 Recipe, 예상 면저항 기능 추가
            - **v1.0**: Sputter Lot 생성, 랜덤 이상 주입, 진단 및 정답 비교 기능 구현
            """
        )

with info_col2:
    with st.expander("프로젝트 안내와 모델의 한계", expanded=False):
        st.markdown(
            """
            - 실제 기업 Recipe나 생산 데이터를 사용하지 않은 교육용 합성 데이터 시뮬레이터입니다.
            - 재료별 증착률과 Recipe는 비교 학습을 위해 단순화한 가정값입니다.
            - 녹는점과 밀도는 참고 정보이며 Sputter 계산식에 직접 사용하지 않습니다.
            - 면저항은 재료의 기준 비저항과 생성된 두께를 이용한 추정값이며 실제 측정값이 아닙니다.
            - 실제 박막 비저항은 결정상, 입도, 불순물, 표면·입계 산란 등에 따라 달라질 수 있습니다.
            - 어려움·전문가 모드의 복합 이상은 교육용 규칙으로 생성되며 실제 장비 고장 빈도를 의미하지 않습니다.
            - AI 성능은 이 시뮬레이터가 생성한 미사용 합성 Case에 대한 결과이며 실제 Fab 성능이 아닙니다.
            """
        )

material_options = list(
    MATERIALS.keys()
)
difficulty_options = (
    DIFFICULTY_ORDER
)

selector_col1, selector_col2 = (
    st.columns(2)
)

with selector_col1:
    selected_material = st.selectbox(
        "증착 재료 선택",
        options=material_options,
        index=material_options.index(
            st.session_state.active_material
        ),
        format_func=lambda symbol: (
            f"{symbol} · "
            f"{MATERIALS[symbol]['name_ko']}"
        ),
    )

with selector_col2:
    selected_difficulty = st.selectbox(
        "Case 난이도",
        options=difficulty_options,
        index=difficulty_options.index(
            st.session_state.active_difficulty
        ),
        format_func=lambda key: (
            f"{DIFFICULTY_LABELS[key]} · "
            f"{DIFFICULTIES[key]['description']}"
        ),
    )

if (
    selected_material
    != st.session_state.active_material
    or selected_difficulty
    != st.session_state.active_difficulty
):
    st.warning(
        "재료 또는 난이도가 변경되었습니다. "
        "아래의 '새 Case 초기화'를 눌러야 적용됩니다."
    )

active_material = (
    st.session_state.active_material
)
active_difficulty = (
    st.session_state.active_difficulty
)
material_props = MATERIALS[active_material]
recipe = get_recipe(active_material)

st.subheader(
    f"Case 01 · {active_material} 박막 "
    "DC Magnetron Sputtering · "
    f"{DIFFICULTY_LABELS[active_difficulty]}"
)

condition_col, recipe_col, property_col = (
    st.columns(3)
)

with condition_col:
    st.markdown(
        f"""
        **생산 조건**

        - 기판: 300 mm Si Wafer
        - Lot 크기: {WAFERS_PER_LOT}장
        - 증착막: {active_material}
        - 난이도: {DIFFICULTY_LABELS[active_difficulty]}
        - 목표 두께: {TARGET_THICKNESS:.0f} nm
        - 두께 관리 범위: {LOWER_SPEC:.0f}~{UPPER_SPEC:.0f} nm
        - 균일도 기준: {UNIFORMITY_LIMIT:.1f}% 이하
        """
    )

with recipe_col:
    st.dataframe(
        pd.DataFrame({
            "교육용 Recipe": [
                "DC Power",
                "Ar Flow",
                "Chamber Pressure",
                "Deposition Time",
                "기준 증착률",
            ],
            "Setpoint": [
                f"{recipe['power_w']:.0f} W",
                f"{recipe['ar_flow_sccm']:.0f} sccm",
                f"{recipe['pressure_mtorr']:.1f} mTorr",
                f"{recipe['deposition_time_s']:.1f} s",
                f"{recipe['base_rate_nm_s']:.2f} nm/s",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

with property_col:
    st.dataframe(
        pd.DataFrame({
            "재료 정보": [
                "기준 비저항",
                "밀도",
                "녹는점",
                "100 nm 기준 예상 면저항",
            ],
            "값": [
                (
                    f"{material_props['bulk_resistivity_uohm_cm']}"
                    " μΩ·cm"
                ),
                (
                    f"{material_props['density_g_cm3']}"
                    " g/cm³"
                ),
                (
                    f"{material_props['melting_point_c']}"
                    " °C"
                ),
                (
                    f"{reference_sheet_resistance(active_material):.4f}"
                    " Ω/□"
                ),
            ],
            "모델 반영": [
                "면저항 계산",
                "참고 정보",
                "참고 정보",
                "비교 기준",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    material_props["description"]
    + " · "
    + DIFFICULTIES[
        active_difficulty
    ]["description"]
)
st.divider()

seed_input = st.number_input(
    "재현용 Case Seed",
    min_value=1,
    max_value=999_999_999,
    value=int(st.session_state.case_seed),
    step=1,
    help=(
        "같은 재료·Seed·버튼 순서를 사용하면 "
        "동일한 Case를 다시 생성할 수 있습니다."
    ),
)

button_col1, button_col2, button_col3 = (
    st.columns(3)
)

with button_col1:
    if st.button(
        "새 Case 초기화",
        use_container_width=True,
    ):
        initialize_case(
            material=selected_material,
            difficulty=selected_difficulty,
            seed=int(seed_input),
        )
        st.rerun()

with button_col2:
    if st.button(
        "다음 Lot 생산",
        type="primary",
        use_container_width=True,
    ):
        new_lot = produce_lot()

        oos_count = int(
            (
                (new_lot["thickness_nm"] < LOWER_SPEC)
                | (new_lot["thickness_nm"] > UPPER_SPEC)
            ).sum()
        )

        uniformity_fail_count = int(
            (
                new_lot["uniformity_pct"]
                > UNIFORMITY_LIMIT
            ).sum()
        )

        st.success(
            f"{active_material} Lot "
            f"{st.session_state.current_lot} 생산 완료 · "
            f"두께 이탈 {oos_count}장 · "
            f"균일도 초과 {uniformity_fail_count}장"
        )

with button_col3:
    if st.button(
        "랜덤 이상 예약",
        use_container_width=True,
    ):
        success, message = inject_fault()

        if success:
            st.success(message)
        else:
            st.warning(message)

if st.session_state.cause_key is None:
    st.info(
        "진행 순서: 정상 Lot 2개 생산 → "
        "랜덤 이상 예약 → Lot 4~5개 추가 생산 → "
        "진단 제출"
    )
else:
    st.info(
        "이상이 예약되어 있습니다. "
        "원인과 시작 Lot은 숨겨져 있습니다. "
        "현재 난이도에서는 복합 이상·센서 편향·"
        "일시 변동이 포함될 수 있습니다."
    )

raw_df = st.session_state.raw_data
summary_df = summarize(raw_df)

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
    st.columns(5)
)

metric_col1.metric(
    "현재 재료",
    active_material,
)
metric_col2.metric(
    "난이도",
    DIFFICULTY_LABELS[
        active_difficulty
    ],
)
metric_col3.metric(
    "누적 생산 Lot",
    st.session_state.current_lot,
)
metric_col4.metric(
    "누적 Wafer",
    len(raw_df),
)
metric_col5.metric(
    "Case Seed",
    st.session_state.case_seed,
)

if not summary_df.empty:
    st.subheader("공정 분석 대시보드")

    lot_min = int(summary_df["lot"].min())
    lot_max = int(summary_df["lot"].max())
    lot_count = max(lot_max - lot_min + 1, 1)
    lot_tick_count = min(lot_count, 10)

    lot_axis = alt.Axis(
        title="Lot",
        labelAngle=0,
        tickCount=lot_tick_count,
        tickMinStep=1,
        labelOverlap="greedy",
    )

    thickness_min = min(
        float(raw_df["thickness_nm"].min()),
        LOWER_SPEC,
    )
    thickness_max = max(
        float(raw_df["thickness_nm"].max()),
        UPPER_SPEC,
    )
    thickness_span = max(
        thickness_max - thickness_min,
        1.0,
    )
    thickness_margin = max(
        thickness_span * 0.08,
        0.25,
    )
    thickness_domain = [
        thickness_min - thickness_margin,
        thickness_max + thickness_margin,
    ]

    dashboard_col1, dashboard_col2 = st.columns(2)

    # -----------------------------------------------------
    # 왼쪽 위: 두께 추이
    # -----------------------------------------------------
    with dashboard_col1:
        st.markdown("#### 1. 두께 추이")

        wafer_points = (
            alt.Chart(raw_df)
            .mark_circle(size=48, opacity=0.42)
            .encode(
                x=alt.X(
                    "lot:Q",
                    axis=lot_axis,
                    scale=alt.Scale(
                        domain=[
                            lot_min - 0.4,
                            lot_max + 0.4,
                        ]
                    ),
                ),
                y=alt.Y(
                    "thickness_nm:Q",
                    title="Thickness (nm)",
                    scale=alt.Scale(
                        domain=thickness_domain,
                        nice=True,
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "lot:Q",
                        title="Lot",
                        format=".0f",
                    ),
                    alt.Tooltip(
                        "wafer:Q",
                        title="Wafer",
                    ),
                    alt.Tooltip(
                        "thickness_nm:Q",
                        title="두께 (nm)",
                        format=".2f",
                    ),
                ],
            )
        )

        lot_mean_line = (
            alt.Chart(summary_df)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "lot:Q",
                    axis=lot_axis,
                    scale=alt.Scale(
                        domain=[
                            lot_min - 0.4,
                            lot_max + 0.4,
                        ]
                    ),
                ),
                y=alt.Y(
                    "mean_thickness_nm:Q",
                    title="Thickness (nm)",
                    scale=alt.Scale(
                        domain=thickness_domain,
                        nice=True,
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "lot:Q",
                        title="Lot",
                        format=".0f",
                    ),
                    alt.Tooltip(
                        "mean_thickness_nm:Q",
                        title="Lot 평균 (nm)",
                        format=".3f",
                    ),
                ],
            )
        )

        thickness_limits = pd.DataFrame({
            "기준": ["USL", "Target", "LSL"],
            "두께": [
                UPPER_SPEC,
                TARGET_THICKNESS,
                LOWER_SPEC,
            ],
        })

        limit_rules = (
            alt.Chart(thickness_limits)
            .mark_rule(strokeDash=[5, 4])
            .encode(
                y=alt.Y(
                    "두께:Q",
                    title="Thickness (nm)",
                ),
                strokeDash=alt.StrokeDash(
                    "기준:N",
                    title="기준",
                ),
                tooltip=[
                    alt.Tooltip(
                        "기준:N",
                        title="기준",
                    ),
                    alt.Tooltip(
                        "두께:Q",
                        title="두께 (nm)",
                        format=".1f",
                    ),
                ],
            )
        )

        thickness_chart = (
            wafer_points
            + lot_mean_line
            + limit_rules
        ).properties(height=285)

        st.altair_chart(
            thickness_chart,
            width="stretch",
        )

    # -----------------------------------------------------
    # 오른쪽 위: 공정 변수 변화
    # -----------------------------------------------------
    with dashboard_col2:
        st.markdown(
            "#### 2. Setpoint 대비 공정 변수 변화"
        )

        process_deviation = pd.DataFrame({
            "Lot": summary_df["lot"],
            "Power (%)": (
                (
                    summary_df["mean_power_w"]
                    - recipe["power_w"]
                )
                / recipe["power_w"]
                * 100
            ),
            "Ar Flow (%)": (
                (
                    summary_df["mean_ar_flow_sccm"]
                    - recipe["ar_flow_sccm"]
                )
                / recipe["ar_flow_sccm"]
                * 100
            ),
            "Pressure (%)": (
                (
                    summary_df["mean_pressure_mtorr"]
                    - recipe["pressure_mtorr"]
                )
                / recipe["pressure_mtorr"]
                * 100
            ),
        })

        process_long = process_deviation.melt(
            id_vars="Lot",
            var_name="공정 변수",
            value_name="Setpoint 대비 편차 (%)",
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"기준": [0]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(y="기준:Q")
        )

        process_chart = (
            alt.Chart(process_long)
            .mark_line(point=True, strokeWidth=2.3)
            .encode(
                x=alt.X(
                    "Lot:Q",
                    axis=lot_axis,
                    scale=alt.Scale(
                        domain=[
                            lot_min - 0.4,
                            lot_max + 0.4,
                        ]
                    ),
                ),
                y=alt.Y(
                    "Setpoint 대비 편차 (%):Q",
                    title="Setpoint 대비 편차 (%)",
                ),
                color=alt.Color(
                    "공정 변수:N",
                    title="공정 변수",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Lot:Q",
                        title="Lot",
                        format=".0f",
                    ),
                    alt.Tooltip(
                        "공정 변수:N",
                        title="공정 변수",
                    ),
                    alt.Tooltip(
                        "Setpoint 대비 편차 (%):Q",
                        title="편차 (%)",
                        format=".3f",
                    ),
                ],
            )
            .properties(height=285)
        )

        st.altair_chart(
            process_chart + zero_line,
            width="stretch",
        )

    dashboard_col3, dashboard_col4 = st.columns(2)

    # -----------------------------------------------------
    # 왼쪽 아래: 면저항 추이
    # -----------------------------------------------------
    with dashboard_col3:
        st.markdown("#### 3. 예상 면저항 추이")

        resistance_reference = pd.DataFrame({
            "기준 면저항": [
                reference_sheet_resistance(
                    active_material
                )
            ]
        })

        resistance_line = (
            alt.Chart(summary_df)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "lot:Q",
                    axis=lot_axis,
                    scale=alt.Scale(
                        domain=[
                            lot_min - 0.4,
                            lot_max + 0.4,
                        ]
                    ),
                ),
                y=alt.Y(
                    "mean_sheet_resistance_ohm_sq:Q",
                    title="Estimated sheet resistance (Ω/□)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(
                        "lot:Q",
                        title="Lot",
                        format=".0f",
                    ),
                    alt.Tooltip(
                        "mean_sheet_resistance_ohm_sq:Q",
                        title="예상 면저항 (Ω/□)",
                        format=".4f",
                    ),
                ],
            )
        )

        resistance_rule = (
            alt.Chart(resistance_reference)
            .mark_rule(strokeDash=[5, 4])
            .encode(
                y=alt.Y(
                    "기준 면저항:Q",
                    title="Estimated sheet resistance (Ω/□)",
                ),
                tooltip=[
                    alt.Tooltip(
                        "기준 면저항:Q",
                        title="100 nm 기준 (Ω/□)",
                        format=".4f",
                    )
                ],
            )
        )

        resistance_chart = (
            resistance_line
            + resistance_rule
        ).properties(height=260)

        st.altair_chart(
            resistance_chart,
            width="stretch",
        )

    # -----------------------------------------------------
    # 오른쪽 아래: Lot 요약
    # -----------------------------------------------------
    with dashboard_col4:
        st.markdown("#### 4. Lot 요약")

        compact_summary = summary_df[
            [
                "lot",
                "mean_thickness_nm",
                "std_thickness_nm",
                "mean_uniformity_pct",
                "mean_sheet_resistance_ohm_sq",
                "oos_count",
            ]
        ].rename(
            columns={
                "lot": "Lot",
                "mean_thickness_nm": "평균 두께 (nm)",
                "std_thickness_nm": "두께 표준편차",
                "mean_uniformity_pct": "평균 균일도 (%)",
                "mean_sheet_resistance_ohm_sq": "예상 면저항 (Ω/□)",
                "oos_count": "두께 이탈 수",
            }
        )

        st.dataframe(
            compact_summary,
            hide_index=True,
            width="stretch",
            height=300,
        )

    raw_tab, full_summary_tab = st.tabs(
        [
            "Wafer별 Raw Data",
            "전체 Lot 요약 데이터",
        ]
    )

    with raw_tab:
        st.dataframe(
            raw_df,
            hide_index=True,
            width="stretch",
            height=360,
        )

        st.download_button(
            "Raw Data CSV 다운로드",
            data=csv_bytes(raw_df),
            file_name=(
                f"sputter_{active_material}_raw_"
                f"seed_{st.session_state.case_seed}.csv"
            ),
            mime="text/csv",
        )

    with full_summary_tab:
        st.dataframe(
            summary_df,
            hide_index=True,
            width="stretch",
            height=360,
        )

st.divider()
st.subheader("5. AI 모델 학습 및 성능")

st.caption(
    "가상 Case를 자동 생성한 뒤 Lot 단위 특징으로 Random Forest를 학습합니다. "
    "학습·시험 데이터는 Case 단위로 분리하며, 난이도별 성능과 오탐률을 따로 평가합니다."
)

ai_setting_col1, ai_setting_col2, ai_setting_col3, ai_setting_col4 = st.columns(
    [1, 1, 1, 1]
)

with ai_setting_col1:
    ai_case_count = st.selectbox(
        "학습용 가상 Case 수",
        options=[
            500,
            1000,
            2000,
            3000,
        ],
        index=3,
    )

with ai_setting_col2:
    training_scope = st.selectbox(
        "학습 난이도 범위",
        options=[
            "mixed",
            "easy",
            "normal",
            "hard",
            "expert",
        ],
        index=0,
        format_func=lambda key: (
            "전체 난이도 혼합"
            if key == "mixed"
            else DIFFICULTY_LABELS[key]
        ),
    )

with ai_setting_col3:
    ai_dataset_seed = st.number_input(
        "학습 데이터 Seed",
        min_value=1,
        max_value=999_999_999,
        value=20260805,
        step=1,
    )

with ai_setting_col4:
    st.write("")
    st.write("")
    train_ai_button = st.button(
        "가상 데이터 생성 및 AI 학습",
        type="primary",
        width="stretch",
    )

if train_ai_button:
    with st.spinner(
        "난이도별 가상 Case 생성과 AI 학습을 진행하고 있습니다..."
    ):
        st.session_state.ai_bundle = train_ai_model(
            int(ai_case_count),
            int(ai_dataset_seed),
            str(training_scope),
        )

if st.session_state.ai_bundle is None:
    st.info(
        "AI 진단을 사용하려면 위 버튼을 눌러 모델을 한 번 학습하세요. "
        "현재 Case 난이도와 무관하게 기본값인 '전체 난이도 혼합' 학습을 권장합니다."
    )
else:
    ai_bundle = st.session_state.ai_bundle

    ai_metric_col1, ai_metric_col2, ai_metric_col3, ai_metric_col4, ai_metric_col5 = (
        st.columns(5)
    )

    ai_metric_col1.metric(
        "생성 Case",
        f"{ai_bundle['case_count']:,}",
    )
    ai_metric_col2.metric(
        "시험 정확도",
        f"{ai_bundle['accuracy'] * 100:.1f}%",
    )
    ai_metric_col3.metric(
        "Macro F1",
        f"{ai_bundle['macro_f1']:.3f}",
    )
    ai_metric_col4.metric(
        "시험 Case",
        f"{ai_bundle['test_cases']:,}",
    )
    ai_metric_col5.metric(
        "학습 범위",
        (
            "전체 혼합"
            if ai_bundle[
                "training_scope"
            ] == "mixed"
            else DIFFICULTY_LABELS[
                ai_bundle[
                    "training_scope"
                ]
            ]
        ),
    )

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:
        st.markdown("#### 혼동행렬")

        matrix = ai_bundle[
            "confusion_matrix"
        ]

        confusion_rows = []

        for actual_index, actual_key in enumerate(
            AI_CLASS_ORDER
        ):
            for predicted_index, predicted_key in enumerate(
                AI_CLASS_ORDER
            ):
                confusion_rows.append({
                    "실제": AI_LABELS[
                        actual_key
                    ],
                    "예측": AI_LABELS[
                        predicted_key
                    ],
                    "건수": int(
                        matrix[
                            actual_index,
                            predicted_index,
                        ]
                    ),
                })

        confusion_df = pd.DataFrame(
            confusion_rows
        )

        heatmap = (
            alt.Chart(confusion_df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "예측:N",
                    title="AI 예측",
                    sort=[
                        AI_LABELS[key]
                        for key in AI_CLASS_ORDER
                    ],
                ),
                y=alt.Y(
                    "실제:N",
                    title="실제 정답",
                    sort=[
                        AI_LABELS[key]
                        for key in AI_CLASS_ORDER
                    ],
                ),
                color=alt.Color(
                    "건수:Q",
                    title="건수",
                ),
                tooltip=[
                    "실제:N",
                    "예측:N",
                    "건수:Q",
                ],
            )
            .properties(height=310)
        )

        heatmap_text = (
            alt.Chart(confusion_df)
            .mark_text()
            .encode(
                x=alt.X(
                    "예측:N",
                    sort=[
                        AI_LABELS[key]
                        for key in AI_CLASS_ORDER
                    ],
                ),
                y=alt.Y(
                    "실제:N",
                    sort=[
                        AI_LABELS[key]
                        for key in AI_CLASS_ORDER
                    ],
                ),
                text="건수:Q",
                color=alt.condition(
                    "datum.건수 > 500",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        st.altair_chart(
            heatmap + heatmap_text,
            width="stretch",
        )

    with performance_col2:
        st.markdown("#### 주요 판단 변수")

        top_importance = (
            ai_bundle[
                "feature_importance"
            ]
            .head(12)
            .sort_values(
                "importance",
                ascending=True,
            )
        )

        importance_chart = (
            alt.Chart(top_importance)
            .mark_bar()
            .encode(
                x=alt.X(
                    "importance:Q",
                    title="중요도",
                ),
                y=alt.Y(
                    "feature_label:N",
                    title=None,
                    sort=None,
                ),
                tooltip=[
                    alt.Tooltip(
                        "feature_label:N",
                        title="변수",
                    ),
                    alt.Tooltip(
                        "importance:Q",
                        title="중요도",
                        format=".4f",
                    ),
                ],
            )
            .properties(height=310)
        )

        st.altair_chart(
            importance_chart,
            width="stretch",
        )

    st.markdown("#### 난이도별 성능")

    difficulty_metrics = (
        ai_bundle[
            "difficulty_metrics"
        ].copy()
    )

    percentage_columns = [
        "Lot 분류 정확도",
        "이상 Lot 정확 일치",
        "±1 Lot 이내",
        "원인 정확도",
        "Top-2 원인 적중률",
        "정상 Case 오탐률",
    ]

    for column in percentage_columns:
        if column in difficulty_metrics:
            difficulty_metrics[
                column
            ] = (
                difficulty_metrics[
                    column
                ]
                * 100
            ).round(1)

    if "Macro F1" in difficulty_metrics:
        difficulty_metrics[
            "Macro F1"
        ] = difficulty_metrics[
            "Macro F1"
        ].round(3)

    st.dataframe(
        difficulty_metrics,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "정확 일치는 최초 이상 Lot을 정확히 맞힌 비율이며, "
        "±1 Lot은 한 Lot 이내로 탐지한 비율입니다. "
        "전문가 난이도에서는 판단 유보와 오탐이 늘어날 수 있습니다."
    )

    with st.expander(
        "원인별 분류 성능과 학습 데이터 확인",
        expanded=False,
    ):
        report_df = (
            ai_bundle[
                "classification_report"
            ].copy()
        )

        numeric_report_columns = [
            column
            for column in [
                "정밀도",
                "재현율",
                "F1",
            ]
            if column in report_df.columns
        ]

        report_df[
            numeric_report_columns
        ] = report_df[
            numeric_report_columns
        ].round(3)

        if "표본 수" in report_df.columns:
            report_df["표본 수"] = (
                report_df[
                    "표본 수"
                ]
                .fillna(0)
                .round()
                .astype(int)
            )

        st.dataframe(
            report_df,
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            f"- 학습 Case: **{ai_bundle['train_cases']:,}개**"
            f" / 시험 Case: **{ai_bundle['test_cases']:,}개**"
            f" / 전체 Lot 표본: **{len(ai_bundle['dataset']):,}개**"
        )

        st.download_button(
            "AI 학습용 Lot 특징 CSV 다운로드",
            data=csv_bytes(
                ai_bundle["dataset"]
            ),
            file_name=(
                "sputter_ai_training_features_"
                f"{ai_bundle['case_count']}_cases_"
                f"{ai_bundle['training_scope']}.csv"
            ),
            mime="text/csv",
        )

st.divider()
st.subheader("6. 이상 진단")

with st.form("diagnosis_form"):
    guess_col1, guess_col2, guess_col3 = st.columns(3)

    with guess_col1:
        guessed_lot = st.number_input(
            "최초 이상 Lot",
            min_value=1,
            max_value=max(
                st.session_state.current_lot,
                1,
            ),
            value=1,
            step=1,
        )

    with guess_col2:
        guessed_cause = st.selectbox(
            "예상 1순위 원인",
            list(CAUSES.values()),
        )

    with guess_col3:
        guessed_secondary_cause = st.selectbox(
            "예상 2순위 원인",
            ["없음"]
            + list(CAUSES.values()),
        )

    evidence = st.text_area(
        "판단 근거",
        placeholder=(
            "Actual 공정 변수, 증착률, 두께, "
            "균일도, 예상 면저항 변화를 연결해 작성하세요."
        ),
        height=110,
    )

    additional_check = st.text_area(
        "추가 확인 항목",
        placeholder=(
            "예: 장비 로그, MFC 교정 이력, "
            "압력 센서, 타겟 사용 이력"
        ),
        height=90,
    )

    corrective_action = st.text_area(
        "조치 방안",
        placeholder=(
            "예: 장비 Hold, 관련 계통 점검, "
            "Monitor Wafer 재검증"
        ),
        height=90,
    )

    submit_diagnosis = st.form_submit_button(
        "진단 제출 및 정답 비교",
        use_container_width=True,
    )

if submit_diagnosis:
    errors = []

    if st.session_state.cause_key is None:
        errors.append(
            "먼저 랜덤 이상을 예약해야 합니다."
        )

    if (
        st.session_state.fault_start_lot is not None
        and st.session_state.current_lot
        < st.session_state.fault_start_lot
    ):
        errors.append(
            "이상이 데이터에 나타날 때까지 "
            "Lot을 더 생산해야 합니다."
        )

    if len(evidence.strip()) < 30:
        errors.append(
            "판단 근거를 30자 이상 작성하세요."
        )

    if len(additional_check.strip()) < 15:
        errors.append(
            "추가 확인 항목을 15자 이상 작성하세요."
        )

    if len(corrective_action.strip()) < 15:
        errors.append(
            "조치 방안을 15자 이상 작성하세요."
        )

    if errors:
        for error in errors:
            st.error(error)

    else:
        actual_key = (
            st.session_state.cause_key
        )
        actual_cause = CAUSES[actual_key]
        actual_secondary_key = (
            st.session_state.secondary_cause_key
        )
        actual_secondary_cause = (
            CAUSES[
                actual_secondary_key
            ]
            if actual_secondary_key
            is not None
            else "없음"
        )
        actual_lot = int(
            st.session_state.fault_start_lot
        )
        automatic_lot = automatic_abnormal_lot(
            summary_df
        )

        ai_diagnosis = ai_diagnose_current_case(
            st.session_state.ai_bundle,
            summary_df,
            active_material,
            active_difficulty,
        )

        lot_correct = (
            int(guessed_lot) == actual_lot
        )
        cause_correct = (
            guessed_cause == actual_cause
        )
        secondary_correct = (
            guessed_secondary_cause
            == actual_secondary_cause
        )

        result = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "material": active_material,
            "difficulty": (
                DIFFICULTY_LABELS[
                    active_difficulty
                ]
            ),
            "case_seed": (
                st.session_state.case_seed
            ),
            "actual_fault_lot": actual_lot,
            "guessed_fault_lot": int(
                guessed_lot
            ),
            "lot_correct": lot_correct,
            "actual_cause": actual_cause,
            "guessed_cause": guessed_cause,
            "cause_correct": cause_correct,
            "actual_secondary_cause": (
                actual_secondary_cause
            ),
            "guessed_secondary_cause": (
                guessed_secondary_cause
            ),
            "secondary_correct": (
                secondary_correct
            ),
            "both_correct": (
                lot_correct and cause_correct
            ),
            "auto_detected_lot": automatic_lot,
            "ai_fault_lot": (
                ai_diagnosis["fault_lot"]
                if ai_diagnosis is not None
                else None
            ),
            "ai_cause": (
                ai_diagnosis["cause_label"]
                if ai_diagnosis is not None
                else "AI 모델 미학습"
            ),
            "ai_confidence": (
                ai_diagnosis["confidence"]
                if ai_diagnosis is not None
                else None
            ),
            "ai_confidence_label": (
                ai_diagnosis[
                    "confidence_label"
                ]
                if ai_diagnosis is not None
                else "미학습"
            ),
            "ai_top_causes": (
                ai_diagnosis[
                    "top_causes"
                ]
                if ai_diagnosis is not None
                else []
            ),
            "ai_composite_possible": (
                ai_diagnosis[
                    "composite_possible"
                ]
                if ai_diagnosis is not None
                else False
            ),
            "ai_prediction_table": (
                ai_diagnosis[
                    "prediction_table"
                ]
                if ai_diagnosis is not None
                else None
            ),
            "evidence": evidence.strip(),
            "additional_check": (
                additional_check.strip()
            ),
            "corrective_action": (
                corrective_action.strip()
            ),
        }

        st.session_state.history.append(
            result
        )
        st.session_state.last_result = (
            result
        )

if st.session_state.last_result:
    result = st.session_state.last_result

    actual_key = next(
        key
        for key, value in CAUSES.items()
        if value == result["actual_cause"]
    )

    st.subheader("진단 결과")

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )

    result_col1.metric(
        "이상 Lot",
        (
            "일치"
            if result["lot_correct"]
            else "불일치"
        ),
        f"정답 Lot {result['actual_fault_lot']}",
    )

    result_col2.metric(
        "이상 원인",
        (
            "일치"
            if result["cause_correct"]
            else "불일치"
        ),
        result["actual_cause"],
    )

    result_col3.metric(
        "통계 기반 자동 탐지",
        (
            f"Lot {result['auto_detected_lot']}"
            if result["auto_detected_lot"]
            is not None
            else "탐지 실패"
        ),
    )

    result_col4.metric(
        "AI 자동 진단",
        (
            f"Lot {result['ai_fault_lot']}"
            if result.get("ai_fault_lot")
            is not None
            else "모델 미학습/탐지 실패"
        ),
        result.get(
            "ai_cause",
            "",
        ),
    )

    if (
        result.get(
            "ai_confidence"
        )
        is not None
    ):
        top_causes = result.get(
            "ai_top_causes",
            [],
        )

        if top_causes:
            first_cause = top_causes[0]
            second_cause = (
                top_causes[1]
                if len(top_causes) > 1
                else None
            )

            ai_text = (
                f"**AI 예측:** Lot "
                f"{result['ai_fault_lot']}부터 "
                f"1순위 {first_cause['cause_label']} "
                f"{first_cause['probability'] * 100:.1f}%"
            )

            if second_cause:
                ai_text += (
                    f" · 2순위 "
                    f"{second_cause['cause_label']} "
                    f"{second_cause['probability'] * 100:.1f}%"
                )

            ai_text += (
                f" · 신뢰도 "
                f"{result.get('ai_confidence_label', '미정')}"
            )

            st.markdown(ai_text)

            if result.get(
                "ai_composite_possible",
                False,
            ):
                st.warning(
                    "AI가 두 원인의 확률을 비슷하게 평가해 "
                    "복합 이상 가능성을 표시했습니다."
                )
        else:
            st.markdown(
                "**AI 예측:** 이상 신호가 기준에 미달해 판단을 유보했습니다."
            )

        with st.expander(
            "AI Lot별 예측 결과",
            expanded=False,
        ):
            ai_prediction_table = (
                result[
                    "ai_prediction_table"
                ].copy()
            )

            for probability_column in [
                "이상 확률",
                "1순위 확률",
                "2순위 확률",
            ]:
                if (
                    probability_column
                    in ai_prediction_table
                ):
                    ai_prediction_table[
                        probability_column
                    ] = (
                        ai_prediction_table[
                            probability_column
                        ]
                        * 100
                    ).round(1)

            st.dataframe(
                ai_prediction_table,
                hide_index=True,
                width="stretch",
            )

    st.markdown(
        f"**재료·난이도:** "
        f"{result['material']} · "
        f"{result.get('difficulty', '')}"
    )
    st.markdown(
        f"**2순위 원인 제출/정답:** "
        f"{result.get('guessed_secondary_cause', '없음')} / "
        f"{result.get('actual_secondary_cause', '없음')}"
    )
    st.markdown(
        f"**정답 패턴:** "
        f"{CAUSE_EXPLANATIONS[actual_key]}"
    )
    st.markdown(
        f"**작성한 판단 근거:** "
        f"{result['evidence']}"
    )
    st.markdown(
        f"**추가 확인 항목:** "
        f"{result['additional_check']}"
    )
    st.markdown(
        f"**조치 방안:** "
        f"{result['corrective_action']}"
    )

st.divider()
st.subheader("7. 누적 진단 이력")

if st.session_state.history:
    history_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in [
                    "ai_prediction_table",
                    "ai_top_causes",
                ]
            }
            for record in st.session_state.history
        ]
    )

    total_cases = len(history_df)
    lot_accuracy = (
        history_df["lot_correct"].mean()
        * 100
    )
    cause_accuracy = (
        history_df["cause_correct"].mean()
        * 100
    )
    both_accuracy = (
        history_df["both_correct"].mean()
        * 100
    )

    history_col1, history_col2, history_col3, history_col4 = (
        st.columns(4)
    )

    history_col1.metric(
        "완료 Case",
        total_cases,
    )
    history_col2.metric(
        "이상 Lot 정확도",
        f"{lot_accuracy:.1f}%",
    )
    history_col3.metric(
        "원인 정확도",
        f"{cause_accuracy:.1f}%",
    )
    history_col4.metric(
        "동시 적중률",
        f"{both_accuracy:.1f}%",
    )

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "진단 이력 CSV 다운로드",
        data=csv_bytes(history_df),
        file_name="diagnosis_history.csv",
        mime="text/csv",
    )

else:
    st.caption(
        "아직 제출된 진단 이력이 없습니다."
    )
