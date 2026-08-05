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

    /* 공정 현황판: PC에서는 4열, 모바일에서는 2열 */
    .process-status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.25rem 0 0.4rem 0;
    }

    .process-status-card {
        min-width: 0;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(127, 127, 127, 0.24);
        border-radius: 0.75rem;
        background: rgba(127, 127, 127, 0.07);
    }

    .process-status-label {
        margin-bottom: 0.2rem;
        font-size: 0.82rem;
        opacity: 0.72;
        white-space: nowrap;
    }

    .process-status-value {
        overflow: hidden;
        font-size: 1.65rem;
        font-weight: 650;
        line-height: 1.2;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .problem-number-line {
        margin: 0.1rem 0 0.85rem 0;
        font-size: 0.82rem;
        text-align: right;
        opacity: 0.7;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 0.7rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.45rem;
        }

        h1 {
            font-size: 1.75rem !important;
            line-height: 1.25 !important;
        }

        h2 {
            font-size: 1.45rem !important;
            line-height: 1.3 !important;
        }

        h3 {
            font-size: 1.2rem !important;
            line-height: 1.35 !important;
        }

        .process-status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.1rem;
        }

        .process-status-card {
            padding: 0.58rem 0.68rem;
            border-radius: 0.6rem;
        }

        .process-status-label {
            margin-bottom: 0.1rem;
            font-size: 0.7rem;
        }

        .process-status-value {
            font-size: 1.18rem;
        }

        .problem-number-line {
            margin-bottom: 0.55rem;
            font-size: 0.72rem;
            text-align: left;
        }

        div[data-testid="stAlert"] {
            padding: 0.7rem 0.8rem;
        }

        div[data-testid="stDataFrame"] {
            font-size: 0.82rem;
        }

        button[kind="primary"],
        button[kind="secondary"] {
            min-height: 2.55rem;
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

APP_VERSION = "v2.4.5"
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
        "description": "배선·전극용 금속 박막의 기초 학습 문제",
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
        "description": "낮은 비저항을 갖는 금속 배선 비교 문제",
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
        "description": "접착층·배리어층 재료 비교 문제",
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
        "description": "배리어층 재료 비교 문제",
    },
}

# 박막에서는 표면·입계 산란 등에 의해 벌크보다 비저항이 커질 수 있으므로
# 교육용 단순화 계수를 적용합니다. 실제 측정값을 의미하지 않습니다.
THIN_FILM_RESISTIVITY_FACTOR = 1.25

CAUSES = {
    "power_drift": "전원 출력 제어 이상",
    "flow_drop": "Ar 유량 제어 이상",
    "pressure_rise": "챔버 압력 제어 이상",
    "target_wear": "타겟 상태 변화",
}

CAUSE_EXPLANATIONS = {
    "power_drift": (
        "실제 전원 출력이 설정값보다 높아지거나 낮아지면서 "
        "증착률·두께·면저항이 함께 변합니다."
    ),
    "flow_drop": (
        "실제 Ar 유량이 설정값에서 벗어나 챔버 압력과 플라즈마 상태가 변하고, "
        "증착률·두께·균일도에 영향을 줍니다."
    ),
    "pressure_rise": (
        "실제 챔버 압력이 설정값보다 높아지거나 낮아지면서 "
        "증착률과 두께가 변하고 균일도와 면저항이 악화될 수 있습니다."
    ),
    "target_wear": (
        "타겟 침식이나 보상 제어로 증착률·두께·균일도·면저항이 변합니다. "
        "두께가 감소하거나 유지되거나 과보상으로 증가할 수 있습니다."
    ),
}


CAUSE_ANSWER_GUIDES = {
    "power_drift": {
        "evidence": (
            "실제 전원 출력이 설정값에서 벗어나고, 같은 시점부터 "
            "증착률·평균 두께·예상 면저항이 함께 변하는지 확인합니다."
        ),
        "additional_check": (
            "전원 공급 장치 출력 로그, 전력 센서 교정 이력, "
            "매칭 네트워크와 케이블·전극의 접촉 상태를 확인합니다."
        ),
        "corrective_action": (
            "해당 Lot을 보류하고 전원 공급·측정 계통을 점검합니다. "
            "설정값과 실제 출력이 일치하는지 확인한 뒤 모니터 웨이퍼를 생산해 "
            "두께와 면저항이 정상 범위로 돌아왔는지 검증합니다."
        ),
    },
    "flow_drop": {
        "evidence": (
            "실제 Ar 유량이 설정값에서 벗어나고, 챔버 압력과 증착률 및 "
            "평균 두께·균일도 지표가 같은 시점부터 변하는지 확인합니다."
        ),
        "additional_check": (
            "질량 유량 제어기 교정 상태, 가스 공급 압력, 밸브 동작 로그, "
            "가스 배관의 누설이나 막힘 여부를 확인합니다."
        ),
        "corrective_action": (
            "해당 Lot을 보류하고 유량 제어기와 가스 공급 계통을 점검합니다. "
            "유량과 압력이 안정된 뒤 모니터 웨이퍼를 생산해 증착률·두께·균일도를 확인합니다."
        ),
    },
    "pressure_rise": {
        "evidence": (
            "실제 챔버 압력이 설정값에서 벗어나고, 같은 시점부터 "
            "증착률·평균 두께·균일도 지표와 예상 면저항이 함께 변하는지 확인합니다."
        ),
        "additional_check": (
            "압력 센서 교정 이력, 스로틀 밸브 작동 로그, 진공 펌프 상태, "
            "챔버와 배관의 누설 여부를 확인합니다."
        ),
        "corrective_action": (
            "해당 Lot을 보류하고 압력 측정·제어 계통과 진공 계통을 점검합니다. "
            "기저 압력과 공정 압력이 안정된 뒤 모니터 웨이퍼로 공정 복귀 여부를 검증합니다."
        ),
    },
    "target_wear": {
        "evidence": (
            "전원 출력·Ar 유량·챔버 압력은 설정값에 가깝지만, 타겟 누적 사용 시간과 함께 "
            "증착률·평균 두께·균일도 지표 또는 예상 면저항이 변하는지 확인합니다."
        ),
        "additional_check": (
            "타겟 누적 사용 시간과 침식 형상, 타겟 냉각·고정 상태, "
            "사전 스퍼터링 이력과 출력 보상 설정을 확인합니다."
        ),
        "corrective_action": (
            "타겟 상태를 점검해 필요하면 교체하거나 출력 보상 조건을 다시 설정합니다. "
            "챔버 컨디셔닝 후 모니터 웨이퍼를 생산해 증착률·두께·균일도와 면저항을 검증합니다."
        ),
    },
}

UNKNOWN_CAUSES = {
    "time_overrun": "증착 시간 과다",
    "metrology_bias": "두께 측정 장비 편향",
    "substrate_temperature": "기판 온도 이상",
    "shutter_timing": "셔터 동작 이상",
}

AI_ABNORMAL_THRESHOLD = 0.58



DIFFICULTIES = {
    "easy": {
        "label": "쉬움",
        "description": "정상 기준 Lot 2개 제공 · 원인 1개 · 강한 신호",
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
        "description": "정상 기준 Lot 2개 제공 · 신호 중첩 · 보통 노이즈",
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
        "description": "초기 정상 상태 미보장 · 약한 이상 · 일부 웨이퍼 이상",
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
        "description": "Lot 1부터 이상 가능 · 복합 이상 · 높은 노이즈",
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

REFERENCE_LOT_DIFFICULTIES = {
    "easy",
    "normal",
}


def has_guaranteed_reference_lots(
    difficulty,
):
    """쉬움·보통에서는 Lot 1~2를 정상 기준으로 보장합니다."""
    return difficulty in REFERENCE_LOT_DIFFICULTIES

FAULT_START_RANGES = {
    "easy": (3, 5),
    "normal": (3, 7),
    "hard": (1, 6),
    "expert": (1, 8),
}


def choose_fault_start_lot(
    rng,
    difficulty,
):
    minimum_lot, maximum_lot = FAULT_START_RANGES[difficulty]
    return int(rng.integers(minimum_lot, maximum_lot + 1))


def choose_fault_direction(
    rng,
    cause,
):
    """원인에 맞는 두께 상승·하락 방향과 내부 부호를 정합니다."""
    if cause == "target_wear":
        direction = str(rng.choice(["up", "down"], p=[0.35, 0.65]))
    else:
        direction = str(rng.choice(["up", "down"], p=[0.50, 0.50]))

    if cause == "pressure_rise":
        sign = -1 if direction == "up" else 1
    else:
        sign = 1 if direction == "up" else -1

    return direction, sign


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
    "lot_number",
    "mean_thickness_nm",
    "std_thickness_nm",
    "mean_uniformity_pct",
    "mean_sheet_resistance_ohm_sq",
    "oos_ratio",
    "thickness_dev_target_nm",
    "thickness_abs_dev_target_nm",
    "uniformity_over_limit_pctp",
    "resistance_dev_reference_pct",
    "power_dev_set_pct",
    "flow_dev_set_pct",
    "pressure_dev_set_pct",
    "rate_dev_set_pct",
    "power_abs_dev_set_pct",
    "flow_abs_dev_set_pct",
    "pressure_abs_dev_set_pct",
    "rate_abs_dev_set_pct",
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

DIFFICULTY_SHORTCUT_FEATURES = [
    "difficulty_easy",
    "difficulty_normal",
    "difficulty_hard",
    "difficulty_expert",
]

AI_FEATURE_LABELS = {
    "material_Al": "재료: Al",
    "material_Cu": "재료: Cu",
    "material_Ti": "재료: Ti",
    "material_Ta": "재료: Ta",
    "lot_number": "Lot 번호",
    "mean_thickness_nm": "평균 두께",
    "std_thickness_nm": "두께 표준편차",
    "mean_uniformity_pct": "평균 균일도",
    "mean_sheet_resistance_ohm_sq": "예상 면저항",
    "oos_ratio": "두께 이탈 비율",
    "thickness_dev_target_nm": "목표 두께 대비 편차",
    "thickness_abs_dev_target_nm": "목표 두께 대비 절대 편차",
    "uniformity_over_limit_pctp": "균일도 기준 초과량",
    "resistance_dev_reference_pct": "기준 면저항 대비 편차",
    "power_dev_set_pct": "전원 출력 설정값 편차",
    "flow_dev_set_pct": "Ar 유량 설정값 편차",
    "pressure_dev_set_pct": "챔버 압력 설정값 편차",
    "rate_dev_set_pct": "증착률 기준 편차",
    "power_abs_dev_set_pct": "전원 출력 절대 편차",
    "flow_abs_dev_set_pct": "Ar 유량 절대 편차",
    "pressure_abs_dev_set_pct": "챔버 압력 절대 편차",
    "rate_abs_dev_set_pct": "증착률 절대 편차",
    "thickness_delta_baseline": "초기 두 Lot 대비 두께 변화",
    "std_delta_baseline": "초기 두 Lot 대비 산포 변화",
    "uniformity_delta_baseline": "초기 두 Lot 대비 균일도 변화",
    "resistance_delta_baseline_pct": "초기 두 Lot 대비 면저항 변화",
    "power_delta_baseline_pct": "초기 두 Lot 대비 전원 출력 변화",
    "flow_delta_baseline_pct": "초기 두 Lot 대비 Ar 유량 변화",
    "pressure_delta_baseline_pct": "초기 두 Lot 대비 압력 변화",
    "rate_delta_baseline_pct": "초기 두 Lot 대비 증착률 변화",
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


RAW_DISPLAY_COLUMNS = {
    "material": "증착 재료",
    "lot": "Lot",
    "wafer": "웨이퍼",
    "lot_section": "Lot 구분",
    "power_set_w": "전원 출력 설정값 (W)",
    "power_actual_w": "실제 전원 출력 (W)",
    "ar_flow_set_sccm": "Ar 유량 설정값 (sccm)",
    "ar_flow_actual_sccm": "실제 Ar 유량 (sccm)",
    "pressure_set_mtorr": "챔버 압력 설정값 (mTorr)",
    "pressure_actual_mtorr": "실제 챔버 압력 (mTorr)",
    "time_set_s": "증착 시간 설정값 (s)",
    "time_actual_s": "실제 증착 시간 (s)",
    "target_usage_h": "타겟 누적 사용 시간 (h)",
    "deposition_rate_nm_s": "증착률 (nm/s)",
    "thickness_nm": "박막 두께 (nm)",
    "uniformity_pct": "균일도 지표 (%)",
    "estimated_sheet_resistance_ohm_sq": "예상 면저항 (Ω/□)",
}

SUMMARY_DISPLAY_COLUMNS = {
    "material": "증착 재료",
    "lot": "Lot",
    "lot_section": "Lot 구분",
    "mean_thickness_nm": "평균 두께 (nm)",
    "std_thickness_nm": "두께 표준편차 (nm)",
    "mean_uniformity_pct": "평균 균일도 지표 (%)",
    "mean_sheet_resistance_ohm_sq": "평균 예상 면저항 (Ω/□)",
    "oos_count": "두께 기준 이탈 웨이퍼 수",
    "mean_power_w": "평균 실제 전원 출력 (W)",
    "mean_ar_flow_sccm": "평균 실제 Ar 유량 (sccm)",
    "mean_pressure_mtorr": "평균 실제 챔버 압력 (mTorr)",
    "mean_rate_nm_s": "평균 증착률 (nm/s)",
}


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

    if has_fault:
        thickness_direction, fault_sign = choose_fault_direction(
            rng,
            primary_cause,
        )
    else:
        thickness_direction = "normal"
        fault_sign = 1

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
        "fault_sign": int(fault_sign),
        "thickness_direction": thickness_direction,
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
    effective_progress = max(progress, 0) * severity
    observable_fraction = 1.0 - compensation

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
            * (0.40 + 0.60 * compensation)
        )
        true_pressure[mask] += (
            sign
            * pressure_setpoint
            * 0.006
            * effective_progress
            * overlap_strength
        )
        hidden_uniformity_effect[mask] += (
            0.08
            * effective_progress
            * overlap_strength
        )

    elif cause == "flow_drop":
        # 유량 제어 이상은 상승과 하락을 모두 허용합니다.
        true_flow[mask] += (
            sign
            * 0.85
            * effective_progress
            * observable_fraction
        )
        true_pressure[mask] += (
            sign
            * 0.08
            * effective_progress
            * observable_fraction
        )
        hidden_rate_effect[mask] += (
            sign
            * base_rate
            * 0.004
            * effective_progress
            * (0.40 + 0.60 * compensation)
        )
        true_power[mask] -= (
            sign
            * power_setpoint
            * 0.0025
            * effective_progress
            * overlap_strength
        )
        hidden_uniformity_effect[mask] += (
            0.12
            * effective_progress
            * (0.4 + overlap_strength)
        )

    elif cause == "pressure_rise":
        # 압력 제어 이상도 설정값보다 높거나 낮은 두 방향을 생성합니다.
        true_pressure[mask] += (
            sign
            * 0.25
            * effective_progress
            * observable_fraction
        )
        hidden_rate_effect[mask] -= (
            sign
            * base_rate
            * 0.0035
            * effective_progress
        )
        hidden_uniformity_effect[mask] += (
            0.34
            * effective_progress
        )
        hidden_resistivity_effect[mask] += (
            0.012
            * effective_progress
        )
        true_flow[mask] -= (
            sign
            * 0.15
            * effective_progress
            * overlap_strength
        )

    elif cause == "target_wear":
        # 음수 방향은 일반적인 증착률 저하, 양수 방향은 보상 제어의 과보상을 뜻합니다.
        if sign < 0:
            hidden_rate_effect[mask] -= (
                base_rate
                * 0.0075
                * effective_progress
            )
            true_power[mask] += (
                power_setpoint
                * 0.0035
                * effective_progress
                * overlap_strength
            )
        else:
            true_power[mask] += (
                power_setpoint
                * 0.0120
                * effective_progress
                * (0.5 + 0.5 * observable_fraction)
            )
            hidden_rate_effect[mask] -= (
                base_rate
                * 0.0008
                * effective_progress
            )

        hidden_uniformity_effect[mask] += (
            0.24
            * effective_progress
        )
        hidden_resistivity_effect[mask] += (
            0.012
            * effective_progress
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
    material=None,
    difficulty=None,
    seed=None,
):
    selected_material = (
        material
        if material is not None
        else st.session_state.get(
            "active_material",
            "Al",
        )
    )
    selected_difficulty = (
        difficulty
        if difficulty is not None
        else st.session_state.get(
            "active_difficulty",
            "easy",
        )
    )
    selected_seed = int(
        seed
        if seed is not None
        else st.session_state.get(
            "case_seed",
            20260805,
        )
    )

    st.session_state.active_material = (
        selected_material
    )
    st.session_state.active_difficulty = (
        selected_difficulty
    )
    st.session_state.case_seed = (
        selected_seed
    )
    st.session_state.rng = (
        np.random.default_rng(
            selected_seed
        )
    )
    st.session_state.current_lot = 0
    st.session_state.raw_data = (
        pd.DataFrame()
    )
    st.session_state.last_result = None
    st.session_state.case_ready = True

    rng = st.session_state.rng

    fault_start_lot = choose_fault_start_lot(
        rng,
        selected_difficulty,
    )
    event_min_lot = (
        3
        if has_guaranteed_reference_lots(selected_difficulty)
        else 1
    )

    event_max_lot = max(
        fault_start_lot + 8,
        10,
    )

    profile = build_case_profile(
        rng=rng,
        difficulty=selected_difficulty,
        fault_start_lot=fault_start_lot,
        event_min_lot=event_min_lot,
        event_max_lot=event_max_lot,
        has_fault=True,
    )

    st.session_state.case_profile = (
        profile
    )
    st.session_state.cause_key = (
        profile[
            "primary_cause"
        ]
    )
    st.session_state.secondary_cause_key = (
        profile[
            "secondary_cause"
        ]
    )
    st.session_state.fault_start_lot = (
        profile[
            "fault_start_lot"
        ]
    )

    if has_guaranteed_reference_lots(
        selected_difficulty
    ):
        for _ in range(2):
            produce_lot()

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
    material = str(rng.choice(list(MATERIALS.keys())))
    lot_count = int(rng.integers(8, 14))
    has_fault = bool(rng.random() >= 0.15)

    if has_fault:
        fault_start_lot = min(
            choose_fault_start_lot(rng, difficulty),
            lot_count - 2,
        )
    else:
        fault_start_lot = None

    profile = build_case_profile(
        rng=rng,
        difficulty=difficulty,
        fault_start_lot=(
            fault_start_lot
            if fault_start_lot is not None
            else 1
        ),
        event_min_lot=(
            3
            if has_guaranteed_reference_lots(difficulty)
            else 1
        ),
        event_max_lot=lot_count - 1,
        has_fault=has_fault,
    )

    summary_rows = []

    for lot in range(1, lot_count + 1):
        measurements = generate_lot_measurements(
            rng=rng,
            material=material,
            lot=lot,
            profile=profile,
        )

        target = (
            profile["primary_cause"]
            if (
                has_fault
                and fault_start_lot is not None
                and lot >= fault_start_lot
            )
            else "normal"
        )

        summary_rows.append(
            lot_measurements_to_summary(
                case_id=case_id,
                material=material,
                difficulty=difficulty,
                lot=lot,
                measurements=measurements,
                target=target,
                primary_cause=profile["primary_cause"],
                secondary_cause=profile["secondary_cause"],
                fault_start_lot=fault_start_lot,
                severity=profile["severity"],
            )
        )

    return pd.DataFrame(summary_rows)

def build_ai_features(
    summary,
    material,
    difficulty=None,
):
    if summary.empty:
        return pd.DataFrame(columns=AI_FEATURE_COLUMNS)

    work = (
        summary
        .sort_values("lot")
        .reset_index(drop=True)
        .copy()
    )
    recipe = get_recipe(material)
    resistance_reference = reference_sheet_resistance(material)

    baseline_rows = work.iloc[: min(2, len(work))]
    baseline = {
        "thickness": float(baseline_rows["mean_thickness_nm"].mean()),
        "std": float(baseline_rows["std_thickness_nm"].mean()),
        "uniformity": float(baseline_rows["mean_uniformity_pct"].mean()),
        "resistance": float(
            baseline_rows["mean_sheet_resistance_ohm_sq"].mean()
        ),
        "power": float(baseline_rows["mean_power_w"].mean()),
        "flow": float(baseline_rows["mean_ar_flow_sccm"].mean()),
        "pressure": float(baseline_rows["mean_pressure_mtorr"].mean()),
        "rate": float(baseline_rows["mean_rate_nm_s"].mean()),
    }

    rows = []

    for index, row in work.iterrows():
        previous = work.iloc[index - 1] if index > 0 else row

        power_dev = (
            (row["mean_power_w"] - recipe["power_w"])
            / recipe["power_w"]
            * 100
        )
        flow_dev = (
            (row["mean_ar_flow_sccm"] - recipe["ar_flow_sccm"])
            / recipe["ar_flow_sccm"]
            * 100
        )
        pressure_dev = (
            (row["mean_pressure_mtorr"] - recipe["pressure_mtorr"])
            / recipe["pressure_mtorr"]
            * 100
        )
        rate_dev = (
            (row["mean_rate_nm_s"] - recipe["base_rate_nm_s"])
            / recipe["base_rate_nm_s"]
            * 100
        )
        thickness_dev = row["mean_thickness_nm"] - TARGET_THICKNESS

        feature_row = {
            "material_Al": int(material == "Al"),
            "material_Cu": int(material == "Cu"),
            "material_Ti": int(material == "Ti"),
            "material_Ta": int(material == "Ta"),
            "lot_number": float(row["lot"]),
            "mean_thickness_nm": float(row["mean_thickness_nm"]),
            "std_thickness_nm": float(row["std_thickness_nm"]),
            "mean_uniformity_pct": float(row["mean_uniformity_pct"]),
            "mean_sheet_resistance_ohm_sq": float(
                row["mean_sheet_resistance_ohm_sq"]
            ),
            "oos_ratio": float(row["oos_count"] / WAFERS_PER_LOT),
            "thickness_dev_target_nm": float(thickness_dev),
            "thickness_abs_dev_target_nm": float(abs(thickness_dev)),
            "uniformity_over_limit_pctp": float(
                max(row["mean_uniformity_pct"] - UNIFORMITY_LIMIT, 0.0)
            ),
            "resistance_dev_reference_pct": float(
                (row["mean_sheet_resistance_ohm_sq"] - resistance_reference)
                / max(abs(resistance_reference), 1e-9)
                * 100
            ),
            "power_dev_set_pct": float(power_dev),
            "flow_dev_set_pct": float(flow_dev),
            "pressure_dev_set_pct": float(pressure_dev),
            "rate_dev_set_pct": float(rate_dev),
            "power_abs_dev_set_pct": float(abs(power_dev)),
            "flow_abs_dev_set_pct": float(abs(flow_dev)),
            "pressure_abs_dev_set_pct": float(abs(pressure_dev)),
            "rate_abs_dev_set_pct": float(abs(rate_dev)),
            "thickness_delta_baseline": float(
                row["mean_thickness_nm"] - baseline["thickness"]
            ),
            "std_delta_baseline": float(
                row["std_thickness_nm"] - baseline["std"]
            ),
            "uniformity_delta_baseline": float(
                row["mean_uniformity_pct"] - baseline["uniformity"]
            ),
            "resistance_delta_baseline_pct": float(
                (row["mean_sheet_resistance_ohm_sq"] - baseline["resistance"])
                / max(abs(baseline["resistance"]), 1e-9)
                * 100
            ),
            "power_delta_baseline_pct": float(
                (row["mean_power_w"] - baseline["power"])
                / recipe["power_w"]
                * 100
            ),
            "flow_delta_baseline_pct": float(
                (row["mean_ar_flow_sccm"] - baseline["flow"])
                / recipe["ar_flow_sccm"]
                * 100
            ),
            "pressure_delta_baseline_pct": float(
                (row["mean_pressure_mtorr"] - baseline["pressure"])
                / recipe["pressure_mtorr"]
                * 100
            ),
            "rate_delta_baseline_pct": float(
                (row["mean_rate_nm_s"] - baseline["rate"])
                / recipe["base_rate_nm_s"]
                * 100
            ),
            "thickness_delta_previous": float(
                row["mean_thickness_nm"] - previous["mean_thickness_nm"]
            ),
            "uniformity_delta_previous": float(
                row["mean_uniformity_pct"] - previous["mean_uniformity_pct"]
            ),
            "resistance_delta_previous_pct": float(
                (
                    row["mean_sheet_resistance_ohm_sq"]
                    - previous["mean_sheet_resistance_ohm_sq"]
                )
                / max(abs(previous["mean_sheet_resistance_ohm_sq"]), 1e-9)
                * 100
            ),
        }
        rows.append(feature_row)

    return pd.DataFrame(rows, columns=AI_FEATURE_COLUMNS)

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



def add_difficulty_shortcut_features(dataset):
    shortcut = dataset[AI_FEATURE_COLUMNS].copy()
    for difficulty in DIFFICULTY_ORDER:
        shortcut[f"difficulty_{difficulty}"] = (
            dataset["difficulty"] == difficulty
        ).astype(int)
    return shortcut


def detect_case_from_probabilities(
    case_data,
    abnormal_threshold,
    confidence_threshold=0.0,
    margin_threshold=0.0,
):
    ordered = case_data.sort_values("lot").reset_index(drop=True).copy()
    normal_probability = (
        ordered["probability_normal"]
        if "probability_normal" in ordered.columns
        else pd.Series(np.zeros(len(ordered)))
    )
    ordered["abnormal_probability"] = 1.0 - normal_probability.to_numpy()

    predicted_start = None
    detection_index = None
    for row_index in range(len(ordered)):
        current = float(ordered.loc[row_index, "abnormal_probability"])
        next_value = (
            float(ordered.loc[row_index + 1, "abnormal_probability"])
            if row_index + 1 < len(ordered)
            else current
        )
        if (
            current >= abnormal_threshold
            and (
                next_value >= abnormal_threshold - 0.05
                or current >= abnormal_threshold + 0.15
            )
        ):
            predicted_start = int(ordered.loc[row_index, "lot"])
            detection_index = row_index
            break

    if detection_index is None:
        return {
            "predicted_start": None,
            "top_one": None,
            "raw_top_one": None,
            "top_two": [],
            "top_probability": 0.0,
            "second_probability": 0.0,
            "margin": 0.0,
            "abstained": True,
        }

    cause_columns = [
        cause for cause in CAUSES
        if f"probability_{cause}" in ordered.columns
    ]
    average_probabilities = {
        cause: float(
            ordered.loc[
                detection_index:,
                f"probability_{cause}",
            ].mean()
        )
        for cause in cause_columns
    }
    ranked = sorted(
        average_probabilities,
        key=average_probabilities.get,
        reverse=True,
    )
    top_one = ranked[0] if ranked else None
    top_two = ranked[:2]
    top_probability = (
        average_probabilities[top_one]
        if top_one is not None
        else 0.0
    )
    second_probability = (
        average_probabilities[top_two[1]]
        if len(top_two) > 1
        else 0.0
    )
    margin = top_probability - second_probability
    abstained = (
        top_one is None
        or top_probability < confidence_threshold
        or margin < margin_threshold
    )

    return {
        "predicted_start": predicted_start,
        "top_one": None if abstained else top_one,
        "raw_top_one": top_one,
        "top_two": top_two,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "margin": margin,
        "abstained": abstained,
    }


def simulate_unknown_case(case_id, rng, difficulty):
    material = str(rng.choice(list(MATERIALS.keys())))
    material_props = MATERIALS[material]
    lot_count = int(rng.integers(8, 14))
    unknown_key = str(rng.choice(list(UNKNOWN_CAUSES.keys())))
    fault_start_lot = int(
        rng.choice(
            [1, 2, int(rng.integers(3, max(4, lot_count - 1)))],
            p=[0.20, 0.20, 0.60],
        )
    )
    severity = float(rng.uniform(0.55, 1.20))
    sign = int(rng.choice([-1, 1]))

    profile = build_case_profile(
        rng=rng,
        difficulty=difficulty,
        fault_start_lot=1,
        event_min_lot=1,
        event_max_lot=lot_count - 1,
        has_fault=False,
    )

    summary_rows = []

    for lot in range(1, lot_count + 1):
        measurements = generate_lot_measurements(
            rng=rng,
            material=material,
            lot=lot,
            profile=profile,
        )

        if lot >= fault_start_lot:
            progress = lot - fault_start_lot + 1
            thickness = measurements["thickness"]
            uniformity = measurements["uniformity"]
            sheet_resistance = measurements["sheet_resistance"]

            if unknown_key == "time_overrun":
                # 공정 변수와 증착률은 정상에 가깝지만 증착 시간이 길어져 두께가 증가합니다.
                factor = 1.0 + 0.0065 * progress * severity
                measurements["time_actual"] *= factor
                thickness *= factor
                sheet_resistance /= factor

            elif unknown_key == "metrology_bias":
                # 실제 공정은 정상인데 두께 측정값만 한 방향으로 치우칩니다.
                offset = sign * 0.50 * progress * severity
                thickness += offset
                sheet_resistance[:] = (
                    10
                    * material_props["bulk_resistivity_uohm_cm"]
                    * THIN_FILM_RESISTIVITY_FACTOR
                    / np.maximum(thickness, 1.0)
                )

            elif unknown_key == "substrate_temperature":
                # 두께 변화는 작지만 막질과 균일도, 면저항이 악화됩니다.
                thickness += sign * 0.10 * progress * severity
                uniformity += 0.36 * progress * severity
                sheet_resistance *= 1.0 + 0.022 * progress * severity

            elif unknown_key == "shutter_timing":
                # 일부 웨이퍼에만 두께 상승 또는 하락이 나타납니다.
                affected_count = max(
                    2,
                    int(round(WAFERS_PER_LOT * rng.uniform(0.20, 0.45))),
                )
                affected = rng.choice(
                    WAFERS_PER_LOT,
                    size=affected_count,
                    replace=False,
                )
                thickness[affected] += (
                    sign
                    * 0.75
                    * progress
                    * severity
                    * rng.uniform(0.75, 1.25, affected_count)
                )
                sheet_resistance[affected] = (
                    10
                    * material_props["bulk_resistivity_uohm_cm"]
                    * THIN_FILM_RESISTIVITY_FACTOR
                    / np.maximum(thickness[affected], 1.0)
                )

        summary_rows.append(
            lot_measurements_to_summary(
                case_id=case_id,
                material=material,
                difficulty=difficulty,
                lot=lot,
                measurements=measurements,
                target="unknown" if lot >= fault_start_lot else "normal",
                primary_cause="unknown",
                secondary_cause=unknown_key,
                fault_start_lot=fault_start_lot,
                severity=severity,
            )
        )

    result = pd.DataFrame(summary_rows)
    result["unknown_cause"] = unknown_key
    return result


@st.cache_data(show_spinner=False)
def generate_unknown_test_dataset(case_count, dataset_seed):
    rng = np.random.default_rng(int(dataset_seed) + 99173)
    rows = []
    for case_id in range(1, int(case_count) + 1):
        difficulty = str(
            rng.choice(
                DIFFICULTY_ORDER,
                p=[0.20, 0.35, 0.30, 0.15],
            )
        )
        case_summary = simulate_unknown_case(case_id, rng, difficulty)
        material = str(case_summary["material"].iloc[0])
        features = build_ai_features(case_summary, material)
        features.insert(0, "case_id", case_id)
        features["material"] = material
        features["difficulty"] = difficulty
        features["lot"] = case_summary["lot"].to_numpy()
        features["unknown_cause"] = case_summary["unknown_cause"].to_numpy()
        features["actual_fault_start_lot"] = case_summary[
            "actual_fault_start_lot"
        ].to_numpy()
        rows.append(features)
    return pd.concat(rows, ignore_index=True)


@st.cache_resource(
    show_spinner=False,
)
def train_ai_model(
    case_count,
    dataset_seed,
    training_scope,
):
    dataset = generate_ai_training_dataset(
        int(case_count),
        int(dataset_seed),
        str(training_scope),
    )

    x = dataset[AI_FEATURE_COLUMNS]
    y = dataset["target"]
    groups = dataset["case_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_index, test_index = next(
        splitter.split(x, y, groups=groups)
    )

    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model_parameters = dict(
        n_estimators=230,
        max_depth=16,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model = RandomForestClassifier(**model_parameters)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, prediction)
    macro_f1 = f1_score(
        y_test,
        prediction,
        average="macro",
        zero_division=0,
    )

    # 비교 실험: 실제 현장에는 없는 난이도 라벨을 입력한 모델
    shortcut_x = add_difficulty_shortcut_features(dataset)
    shortcut_model = RandomForestClassifier(**model_parameters)
    shortcut_model.fit(
        shortcut_x.iloc[train_index],
        y_train,
    )
    shortcut_prediction = shortcut_model.predict(
        shortcut_x.iloc[test_index]
    )
    shortcut_accuracy = accuracy_score(
        y_test,
        shortcut_prediction,
    )
    shortcut_macro_f1 = f1_score(
        y_test,
        shortcut_prediction,
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
        target_names=[AI_LABELS[label] for label in AI_CLASS_ORDER],
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
            "feature": AI_FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["feature_label"] = importance_df["feature"].map(
        AI_FEATURE_LABELS
    )

    test_dataset = dataset.iloc[test_index].copy().reset_index(drop=True)
    test_dataset["prediction"] = prediction
    probability_df = pd.DataFrame(
        probability,
        columns=list(model.classes_),
    )
    for class_name in model.classes_:
        test_dataset[f"probability_{class_name}"] = probability_df[
            class_name
        ].to_numpy()

    # 먼저 유보 기준 없이 평가해 신뢰도 임계값을 보정합니다.
    raw_case_records = []
    for case_id, case_data in test_dataset.groupby("case_id"):
        raw = detect_case_from_probabilities(
            case_data,
            abnormal_threshold=AI_ABNORMAL_THRESHOLD,
        )
        actual_start = int(case_data["actual_fault_start_lot"].iloc[0])
        primary = str(case_data["actual_case_cause"].iloc[0])
        if (
            actual_start > 0
            and raw["predicted_start"] is not None
            and raw["raw_top_one"] == primary
        ):
            raw_case_records.append(raw)

    if raw_case_records:
        confidence_threshold = float(
            np.clip(
                np.quantile(
                    [row["top_probability"] for row in raw_case_records],
                    0.20,
                ),
                0.48,
                0.70,
            )
        )
        margin_threshold = float(
            np.clip(
                np.quantile(
                    [row["margin"] for row in raw_case_records],
                    0.15,
                ),
                0.06,
                0.20,
            )
        )
    else:
        confidence_threshold = 0.55
        margin_threshold = 0.10

    def build_case_records(frame):
        records = []
        for case_id, case_data in frame.groupby("case_id"):
            result = detect_case_from_probabilities(
                case_data,
                abnormal_threshold=AI_ABNORMAL_THRESHOLD,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            )
            actual_start = int(
                case_data["actual_fault_start_lot"].iloc[0]
            )
            records.append({
                "case_id": case_id,
                "difficulty": str(case_data["difficulty"].iloc[0]),
                "is_fault_case": actual_start > 0,
                "actual_start": actual_start,
                "predicted_start": result["predicted_start"],
                "primary_cause": str(
                    case_data["actual_case_cause"].iloc[0]
                ),
                "secondary_cause": str(
                    case_data["secondary_cause"].iloc[0]
                ),
                "predicted_primary": result["top_one"],
                "raw_top_one": result["raw_top_one"],
                "top_two": result["top_two"],
                "abstained": result["abstained"],
                "top_probability": result["top_probability"],
                "margin": result["margin"],
            })
        return pd.DataFrame(records)

    case_frame = build_case_records(test_dataset)

    difficulty_rows = []
    for difficulty in DIFFICULTY_ORDER:
        lot_subset = test_dataset[test_dataset["difficulty"] == difficulty]
        case_subset = case_frame[case_frame["difficulty"] == difficulty]
        if lot_subset.empty or case_subset.empty:
            continue

        fault_cases = case_subset[case_subset["is_fault_case"]]
        normal_cases = case_subset[~case_subset["is_fault_case"]]
        predicted_numeric = pd.to_numeric(
            fault_cases["predicted_start"],
            errors="coerce",
        )
        top_two_hits = [
            (
                row["primary_cause"] in row["top_two"]
                or (
                    bool(row["secondary_cause"])
                    and row["secondary_cause"] in row["top_two"]
                )
            )
            for _, row in fault_cases.iterrows()
        ]

        difficulty_rows.append({
            "난이도": DIFFICULTY_LABELS[difficulty],
            "Lot 분류 정확도": accuracy_score(
                lot_subset["target"],
                lot_subset["prediction"],
            ),
            "평균 F1 점수": f1_score(
                lot_subset["target"],
                lot_subset["prediction"],
                average="macro",
                zero_division=0,
            ),
            "이상 Lot 정확 일치": float(
                (fault_cases["predicted_start"] == fault_cases["actual_start"]).mean()
            ) if not fault_cases.empty else np.nan,
            "±1 Lot 이내": float(
                ((predicted_numeric - fault_cases["actual_start"]).abs() <= 1).mean()
            ) if not fault_cases.empty else np.nan,
            "원인 정확도": float(
                (fault_cases["predicted_primary"] == fault_cases["primary_cause"]).mean()
            ) if not fault_cases.empty else np.nan,
            "상위 2개 원인 적중률": float(np.mean(top_two_hits))
            if top_two_hits else np.nan,
            "원인 판단 유보율": float(fault_cases["abstained"].mean())
            if not fault_cases.empty else np.nan,
            "정상 문제 오탐률": float(normal_cases["predicted_start"].notna().mean())
            if not normal_cases.empty else np.nan,
            "평가 문제": int(case_subset.shape[0]),
        })

    difficulty_metrics = pd.DataFrame(difficulty_rows)

    # 초기 이상과 깨끗한 기준 확보 후 이상을 분리해 평가합니다.
    start_rows = []
    condition_definitions = [
        ("첫 Lot부터 이상", lambda frame: frame["actual_start"] == 1),
        ("두 번째 Lot부터 이상", lambda frame: frame["actual_start"] == 2),
        ("Lot 3 이후 이상", lambda frame: frame["actual_start"] >= 3),
        ("정상 문제", lambda frame: ~frame["is_fault_case"]),
    ]
    for label, selector in condition_definitions:
        subset = case_frame[selector(case_frame)]
        if subset.empty:
            continue
        if label == "정상 문제":
            start_rows.append({
                "발생 조건": label,
                "평가 문제": len(subset),
                "이상 시점 정확 일치": np.nan,
                "±1 Lot 이내": np.nan,
                "원인 정확도": np.nan,
                "원인 판단 유보율": np.nan,
                "오탐률": float(subset["predicted_start"].notna().mean()),
            })
        else:
            predicted_numeric = pd.to_numeric(
                subset["predicted_start"],
                errors="coerce",
            )
            start_rows.append({
                "발생 조건": label,
                "평가 문제": len(subset),
                "이상 시점 정확 일치": float(
                    (subset["predicted_start"] == subset["actual_start"]).mean()
                ),
                "±1 Lot 이내": float(
                    ((predicted_numeric - subset["actual_start"]).abs() <= 1).mean()
                ),
                "원인 정확도": float(
                    (subset["predicted_primary"] == subset["primary_cause"]).mean()
                ),
                "원인 판단 유보율": float(subset["abstained"].mean()),
                "오탐률": np.nan,
            })
    start_condition_metrics = pd.DataFrame(start_rows)

    # 학습에 없던 원인만 별도 생성해 유보가 실제로 작동하는지 확인합니다.
    unknown_case_count = max(120, min(400, int(case_count) // 5))
    unknown_dataset = generate_unknown_test_dataset(
        unknown_case_count,
        int(dataset_seed),
    )
    unknown_probability = model.predict_proba(
        unknown_dataset[AI_FEATURE_COLUMNS]
    )
    unknown_probability_df = pd.DataFrame(
        unknown_probability,
        columns=list(model.classes_),
    )
    for class_name in model.classes_:
        unknown_dataset[f"probability_{class_name}"] = unknown_probability_df[
            class_name
        ].to_numpy()

    unknown_records = []
    for case_id, case_data in unknown_dataset.groupby("case_id"):
        result = detect_case_from_probabilities(
            case_data,
            abnormal_threshold=AI_ABNORMAL_THRESHOLD,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        unknown_records.append({
            "case_id": case_id,
            "unknown_cause": str(case_data["unknown_cause"].iloc[0]),
            "detected": result["predicted_start"] is not None,
            "abstained": result["abstained"],
            "false_confident": (
                result["predicted_start"] is not None
                and not result["abstained"]
            ),
        })
    unknown_frame = pd.DataFrame(unknown_records)

    unknown_rows = []
    for unknown_key, subset in unknown_frame.groupby("unknown_cause"):
        detected_subset = subset[subset["detected"]]
        unknown_rows.append({
            "학습에 없던 원인": UNKNOWN_CAUSES[unknown_key],
            "평가 문제": len(subset),
            "이상 탐지율": float(subset["detected"].mean()),
            "탐지 후 원인 판단 유보율": float(
                detected_subset["abstained"].mean()
            ) if not detected_subset.empty else np.nan,
            "기존 원인으로 확신한 비율": float(
                subset["false_confident"].mean()
            ),
        })
    unknown_metrics = pd.DataFrame(unknown_rows)
    detected_unknown = unknown_frame[unknown_frame["detected"]]
    unknown_overall = {
        "case_count": int(len(unknown_frame)),
        "detection_rate": float(unknown_frame["detected"].mean()),
        "abstention_rate_after_detection": float(
            detected_unknown["abstained"].mean()
        ) if not detected_unknown.empty else np.nan,
        "false_confident_rate": float(unknown_frame["false_confident"].mean()),
    }

    train_case_count = int(dataset.iloc[train_index]["case_id"].nunique())
    test_case_count = int(dataset.iloc[test_index]["case_id"].nunique())

    return {
        "model": model,
        "dataset": dataset,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "shortcut_accuracy": float(shortcut_accuracy),
        "shortcut_macro_f1": float(shortcut_macro_f1),
        "shortcut_delta_pp": float((shortcut_accuracy - accuracy) * 100),
        "confusion_matrix": matrix,
        "classification_report": report_df,
        "feature_importance": importance_df,
        "difficulty_metrics": difficulty_metrics,
        "start_condition_metrics": start_condition_metrics,
        "unknown_metrics": unknown_metrics,
        "unknown_overall": unknown_overall,
        "abnormal_threshold": float(AI_ABNORMAL_THRESHOLD),
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "train_cases": train_case_count,
        "test_cases": test_case_count,
        "case_count": int(case_count),
        "dataset_seed": int(dataset_seed),
        "training_scope": str(training_scope),
    }

def ai_diagnose_current_case(
    ai_bundle,
    summary,
    material,
    difficulty=None,
):
    if ai_bundle is None or summary.empty:
        return None

    features = build_ai_features(
        summary=summary,
        material=material,
    )
    model = ai_bundle["model"]
    probability = model.predict_proba(features[AI_FEATURE_COLUMNS])
    probability_df = pd.DataFrame(
        probability,
        columns=list(model.classes_),
    )

    normal_probability = (
        probability_df["normal"]
        if "normal" in probability_df.columns
        else pd.Series(np.zeros(len(features)))
    )
    abnormal_probability = 1.0 - normal_probability
    cause_columns = [
        cause for cause in CAUSES
        if cause in probability_df.columns
    ]
    threshold = ai_bundle.get("abnormal_threshold", AI_ABNORMAL_THRESHOLD)

    lot_predictions = []
    for index, row in summary.reset_index(drop=True).iterrows():
        ranked_causes = (
            probability_df.loc[index, cause_columns]
            .sort_values(ascending=False)
        )
        predicted_key = str(ranked_causes.index[0])
        abnormal_prob = float(abnormal_probability.iloc[index])
        predicted_state = predicted_key if abnormal_prob >= threshold else "normal"
        lot_predictions.append({
            "Lot": int(row["lot"]),
            "AI 판정": AI_LABELS[predicted_state],
            "이상 확률": abnormal_prob,
            "1순위 원인": CAUSES[predicted_key],
            "1순위 확률": float(ranked_causes.iloc[0]),
            "2순위 원인": (
                CAUSES[str(ranked_causes.index[1])]
                if len(ranked_causes) > 1
                else ""
            ),
            "2순위 확률": (
                float(ranked_causes.iloc[1])
                if len(ranked_causes) > 1
                else 0.0
            ),
        })

    prediction_table = pd.DataFrame(lot_predictions)
    first_fault_index = None
    for index in range(len(prediction_table)):
        current = float(prediction_table.loc[index, "이상 확률"])
        next_value = (
            float(prediction_table.loc[index + 1, "이상 확률"])
            if index + 1 < len(prediction_table)
            else current
        )
        if (
            current >= threshold
            and (
                next_value >= threshold - 0.05
                or current >= threshold + 0.15
            )
        ):
            first_fault_index = index
            break

    if first_fault_index is None:
        return {
            "fault_lot": None,
            "cause_key": None,
            "cause_label": "이상 시점과 원인을 특정하기 어려움",
            "confidence": float(abnormal_probability.max()),
            "confidence_label": "낮음",
            "top_causes": [],
            "composite_possible": False,
            "abstained": True,
            "prediction_table": prediction_table,
        }

    fault_lot = int(prediction_table.loc[first_fault_index, "Lot"])
    aggregate_probability = probability_df.loc[
        first_fault_index:,
        cause_columns,
    ].mean()
    ranked = aggregate_probability.sort_values(ascending=False)
    top_causes = [
        {
            "cause_key": str(cause_key),
            "cause_label": CAUSES[str(cause_key)],
            "probability": float(value),
        }
        for cause_key, value in ranked.iloc[:2].items()
    ]
    top_probability = top_causes[0]["probability"] if top_causes else 0.0
    second_probability = top_causes[1]["probability"] if len(top_causes) > 1 else 0.0
    margin = top_probability - second_probability
    confidence_threshold = ai_bundle.get("confidence_threshold", 0.55)
    margin_threshold = ai_bundle.get("margin_threshold", 0.10)
    abstained = (
        top_probability < confidence_threshold
        or margin < margin_threshold
    )

    if abstained:
        confidence_label = "낮음"
        cause_key = None
        cause_label = "원인을 특정하기 어려움"
    elif top_probability >= max(0.75, confidence_threshold + 0.15) and margin >= 0.20:
        confidence_label = "높음"
        cause_key = top_causes[0]["cause_key"]
        cause_label = top_causes[0]["cause_label"]
    else:
        confidence_label = "보통"
        cause_key = top_causes[0]["cause_key"]
        cause_label = top_causes[0]["cause_label"]

    composite_possible = (
        second_probability >= 0.25
        and margin <= 0.25
    )

    return {
        "fault_lot": fault_lot,
        "cause_key": cause_key,
        "cause_label": cause_label,
        "confidence": top_probability,
        "confidence_label": confidence_label,
        "top_causes": top_causes,
        "composite_possible": composite_possible,
        "abstained": abstained,
        "probability_margin": margin,
        "prediction_table": prediction_table,
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


def build_ai_signal_summary(
    summary,
    material,
    detected_lot,
):
    """AI가 이상으로 본 Lot의 값을 목표값과 공정 설정값에 비교해 설명합니다."""
    if (
        summary is None
        or summary.empty
        or detected_lot is None
    ):
        return []

    required_columns = [
        "lot",
        "mean_thickness_nm",
        "mean_power_w",
        "mean_ar_flow_sccm",
        "mean_pressure_mtorr",
        "mean_rate_nm_s",
        "mean_uniformity_pct",
        "mean_sheet_resistance_ohm_sq",
    ]

    if any(
        column not in summary.columns
        for column in required_columns
    ):
        return []

    ordered = (
        summary[required_columns]
        .copy()
        .sort_values("lot")
        .reset_index(drop=True)
    )

    for column in required_columns:
        ordered[column] = pd.to_numeric(
            ordered[column],
            errors="coerce",
        )

    ordered = ordered.dropna(
        subset=required_columns
    )

    if ordered.empty:
        return []

    detected_rows = ordered[
        ordered["lot"]
        == int(detected_lot)
    ]

    if detected_rows.empty:
        detected_row = ordered.iloc[-1]
    else:
        detected_row = detected_rows.iloc[0]

    recipe = get_recipe(material)
    resistance_reference = (
        reference_sheet_resistance(
            material
        )
    )

    thickness_delta = float(
        detected_row["mean_thickness_nm"]
        - TARGET_THICKNESS
    )
    power_delta = float(
        detected_row["mean_power_w"]
        - recipe["power_w"]
    )
    flow_delta = float(
        detected_row["mean_ar_flow_sccm"]
        - recipe["ar_flow_sccm"]
    )
    pressure_delta = float(
        detected_row["mean_pressure_mtorr"]
        - recipe["pressure_mtorr"]
    )
    rate_delta = float(
        detected_row["mean_rate_nm_s"]
        - recipe["base_rate_nm_s"]
    )
    uniformity_excess = float(
        detected_row["mean_uniformity_pct"]
        - UNIFORMITY_LIMIT
    )
    resistance_delta = float(
        detected_row[
            "mean_sheet_resistance_ohm_sq"
        ]
        - resistance_reference
    )

    signal_candidates = [
        {
            "score": abs(
                thickness_delta
            ) / 0.8,
            "text": (
                f"평균 두께가 목표값보다 "
                f"{thickness_delta:+.2f} nm 차이 났습니다."
            ),
        },
        {
            "score": abs(
                power_delta
            ) / max(
                recipe["power_w"] * 0.009,
                4.5,
            ),
            "text": (
                f"실제 전원 출력이 설정값보다 "
                f"{power_delta:+.2f} W 차이 났습니다."
            ),
        },
        {
            "score": abs(
                flow_delta
            ) / 0.7,
            "text": (
                f"실제 Ar 유량이 설정값보다 "
                f"{flow_delta:+.2f} sccm 차이 났습니다."
            ),
        },
        {
            "score": abs(
                pressure_delta
            ) / 0.20,
            "text": (
                f"실제 챔버 압력이 설정값보다 "
                f"{pressure_delta:+.3f} mTorr 차이 났습니다."
            ),
        },
        {
            "score": abs(
                rate_delta
            ) / max(
                recipe["base_rate_nm_s"]
                * 0.006,
                0.003,
            ),
            "text": (
                f"평균 증착률이 기준값보다 "
                f"{rate_delta:+.4f} nm/s 차이 났습니다."
            ),
        },
        {
            "score": max(
                uniformity_excess,
                0.0,
            ) / 0.5,
            "text": (
                f"평균 균일도 지표가 관리 기준을 "
                f"{max(uniformity_excess, 0.0):.2f}%p 초과했습니다."
            ),
        },
        {
            "score": abs(
                resistance_delta
            ) / max(
                abs(
                    resistance_reference
                ) * 0.06,
                0.01,
            ),
            "text": (
                f"예상 면저항이 100 nm 기준값보다 "
                f"{resistance_delta:+.4f} Ω/□ 차이 났습니다."
            ),
        },
    ]

    ranked = sorted(
        signal_candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    return [
        item["text"]
        for item in ranked[:3]
        if item["score"] >= 0.25
    ]

def render_ai_model_details(ai_bundle):
    """AI 모델의 세부 성능과 실패 조건을 표시합니다."""
    if ai_bundle is None:
        st.info(
            "아직 준비된 AI 모델이 없습니다. 첫 진단을 제출하면 기본 모델이 자동으로 준비됩니다."
        )
        return

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)
    model_col1.metric("학습용 문제", f"{ai_bundle['case_count']:,}개")
    model_col2.metric("난이도 정보 제거 후 정확도", f"{ai_bundle['accuracy'] * 100:.1f}%")
    model_col3.metric("평균 F1 점수", f"{ai_bundle['macro_f1']:.3f}")
    model_col4.metric("평가에 사용한 문제", f"{ai_bundle['test_cases']:,}개")

    st.markdown("##### 실제로 사용할 수 없는 난이도 정보 제거 실험")
    ablation_df = pd.DataFrame([
        {
            "모델": "난이도 정보를 입력한 비교 모델",
            "평가 정확도 (%)": ai_bundle["shortcut_accuracy"] * 100,
            "평균 F1 점수": ai_bundle["shortcut_macro_f1"],
            "실제 진단에 사용": "아니요",
        },
        {
            "모델": "난이도 정보를 제거한 현재 모델",
            "평가 정확도 (%)": ai_bundle["accuracy"] * 100,
            "평균 F1 점수": ai_bundle["macro_f1"],
            "실제 진단에 사용": "예",
        },
    ])
    ablation_df["평가 정확도 (%)"] = ablation_df["평가 정확도 (%)"].round(1)
    ablation_df["평균 F1 점수"] = ablation_df["평균 F1 점수"].round(3)
    st.dataframe(ablation_df, hide_index=True, width="stretch")
    delta = ai_bundle["shortcut_delta_pp"]
    if delta >= 0:
        st.caption(
            f"난이도 정보를 넣었을 때 정확도가 {delta:.1f}%p 높았습니다. "
            "현재 모델은 실제 공정에서 알 수 없는 난이도 정보를 입력하지 않습니다."
        )
    else:
        st.caption(
            f"난이도 정보를 제거한 모델이 비교 모델보다 {-delta:.1f}%p 높았습니다. "
            "난이도 라벨은 현재 진단 모델의 입력에서 제외했습니다."
        )

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("##### 실제 원인과 AI 예측 비교")
        matrix = ai_bundle["confusion_matrix"]
        confusion_rows = []
        for actual_index, actual_key in enumerate(AI_CLASS_ORDER):
            for predicted_index, predicted_key in enumerate(AI_CLASS_ORDER):
                confusion_rows.append({
                    "실제 발생 조건": AI_LABELS[actual_key],
                    "AI 예측": AI_LABELS[predicted_key],
                    "건수": int(matrix[actual_index, predicted_index]),
                })
        confusion_df = pd.DataFrame(confusion_rows)
        heatmap = (
            alt.Chart(confusion_df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "AI 예측:N",
                    title="AI 예측",
                    sort=[AI_LABELS[key] for key in AI_CLASS_ORDER],
                ),
                y=alt.Y(
                    "실제 발생 조건:N",
                    title="실제 발생 조건",
                    sort=[AI_LABELS[key] for key in AI_CLASS_ORDER],
                ),
                color=alt.Color("건수:Q", title="건수"),
                tooltip=["실제 발생 조건:N", "AI 예측:N", "건수:Q"],
            )
            .properties(height=300)
        )
        heatmap_text = (
            alt.Chart(confusion_df)
            .mark_text()
            .encode(
                x=alt.X(
                    "AI 예측:N",
                    sort=[AI_LABELS[key] for key in AI_CLASS_ORDER],
                ),
                y=alt.Y(
                    "실제 발생 조건:N",
                    sort=[AI_LABELS[key] for key in AI_CLASS_ORDER],
                ),
                text="건수:Q",
                color=alt.condition(
                    "datum.건수 > 500",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )
        st.altair_chart(heatmap + heatmap_text, width="stretch")

    with detail_col2:
        st.markdown("##### AI가 중요하게 본 변수")
        top_importance = (
            ai_bundle["feature_importance"]
            .head(12)
            .sort_values("importance", ascending=True)
        )
        importance_chart = (
            alt.Chart(top_importance)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="중요도"),
                y=alt.Y("feature_label:N", title=None, sort=None),
                tooltip=[
                    alt.Tooltip("feature_label:N", title="변수"),
                    alt.Tooltip("importance:Q", title="중요도", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(importance_chart, width="stretch")

    st.markdown("##### 이상이 처음부터 존재할 때의 성능")
    start_metrics = ai_bundle["start_condition_metrics"].copy()
    for column in [
        "이상 시점 정확 일치",
        "±1 Lot 이내",
        "원인 정확도",
        "원인 판단 유보율",
        "오탐률",
    ]:
        if column in start_metrics.columns:
            start_metrics[column] = (start_metrics[column] * 100).round(1)
    st.dataframe(start_metrics, hide_index=True, width="stretch")
    st.caption(
        "첫 Lot 또는 두 번째 Lot부터 이상이 시작된 문제에서는 초기 두 Lot 평균이 깨끗한 정상 기준이 아닙니다. "
        "목표 두께와 공정 설정값 대비 편차를 함께 사용해 이 조건을 별도로 평가합니다."
    )

    st.markdown("##### 학습에 없던 원인 시험")
    unknown_overall = ai_bundle["unknown_overall"]
    unknown_col1, unknown_col2, unknown_col3 = st.columns(3)
    unknown_col1.metric(
        "미지 원인 이상 탐지율",
        f"{unknown_overall['detection_rate'] * 100:.1f}%",
    )
    unknown_col2.metric(
        "탐지 후 원인 판단 유보율",
        f"{unknown_overall['abstention_rate_after_detection'] * 100:.1f}%",
    )
    unknown_col3.metric(
        "기존 원인으로 확신한 비율",
        f"{unknown_overall['false_confident_rate'] * 100:.1f}%",
    )
    unknown_metrics = ai_bundle["unknown_metrics"].copy()
    for column in [
        "이상 탐지율",
        "탐지 후 원인 판단 유보율",
        "기존 원인으로 확신한 비율",
    ]:
        unknown_metrics[column] = (unknown_metrics[column] * 100).round(1)
    st.dataframe(unknown_metrics, hide_index=True, width="stretch")
    st.caption(
        "증착 시간 과다, 두께 측정 장비 편향, 기판 온도 이상, 셔터 동작 이상은 학습 클래스에 넣지 않았습니다. "
        "모델이 모르는 원인에서도 무조건 기존 네 원인 중 하나를 확신하는지 확인하는 별도 시험입니다."
    )

    st.markdown("##### 문제 난이도별 성능")
    difficulty_metrics = ai_bundle["difficulty_metrics"].copy()
    percentage_columns = [
        "Lot 분류 정확도",
        "이상 Lot 정확 일치",
        "±1 Lot 이내",
        "원인 정확도",
        "상위 2개 원인 적중률",
        "원인 판단 유보율",
        "정상 문제 오탐률",
    ]
    for column in percentage_columns:
        if column in difficulty_metrics.columns:
            difficulty_metrics[column] = (difficulty_metrics[column] * 100).round(1)
    if "평균 F1 점수" in difficulty_metrics.columns:
        difficulty_metrics = difficulty_metrics.rename(
            columns={
                "평균 F1 점수": "평균 F1 점수",
                "상위 2개 원인 적중률": "상위 2개 원인 적중률",
                "Lot 분류 정확도": "Lot 상태 분류 정확도",
            }
        )
        difficulty_metrics["평균 F1 점수"] = difficulty_metrics[
            "평균 F1 점수"
        ].round(3)
    st.dataframe(difficulty_metrics, hide_index=True, width="stretch")

    st.caption(
        f"원인 판단 유보 기준: 1순위 확률 {ai_bundle['confidence_threshold']:.2f} 이상, "
        f"1·2순위 확률 차이 {ai_bundle['margin_threshold']:.2f} 이상. "
        "이 기준은 미사용 평가 데이터에서 보정했습니다."
    )

ensure_state()

# 첫 진단 시 자동으로 준비할 기본 AI 모델 설정
DEFAULT_AI_CASE_COUNT = 1000
DEFAULT_AI_DATASET_SEED = 20260805
DEFAULT_AI_TRAINING_SCOPE = "mixed"

st.title(
    "가상 반도체 박막 공정 이상 진단 훈련"
)
st.caption(
    f"{APP_VERSION} · {LAST_UPDATED} 업데이트 · "
    "증착 데이터를 분석해 이상 시점과 원인을 진단하고 AI 결과와 비교합니다."
)

top_info_col1, top_info_col2 = st.columns(2)

with top_info_col1:
    with st.expander(
        "업데이트 내역",
        expanded=False,
    ):
        st.markdown(
            """
            - **v2.4.5**: 사용 방법의 취소선 오류 수정, 정답 해설에 판단 근거·추가 확인·대응 방안 추가, 풀이 기록 화면 정리
            - **v2.4.4**: 결과 화면을 내 답과 실제 정답 중심으로 재구성, 중복 오답 카드 제거, 규칙 기반 탐지와 AI를 보조 분석 영역으로 통합
            - **v2.4.3**: 진단 서술형 답변의 최소 글자 수 제한 제거, 짧은 예시 문구와 입력창 높이 조정
            - **v2.4.2**: 모바일 현황판 2×2 압축, 데이터 분석 영역에 Lot 생산 버튼 추가, 문제 번호와 표 명칭·열 이름 한글화
            - **v2.4.1**: 새 문제와 동일 문제 다시 시작 기능 분리, 쉬움·보통 이상 시작 범위 확대, 상승·하락 방향 균형화
            - **v2.4.0**: 이상 발생 버튼 제거, 새 문제 시작 시 이상 조건 자동 설정, 난이도별 초기 Lot 안내 유지, 결과 비교 화면 단순화
            - **v2.3.2**: 쉬움·보통은 Lot 1~2를 정상 기준으로 보장하고, 어려움·전문가는 초기 이상을 허용하도록 문제 규칙과 AI 학습 데이터를 통일
            - **v2.3.1**: 진단 제출 후 실제 발생 조건을 먼저 공개하고, AI 비교는 선택 실행하도록 변경 · AI 설명 생성 오류 수정
            - **v2.3.0**: 난이도 입력 변수 제거, 초기 이상 학습, 미지 원인 시험, 두께 상승·하락 이상 추가
            - **v2.2.0**: 사용 방법 추가, 화면 순서 개편, 전체 한글 표현 개선, AI 결과를 진단 제출 후 공개
            - **v2.1.0**: 문제 난이도 선택, 센서 편향·일시적 변동·부분 웨이퍼·복합 이상 추가
            - **v2.0.0**: 합성 학습 데이터 생성과 랜덤 포레스트 기반 AI 진단 추가
            - **v1.2.0~v1.2.2**: 분석 대시보드, 화면 폭, 그래프 축 개선
            - **v1.1**: Al·Cu·Ti·Ta 선택과 예상 면저항 기능 추가
            - **v1.0**: 기본 Sputter 공정 시뮬레이터 구현
            """
        )

with top_info_col2:
    with st.expander(
        "프로젝트 안내와 한계",
        expanded=False,
    ):
        st.markdown(
            """
            - 실제 기업의 공정 조건이나 생산 데이터를 사용하지 않은 교육용 합성 데이터 시뮬레이터입니다.
            - 재료별 증착률과 공정 조건은 비교 학습을 위해 단순화한 가정값입니다.
            - 녹는점과 밀도는 참고 정보이며 Sputter 계산식에 직접 사용하지 않습니다.
            - 예상 면저항은 기준 비저항과 생성된 두께를 이용한 추정값이며 실제 측정값이 아닙니다.
            - AI 성능은 이 시뮬레이터가 생성한 미사용 합성 문제에 대한 결과이며 실제 생산 라인의 성능을 의미하지 않습니다.
            """
        )

# =========================================================
# 1. 사용 방법
# =========================================================
st.subheader("1. 사용 방법")

with st.container(border=True):
    st.markdown(
        """
        1. **증착 재료와 문제 난이도**를 선택합니다.
        2. **새 랜덤 문제**를 누르면 새로운 공정 문제가 생성됩니다. 현재 문제를 처음부터 다시 풀 때는 **같은 문제 다시 시작**을 누릅니다.
        3. **쉬움·보통**에서는 Lot 1과 Lot 2가 정상 기준 Lot으로 자동 생성됩니다. 쉬움에서는 Lot 3, 4, 5 중 한 시점에 이상이 시작되고, 보통에서는 Lot 3부터 7까지 중 한 시점에 이상이 시작됩니다.
        4. **어려움·전문가**에서는 초기 Lot의 정상 여부가 공개되지 않으며 Lot 1부터 이상이 포함될 수 있습니다.
        5. **다음 Lot 생산**을 누르면서 두께와 공정 변수의 변화를 확인합니다.
        6. 이상이 처음 나타난 Lot과 원인을 추정한 뒤 판단 근거, 추가 확인 항목과 대응 방안을 작성합니다.
        7. 진단을 제출하면 내 답과 실제 정답을 먼저 비교합니다. 규칙 기반 탐지와 AI 분석은 필요할 때만 확인합니다.
        """
    )
    st.info(
        "AI는 힌트를 미리 제공하지 않습니다. "
        "사용자가 진단을 제출한 뒤에만 별도로 확인할 수 있습니다."
    )

# =========================================================
# 2. 문제 설정
# =========================================================
st.subheader("2. 문제 설정")

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
        "증착 재료",
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
        "문제 난이도",
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
        "선택한 재료와 난이도를 적용하려면 "
        "아래에서 '새 랜덤 문제' 또는 '같은 문제 다시 시작'을 눌러주세요."
    )

active_material = (
    st.session_state.active_material
)
active_difficulty = (
    st.session_state.active_difficulty
)
material_props = MATERIALS[
    active_material
]
recipe = get_recipe(
    active_material
)
reference_lots_guaranteed = has_guaranteed_reference_lots(
    active_difficulty
)
initial_condition_text = (
    "Lot 1과 Lot 2 정상 기준 보장"
    if reference_lots_guaranteed
    else "초기 정상 상태 미보장"
)

st.markdown(
    f"#### {active_material} 박막 DC 마그네트론 스퍼터링 · "
    f"{DIFFICULTY_LABELS[active_difficulty]}"
)

condition_col, recipe_col, property_col = (
    st.columns(3)
)

with condition_col:
    st.markdown(
        f"""
        **생산 조건**

        - 기판: 300 mm 실리콘 웨이퍼
        - Lot 크기: {WAFERS_PER_LOT}장
        - 증착막: {active_material}
        - 문제 난이도: {DIFFICULTY_LABELS[active_difficulty]}
        - 초기 Lot 조건: {initial_condition_text}
        - 목표 두께: {TARGET_THICKNESS:.0f} nm
        - 두께 관리 범위: {LOWER_SPEC:.0f}~{UPPER_SPEC:.0f} nm
        - 균일도 관리 기준: {UNIFORMITY_LIMIT:.1f}% 이하
        """
    )

with recipe_col:
    st.dataframe(
        pd.DataFrame({
            "기준 공정 조건": [
                "직류 전원 출력",
                "Ar 유량",
                "챔버 압력",
                "증착 시간",
                "기준 증착률",
            ],
            "설정값": [
                f"{recipe['power_w']:.0f} W",
                f"{recipe['ar_flow_sccm']:.0f} sccm",
                f"{recipe['pressure_mtorr']:.1f} mTorr",
                f"{recipe['deposition_time_s']:.1f} s",
                f"{recipe['base_rate_nm_s']:.2f} nm/s",
            ],
        }),
        hide_index=True,
        width="stretch",
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
            "사용 방식": [
                "예상 면저항 계산",
                "참고 정보",
                "참고 정보",
                "비교 기준",
            ],
        }),
        hide_index=True,
        width="stretch",
    )

st.caption(
    material_props["description"]
    + " · "
    + DIFFICULTIES[
        active_difficulty
    ]["description"]
)

if has_guaranteed_reference_lots(
    active_difficulty
):
    st.info(
        "초기 Lot 안내: Lot 1과 Lot 2는 정상 상태의 기준 Lot입니다. "
        "쉬움은 Lot 3부터 5까지 중 한 시점에, 보통은 Lot 3부터 7까지 중 한 시점에 이상이 시작됩니다. "
        "따라서 Lot 3이 정상일 수도 있습니다."
    )
else:
    st.warning(
        "초기 Lot 안내: 초기 Lot이 정상이라는 보장은 없습니다. "
        "Lot 1부터 이상이 포함될 수 있으므로 목표값과 공정 설정값을 함께 확인하세요."
    )

# =========================================================
# 3. 공정 가동
# =========================================================
st.subheader("3. 공정 가동")

if "seed_input" not in st.session_state:
    st.session_state.seed_input = int(st.session_state.case_seed)

seed_input = st.number_input(
    "문제 번호",
    min_value=1,
    max_value=999_999_999,
    step=1,
    key="seed_input",
    help=(
        "같은 문제를 처음부터 다시 시작하거나, "
        "다른 사람과 동일한 문제를 확인할 때 사용합니다. "
        "'새 랜덤 문제'를 누르면 번호가 자동으로 바뀝니다."
    ),
)

st.caption(
    f"현재 진행 중인 문제 번호: {st.session_state.case_seed}"
)

def start_random_problem(
    material,
    difficulty,
):
    new_seed = random.SystemRandom().randint(
        1,
        999_999_999,
    )
    st.session_state.seed_input = new_seed
    initialize_case(
        material=material,
        difficulty=difficulty,
        seed=new_seed,
    )


def replay_problem(
    material,
    difficulty,
):
    initialize_case(
        material=material,
        difficulty=difficulty,
        seed=int(st.session_state.seed_input),
    )


def produce_next_lot_with_message():
    new_lot = produce_lot()

    oos_count = int(
        (
            (
                new_lot["thickness_nm"]
                < LOWER_SPEC
            )
            | (
                new_lot["thickness_nm"]
                > UPPER_SPEC
            )
        ).sum()
    )

    uniformity_fail_count = int(
        (
            new_lot["uniformity_pct"]
            > UNIFORMITY_LIMIT
        ).sum()
    )

    return (
        f"{st.session_state.active_material} Lot "
        f"{st.session_state.current_lot} 생산 완료 · "
        f"두께 기준 이탈 {oos_count}장 · "
        f"균일도 기준 초과 {uniformity_fail_count}장"
    )


button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    st.button(
        "새 랜덤 문제",
        type="primary",
        width="stretch",
        on_click=start_random_problem,
        args=(
            selected_material,
            selected_difficulty,
        ),
    )

with button_col2:
    st.button(
        "같은 문제 다시 시작",
        width="stretch",
        on_click=replay_problem,
        args=(
            selected_material,
            selected_difficulty,
        ),
    )

with button_col3:
    if st.button(
        "다음 Lot 생산",
        width="stretch",
        key="top_next_lot",
    ):
        st.success(
            produce_next_lot_with_message()
        )

if has_guaranteed_reference_lots(
    active_difficulty
):
    st.info(
        "Lot 1과 Lot 2는 정상 기준 Lot으로 이미 생성되어 있습니다. "
        "Lot 3 이후에도 정상 Lot이 이어질 수 있으며, 이상이 시작되면 두께는 원인에 따라 상승하거나 하락할 수 있습니다. "
        "정확한 시작 시점과 원인은 공개되지 않습니다."
    )
else:
    st.warning(
        "초기 정상 상태가 보장되지 않는 문제입니다. "
        "Lot 1부터 이상이 포함될 수 있으며 정확한 시작 시점과 원인은 공개되지 않습니다."
    )

raw_df = (
    st.session_state.raw_data
)
summary_df = summarize(
    raw_df
)

st.markdown(
    f"""
    <div class="process-status-grid">
        <div class="process-status-card">
            <div class="process-status-label">증착 재료</div>
            <div class="process-status-value">{active_material}</div>
        </div>
        <div class="process-status-card">
            <div class="process-status-label">문제 난이도</div>
            <div class="process-status-value">{DIFFICULTY_LABELS[active_difficulty]}</div>
        </div>
        <div class="process-status-card">
            <div class="process-status-label">생산 Lot</div>
            <div class="process-status-value">{st.session_state.current_lot}</div>
        </div>
        <div class="process-status-card">
            <div class="process-status-label">웨이퍼 수</div>
            <div class="process-status-value">{len(raw_df)}</div>
        </div>
    </div>
    <div class="problem-number-line">
        문제 번호: {st.session_state.case_seed}
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. 데이터 분석
# =========================================================
st.subheader("4. 데이터 분석")

if st.button(
    "다음 Lot 생산",
    type="primary",
    width="stretch",
    key="analysis_next_lot",
    help=(
        "데이터를 확인하면서 다음 Lot을 바로 생산합니다."
    ),
):
    st.success(
        produce_next_lot_with_message()
    )
    raw_df = st.session_state.raw_data
    summary_df = summarize(
        raw_df
    )

if summary_df.empty:
    st.info(
        "'다음 Lot 생산'을 누르면 두께, 공정 변수, 예상 면저항과 "
        "Lot별 요약 데이터가 표시됩니다."
    )
else:
    display_summary_df = summary_df.copy()
    display_raw_df = raw_df.copy()

    if reference_lots_guaranteed:
        display_summary_df.insert(
            2,
            "lot_section",
            np.where(display_summary_df["lot"] <= 2, "정상 기준 Lot", "분석 대상 Lot"),
        )
        display_raw_df.insert(
            3,
            "lot_section",
            np.where(display_raw_df["lot"] <= 2, "정상 기준 Lot", "분석 대상 Lot"),
        )
    else:
        display_summary_df.insert(2, "lot_section", "초기 상태 미공개")
        display_raw_df.insert(3, "lot_section", "초기 상태 미공개")

    lot_min = int(
        summary_df["lot"].min()
    )
    lot_max = int(
        summary_df["lot"].max()
    )
    lot_count = max(
        lot_max - lot_min + 1,
        1,
    )
    lot_tick_count = min(
        lot_count,
        10,
    )

    lot_axis = alt.Axis(
        title="Lot",
        labelAngle=0,
        tickCount=lot_tick_count,
        tickMinStep=1,
        labelOverlap="greedy",
    )

    thickness_min = min(
        float(
            raw_df[
                "thickness_nm"
            ].min()
        ),
        LOWER_SPEC,
    )
    thickness_max = max(
        float(
            raw_df[
                "thickness_nm"
            ].max()
        ),
        UPPER_SPEC,
    )
    thickness_span = max(
        thickness_max
        - thickness_min,
        1.0,
    )
    thickness_margin = max(
        thickness_span * 0.08,
        0.25,
    )
    thickness_domain = [
        thickness_min
        - thickness_margin,
        thickness_max
        + thickness_margin,
    ]

    dashboard_col1, dashboard_col2 = (
        st.columns(2)
    )

    with dashboard_col1:
        st.markdown("#### 두께 추이")

        wafer_points = (
            alt.Chart(raw_df)
            .mark_circle(
                size=48,
                opacity=0.42,
            )
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
                    title="두께 (nm)",
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
                        title="웨이퍼",
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
            .mark_line(
                point=True,
                strokeWidth=2.5,
            )
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
                    title="두께 (nm)",
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
                        title="Lot 평균 두께 (nm)",
                        format=".3f",
                    ),
                ],
            )
        )

        thickness_limits = pd.DataFrame({
            "기준": [
                "상한",
                "목표",
                "하한",
            ],
            "두께": [
                UPPER_SPEC,
                TARGET_THICKNESS,
                LOWER_SPEC,
            ],
        })

        limit_rules = (
            alt.Chart(
                thickness_limits
            )
            .mark_rule(
                strokeDash=[5, 4]
            )
            .encode(
                y=alt.Y(
                    "두께:Q",
                    title="두께 (nm)",
                ),
                strokeDash=alt.StrokeDash(
                    "기준:N",
                    title="관리 기준",
                ),
                tooltip=[
                    alt.Tooltip(
                        "기준:N",
                        title="관리 기준",
                    ),
                    alt.Tooltip(
                        "두께:Q",
                        title="두께 (nm)",
                        format=".1f",
                    ),
                ],
            )
        )

        thickness_layers = wafer_points + lot_mean_line + limit_rules

        if reference_lots_guaranteed:
            reference_lot_region = (
                alt.Chart(pd.DataFrame({"시작": [0.5], "끝": [2.5]}))
                .mark_rect(opacity=0.08)
                .encode(
                    x=alt.X(
                        "시작:Q",
                        scale=alt.Scale(domain=[lot_min - 0.4, lot_max + 0.4]),
                    ),
                    x2="끝:Q",
                )
            )
            thickness_layers = reference_lot_region + thickness_layers
            st.caption("옅게 표시된 Lot 1~2 구간은 정상 기준 Lot입니다.")
        else:
            st.caption("초기 정상 상태가 보장되지 않으므로 목표 두께와 관리 기준을 중심으로 해석하세요.")

        st.altair_chart(
            thickness_layers.properties(height=285),
            width="stretch",
        )

    with dashboard_col2:
        st.markdown(
            "#### 기준값 대비 공정 변수 변화"
        )

        process_deviation = pd.DataFrame({
            "Lot": summary_df["lot"],
            "전원 출력 (%)": (
                (
                    summary_df[
                        "mean_power_w"
                    ]
                    - recipe["power_w"]
                )
                / recipe["power_w"]
                * 100
            ),
            "Ar 유량 (%)": (
                (
                    summary_df[
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
            "챔버 압력 (%)": (
                (
                    summary_df[
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
        })

        process_long = (
            process_deviation.melt(
                id_vars="Lot",
                var_name="공정 변수",
                value_name=(
                    "기준값 대비 편차 (%)"
                ),
            )
        )

        zero_line = (
            alt.Chart(
                pd.DataFrame({
                    "기준": [0]
                })
            )
            .mark_rule(
                strokeDash=[4, 4]
            )
            .encode(
                y="기준:Q"
            )
        )

        process_chart = (
            alt.Chart(
                process_long
            )
            .mark_line(
                point=True,
                strokeWidth=2.3,
            )
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
                    "기준값 대비 편차 (%):Q",
                    title=(
                        "기준값 대비 편차 (%)"
                    ),
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
                        "기준값 대비 편차 (%):Q",
                        title="편차 (%)",
                        format=".3f",
                    ),
                ],
            )
            .properties(height=285)
        )

        st.altair_chart(
            process_chart
            + zero_line,
            width="stretch",
        )

    dashboard_col3, dashboard_col4 = (
        st.columns(2)
    )

    with dashboard_col3:
        st.markdown(
            "#### 예상 면저항 추이"
        )

        resistance_reference = (
            pd.DataFrame({
                "기준 면저항": [
                    reference_sheet_resistance(
                        active_material
                    )
                ]
            })
        )

        resistance_line = (
            alt.Chart(summary_df)
            .mark_line(
                point=True,
                strokeWidth=2.5,
            )
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
                    title=(
                        "예상 면저항 (Ω/□)"
                    ),
                    scale=alt.Scale(
                        zero=False
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "lot:Q",
                        title="Lot",
                        format=".0f",
                    ),
                    alt.Tooltip(
                        "mean_sheet_resistance_ohm_sq:Q",
                        title=(
                            "예상 면저항 (Ω/□)"
                        ),
                        format=".4f",
                    ),
                ],
            )
        )

        resistance_rule = (
            alt.Chart(
                resistance_reference
            )
            .mark_rule(
                strokeDash=[5, 4]
            )
            .encode(
                y=alt.Y(
                    "기준 면저항:Q",
                    title=(
                        "예상 면저항 (Ω/□)"
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "기준 면저항:Q",
                        title=(
                            "100 nm 기준 (Ω/□)"
                        ),
                        format=".4f",
                    )
                ],
            )
        )

        st.altair_chart(
            (
                resistance_line
                + resistance_rule
            ).properties(height=260),
            width="stretch",
        )

    with dashboard_col4:
        st.markdown(
            "#### Lot 요약"
        )

        compact_summary = (
            display_summary_df[
                [
                    "lot",
                    "lot_section",
                    "mean_thickness_nm",
                    "std_thickness_nm",
                    "mean_uniformity_pct",
                    "mean_sheet_resistance_ohm_sq",
                    "oos_count",
                ]
            ]
            .rename(
                columns={
                    "lot": "Lot",
                    "lot_section": "Lot 구분",
                    "mean_thickness_nm": "평균 두께 (nm)",
                    "std_thickness_nm": "두께 표준편차",
                    "mean_uniformity_pct": "평균 균일도 (%)",
                    "mean_sheet_resistance_ohm_sq": "예상 면저항 (Ω/□)",
                    "oos_count": "두께 이탈 수",
                }
            )
        )

        st.dataframe(
            compact_summary,
            hide_index=True,
            width="stretch",
            height=300,
        )

    raw_tab, full_summary_tab = (
        st.tabs(
            [
                "웨이퍼별 상세 데이터",
                "Lot별 요약 데이터",
            ]
        )
    )

    with raw_tab:
        st.dataframe(
            display_raw_df.rename(
                columns=RAW_DISPLAY_COLUMNS
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )

        st.download_button(
            "웨이퍼별 상세 데이터 CSV 다운로드",
            data=csv_bytes(
                raw_df.rename(
                    columns=RAW_DISPLAY_COLUMNS
                )
            ),
            file_name=(
                f"sputter_{active_material}_wafer_"
                f"problem_{st.session_state.case_seed}.csv"
            ),
            mime="text/csv",
        )

    with full_summary_tab:
        st.dataframe(
            display_summary_df.rename(
                columns=SUMMARY_DISPLAY_COLUMNS
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )

        st.download_button(
            "Lot별 요약 데이터 CSV 다운로드",
            data=csv_bytes(
                summary_df.rename(
                    columns=SUMMARY_DISPLAY_COLUMNS
                )
            ),
            file_name=(
                f"sputter_{active_material}_lot_summary_"
                f"problem_{st.session_state.case_seed}.csv"
            ),
            mime="text/csv",
        )

# =========================================================
# 5. 사용자 진단
# =========================================================
st.subheader("5. 사용자 진단")

st.caption(
    (
        "Lot 1과 Lot 2를 정상 기준으로 삼아 이후 변화를 분석하세요. "
        if reference_lots_guaranteed
        else "초기 정상 Lot이 보장되지 않으므로 목표 두께, 관리 범위와 공정 설정값을 기준으로 분석하세요. "
    )
    + "서술형 답변은 짧게 작성해도 됩니다. "
    + "AI 분석과 실제 발생 조건은 제출 전에는 공개되지 않습니다."
)

with st.form(
    "diagnosis_form"
):
    guess_col1, guess_col2, guess_col3 = (
        st.columns(3)
    )

    with guess_col1:
        guessed_lot = st.number_input(
            "이상이 처음 나타난 Lot",
            min_value=1,
            max_value=max(
                st.session_state.current_lot,
                1,
            ),
            value=1,
            step=1,
        )

    with guess_col2:
        guessed_cause = (
            st.selectbox(
                "가장 가능성이 높은 원인",
                list(
                    CAUSES.values()
                ),
            )
        )

    with guess_col3:
        guessed_secondary_cause = (
            st.selectbox(
                "다음으로 가능성이 높은 원인",
                ["없음"]
                + list(
                    CAUSES.values()
                ),
            )
        )

    evidence = st.text_area(
        "판단 근거",
        placeholder=(
            "예: Lot 5부터 두께 하락"
        ),
        height=82,
    )

    additional_check = (
        st.text_area(
            "추가로 확인할 항목",
            placeholder=(
                "예: 전원 출력 로그"
            ),
            height=72,
        )
    )

    corrective_action = (
        st.text_area(
            "대응 방안",
            placeholder=(
                "예: 전원 계통 점검"
            ),
            height=72,
        )
    )

    submit_diagnosis = (
        st.form_submit_button(
            "진단 제출하고 결과 확인",
            width="stretch",
        )
    )

if submit_diagnosis:
    errors = []

    if (
        st.session_state.cause_key
        is None
    ):
        errors.append(
            "새 문제를 다시 시작한 뒤 Lot을 생산해주세요."
        )

    if (
        st.session_state.fault_start_lot
        is not None
        and st.session_state.current_lot
        < st.session_state.fault_start_lot
    ):
        errors.append(
            "이상 신호가 데이터에 나타날 때까지 "
            "Lot을 더 생산하세요."
        )

    if not evidence.strip():
        errors.append(
            "판단 근거를 입력하세요."
        )

    if not additional_check.strip():
        errors.append(
            "추가로 확인할 항목을 입력하세요."
        )

    if not corrective_action.strip():
        errors.append(
            "대응 방안을 입력하세요."
        )

    if errors:
        for error in errors:
            st.error(error)

    else:
        actual_key = (
            st.session_state.cause_key
        )
        actual_cause = (
            CAUSES[actual_key]
        )
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

        automatic_lot = (
            automatic_abnormal_lot(
                summary_df
            )
        )

        lot_correct = (
            int(guessed_lot)
            == actual_lot
        )
        cause_correct = (
            guessed_cause
            == actual_cause
        )
        secondary_correct = (
            guessed_secondary_cause
            == actual_secondary_cause
        )

        result = {
            "timestamp": (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
            "material": active_material,
            "material_key": active_material,
            "difficulty": (
                DIFFICULTY_LABELS[
                    active_difficulty
                ]
            ),
            "difficulty_key": active_difficulty,
            "case_seed": (
                st.session_state.case_seed
            ),
            "actual_fault_lot": (
                actual_lot
            ),
            "guessed_fault_lot": int(
                guessed_lot
            ),
            "lot_correct": lot_correct,
            "actual_cause": (
                actual_cause
            ),
            "guessed_cause": (
                guessed_cause
            ),
            "cause_correct": (
                cause_correct
            ),
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
                lot_correct
                and cause_correct
            ),
            "auto_detected_lot": (
                automatic_lot
            ),
            "summary_snapshot": (
                summary_df.copy()
            ),
            "ai_ready": False,
            "ai_fault_lot": None,
            "ai_cause": None,
            "ai_lot_correct": None,
            "ai_cause_correct": None,
            "ai_confidence": None,
            "ai_confidence_label": None,
            "ai_top_causes": [],
            "ai_composite_possible": False,
            "ai_abstained": None,
            "ai_prediction_table": None,
            "ai_signal_summary": [],
            "evidence": (
                evidence.strip()
            ),
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
        st.success(
            "진단을 제출했습니다. "
            "아래에서 실제 발생 조건을 먼저 확인하세요."
        )

# =========================================================
# 6. 진단 결과 비교
# =========================================================
st.subheader("6. 진단 결과")

if (
    st.session_state.last_result
    is None
):
    st.info(
        "진단을 제출하면 내가 선택한 답과 실제 정답을 "
        "항목별로 바로 비교할 수 있습니다."
    )
else:
    result = (
        st.session_state.last_result
    )

    def compare_lot_prediction(
        predicted_lot,
        actual_lot,
    ):
        if predicted_lot is None:
            return "탐지하지 못함"

        gap = (
            int(predicted_lot)
            - int(actual_lot)
        )

        if gap == 0:
            return "일치"
        if gap > 0:
            return (
                f"{gap}개 Lot 늦음"
            )
        return (
            f"{abs(gap)}개 Lot 빠름"
        )

    st.markdown(
        "### 내 답과 실제 정답"
    )
    st.caption(
        "실제 정답은 프로그램이 문제를 만들 때 정한 "
        "발생 조건 하나입니다."
    )

    comparison_rows = [
        {
            "항목": "이상 시작 Lot",
            "내가 제출한 답": (
                f"Lot {result['guessed_fault_lot']}"
            ),
            "실제 정답": (
                f"Lot {result['actual_fault_lot']}"
            ),
            "판정": (
                compare_lot_prediction(
                    result[
                        "guessed_fault_lot"
                    ],
                    result[
                        "actual_fault_lot"
                    ],
                )
            ),
        },
        {
            "항목": "주요 원인",
            "내가 제출한 답": (
                result["guessed_cause"]
            ),
            "실제 정답": (
                result["actual_cause"]
            ),
            "판정": (
                "일치"
                if result["cause_correct"]
                else "불일치"
            ),
        },
        {
            "항목": "추가 원인",
            "내가 제출한 답": (
                result[
                    "guessed_secondary_cause"
                ]
            ),
            "실제 정답": (
                result[
                    "actual_secondary_cause"
                ]
            ),
            "판정": (
                "일치"
                if result[
                    "secondary_correct"
                ]
                else "불일치"
            ),
        },
    ]

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        width="stretch",
        column_config={
            "항목": st.column_config.TextColumn(
                "항목",
                width="small",
            ),
            "내가 제출한 답": st.column_config.TextColumn(
                "내가 제출한 답",
                width="medium",
            ),
            "실제 정답": st.column_config.TextColumn(
                "실제 정답",
                width="medium",
            ),
            "판정": st.column_config.TextColumn(
                "판정",
                width="small",
            ),
        },
    )

    correct_count = sum([
        bool(result["lot_correct"]),
        bool(result["cause_correct"]),
        bool(result["secondary_correct"]),
    ])

    if correct_count == 3:
        st.success(
            "3개 항목이 모두 실제 정답과 일치합니다."
        )
    else:
        st.warning(
            f"3개 항목 중 {correct_count}개가 실제 정답과 일치합니다."
        )

    actual_primary_key = next(
        key
        for key, value
        in CAUSES.items()
        if value
        == result[
            "actual_cause"
        ]
    )

    actual_cause_items = [
        (
            "주요 원인",
            actual_primary_key,
        )
    ]

    if (
        result[
            "actual_secondary_cause"
        ]
        != "없음"
    ):
        actual_secondary_key = next(
            key
            for key, value
            in CAUSES.items()
            if value
            == result[
                "actual_secondary_cause"
            ]
        )
        actual_cause_items.append(
            (
                "추가 원인",
                actual_secondary_key,
            )
        )

    with st.expander(
        "정답 해설 보기",
        expanded=False,
    ):
        st.markdown(
            f"- **실제 이상 시작:** "
            f"Lot {result['actual_fault_lot']}"
        )
        st.markdown(
            f"- **주요 원인:** "
            f"{result['actual_cause']}"
        )
        st.markdown(
            f"- **추가 원인:** "
            f"{result['actual_secondary_cause']}"
        )

        st.divider()

        for cause_role, cause_key in (
            actual_cause_items
        ):
            answer_guide = (
                CAUSE_ANSWER_GUIDES[
                    cause_key
                ]
            )

            st.markdown(
                f"#### {cause_role}: "
                f"{CAUSES[cause_key]}"
            )
            st.markdown(
                "**판단 근거**"
            )
            st.write(
                f"Lot {result['actual_fault_lot']}부터 "
                f"{answer_guide['evidence']}"
            )

            st.markdown(
                "**추가로 확인할 항목**"
            )
            st.write(
                answer_guide[
                    "additional_check"
                ]
            )

            st.markdown(
                "**대응 방안**"
            )
            st.write(
                answer_guide[
                    "corrective_action"
                ]
            )

            st.caption(
                "난이도와 노이즈에 따라 일부 변화는 작게 보이거나 "
                "다른 변수의 변화와 겹쳐 나타날 수 있습니다."
            )

    with st.expander(
        "내가 작성한 내용 보기",
        expanded=False,
    ):
        st.markdown(
            "**판단 근거**"
        )
        st.write(
            result["evidence"]
        )

        st.markdown(
            "**추가로 확인할 항목**"
        )
        st.write(
            result[
                "additional_check"
            ]
        )

        st.markdown(
            "**대응 방안**"
        )
        st.write(
            result[
                "corrective_action"
            ]
        )

    with st.expander(
        "자동 분석 결과 비교 보기",
        expanded=result.get(
            "ai_ready",
            False,
        ),
    ):
        st.caption(
            "아래 결과는 정답이 아니라 실제 정답을 추정한 보조 분석입니다. "
            "규칙 기반 탐지는 사람이 정한 기준으로 이상 시점만 찾고, "
            "AI는 여러 공정 변수의 패턴을 학습해 이상 시점과 원인을 추정합니다."
        )

        rule_lot = result.get(
            "auto_detected_lot"
        )

        automatic_rows = [
            {
                "분석 방법": "규칙 기반 탐지",
                "탐지한 이상 시작 Lot": (
                    f"Lot {rule_lot}"
                    if rule_lot is not None
                    else "찾지 못함"
                ),
                "추정 원인": "원인 판단 안 함",
                "실제 정답과 비교": (
                    compare_lot_prediction(
                        rule_lot,
                        result[
                            "actual_fault_lot"
                        ],
                    )
                ),
            }
        ]

        if result.get(
            "ai_ready",
            False,
        ):
            ai_lot_comparison = (
                compare_lot_prediction(
                    result.get(
                        "ai_fault_lot"
                    ),
                    result[
                        "actual_fault_lot"
                    ],
                )
            )
            ai_cause_comparison = (
                "원인 일치"
                if result.get(
                    "ai_cause_correct",
                    False,
                )
                else "원인 불일치"
            )

            automatic_rows.append({
                "분석 방법": "AI 분석",
                "탐지한 이상 시작 Lot": (
                    f"Lot {result['ai_fault_lot']}"
                    if result.get(
                        "ai_fault_lot"
                    )
                    is not None
                    else "찾지 못함"
                ),
                "추정 원인": (
                    result.get(
                        "ai_cause"
                    )
                    or "특정하지 못함"
                ),
                "실제 정답과 비교": (
                    f"{ai_lot_comparison} · "
                    f"{ai_cause_comparison}"
                ),
            })
        else:
            automatic_rows.append({
                "분석 방법": "AI 분석",
                "탐지한 이상 시작 Lot": "분석 전",
                "추정 원인": "분석 전",
                "실제 정답과 비교": "분석 전",
            })

        st.dataframe(
            pd.DataFrame(
                automatic_rows
            ),
            hide_index=True,
            width="stretch",
        )

        if not result.get(
            "ai_ready",
            False,
        ):
            st.caption(
                "AI 분석은 선택사항입니다. 처음 실행할 때는 "
                "학습용 데이터와 모델을 준비하느라 시간이 걸릴 수 있습니다."
            )

            if st.button(
                "AI 분석 실행",
                type="primary",
                width="stretch",
                key=(
                    "run_ai_"
                    f"{result['timestamp']}_"
                    f"{result['case_seed']}"
                ),
            ):
                if (
                    st.session_state.ai_bundle
                    is None
                ):
                    with st.spinner(
                        "AI 학습용 데이터를 만들고 모델을 준비하고 있습니다..."
                    ):
                        st.session_state.ai_bundle = (
                            train_ai_model(
                                DEFAULT_AI_CASE_COUNT,
                                DEFAULT_AI_DATASET_SEED,
                                DEFAULT_AI_TRAINING_SCOPE,
                            )
                        )

                diagnosis_summary = (
                    result.get(
                        "summary_snapshot"
                    )
                )

                if (
                    diagnosis_summary is None
                    or diagnosis_summary.empty
                ):
                    st.error(
                        "진단 당시의 데이터를 불러오지 못했습니다. "
                        "새 문제에서 다시 진단해주세요."
                    )
                else:
                    with st.spinner(
                        "AI가 동일한 공정 데이터를 분석하고 있습니다..."
                    ):
                        ai_diagnosis = (
                            ai_diagnose_current_case(
                                st.session_state.ai_bundle,
                                diagnosis_summary,
                                result[
                                    "material_key"
                                ],
                                result[
                                    "difficulty_key"
                                ],
                            )
                        )

                    ai_fault_lot = (
                        ai_diagnosis[
                            "fault_lot"
                        ]
                        if ai_diagnosis
                        is not None
                        else None
                    )
                    ai_cause = (
                        ai_diagnosis[
                            "cause_label"
                        ]
                        if ai_diagnosis
                        is not None
                        else "AI가 분석 결과를 만들지 못함"
                    )

                    updated_result = (
                        result.copy()
                    )
                    updated_result.update({
                        "ai_ready": True,
                        "ai_fault_lot": (
                            ai_fault_lot
                        ),
                        "ai_cause": (
                            ai_cause
                        ),
                        "ai_lot_correct": (
                            ai_fault_lot
                            == result[
                                "actual_fault_lot"
                            ]
                        ),
                        "ai_cause_correct": (
                            ai_cause
                            == result[
                                "actual_cause"
                            ]
                        ),
                        "ai_confidence": (
                            ai_diagnosis[
                                "confidence"
                            ]
                            if ai_diagnosis
                            is not None
                            else None
                        ),
                        "ai_confidence_label": (
                            ai_diagnosis[
                                "confidence_label"
                            ]
                            if ai_diagnosis
                            is not None
                            else "미확인"
                        ),
                        "ai_top_causes": (
                            ai_diagnosis[
                                "top_causes"
                            ]
                            if ai_diagnosis
                            is not None
                            else []
                        ),
                        "ai_composite_possible": (
                            ai_diagnosis[
                                "composite_possible"
                            ]
                            if ai_diagnosis
                            is not None
                            else False
                        ),
                        "ai_abstained": (
                            ai_diagnosis.get(
                                "abstained",
                                False,
                            )
                            if ai_diagnosis
                            is not None
                            else True
                        ),
                        "ai_prediction_table": (
                            ai_diagnosis[
                                "prediction_table"
                            ]
                            if ai_diagnosis
                            is not None
                            else None
                        ),
                        "ai_signal_summary": (
                            build_ai_signal_summary(
                                summary=(
                                    diagnosis_summary
                                ),
                                material=result[
                                    "material_key"
                                ],
                                detected_lot=(
                                    ai_fault_lot
                                ),
                            )
                        ),
                    })

                    st.session_state.last_result = (
                        updated_result
                    )

                    for history_index in range(
                        len(
                            st.session_state.history
                        )
                        - 1,
                        -1,
                        -1,
                    ):
                        history_record = (
                            st.session_state.history[
                                history_index
                            ]
                        )
                        if (
                            history_record.get(
                                "timestamp"
                            )
                            == result[
                                "timestamp"
                            ]
                            and history_record.get(
                                "case_seed"
                            )
                            == result[
                                "case_seed"
                            ]
                        ):
                            st.session_state.history[
                                history_index
                            ] = updated_result
                            break

                    st.rerun()

        else:
            ai_confidence = result.get(
                "ai_confidence"
            )
            ai_confidence_text = (
                f"{ai_confidence * 100:.1f}%"
                if ai_confidence
                is not None
                else "확인할 수 없음"
            )

            st.markdown(
                "#### AI 세부 분석"
            )
            st.markdown(
                f"- **예측 신뢰도:** "
                f"{result.get('ai_confidence_label', '미확인')} "
                f"({ai_confidence_text})"
            )

            if result.get(
                "ai_abstained",
                False,
            ):
                st.warning(
                    "AI가 이상 가능성은 감지했지만 원인을 하나로 "
                    "특정할 만큼 확신하지 못했습니다."
                )

            ai_detail_col1, ai_detail_col2 = (
                st.columns(2)
            )

            with ai_detail_col1:
                st.markdown(
                    "##### AI의 원인 후보"
                )

                ai_top_causes = (
                    result.get(
                        "ai_top_causes",
                        [],
                    )
                )

                if ai_top_causes:
                    for rank, cause in enumerate(
                        ai_top_causes,
                        start=1,
                    ):
                        st.markdown(
                            f"{rank}. **{cause['cause_label']}** "
                            f"({cause['probability'] * 100:.1f}%)"
                        )
                else:
                    st.markdown(
                        "- 원인을 특정하지 못했습니다."
                    )

                if result.get(
                    "ai_composite_possible",
                    False,
                ):
                    st.info(
                        "두 원인의 확률이 비슷해 복합 이상 가능성이 있습니다."
                    )

            with ai_detail_col2:
                st.markdown(
                    "##### AI가 확인한 주요 변화"
                )

                if result.get(
                    "ai_signal_summary"
                ):
                    for index, signal in enumerate(
                        result[
                            "ai_signal_summary"
                        ],
                        start=1,
                    ):
                        st.markdown(
                            f"{index}. {signal}"
                        )
                else:
                    st.markdown(
                        "- 뚜렷한 단일 변화 신호를 정리하지 못했습니다."
                    )

            show_lot_predictions = st.toggle(
                "Lot별 AI 예측 결과 보기",
                value=False,
                key=(
                    "show_ai_lot_predictions_"
                    f"{result['timestamp']}_"
                    f"{result['case_seed']}"
                ),
            )

            if show_lot_predictions:
                ai_prediction_table = (
                    result[
                        "ai_prediction_table"
                    ].copy()
                    if result[
                        "ai_prediction_table"
                    ]
                    is not None
                    else pd.DataFrame()
                )

                for probability_column in [
                    "이상 확률",
                    "1순위 확률",
                    "2순위 확률",
                ]:
                    if (
                        probability_column
                        in ai_prediction_table.columns
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


# =========================================================
# 7. 풀이 기록
# =========================================================
st.subheader("7. 풀이 기록")
st.caption(
    "지금까지 제출한 문제의 정답률과 상세 결과를 확인합니다."
)

if st.session_state.history:
    history_df = pd.DataFrame(
        st.session_state.history
    )

    total_problems = len(
        history_df
    )
    user_lot_accuracy = (
        history_df[
            "lot_correct"
        ].fillna(False).mean()
        * 100
    )
    user_cause_accuracy = (
        history_df[
            "cause_correct"
        ].fillna(False).mean()
        * 100
    )
    user_secondary_accuracy = (
        history_df[
            "secondary_correct"
        ].fillna(False).mean()
        * 100
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = (
        st.columns(4)
    )

    summary_col1.metric(
        "푼 문제 수",
        total_problems,
    )
    summary_col2.metric(
        "이상 시작 Lot 정답률",
        f"{user_lot_accuracy:.1f}%",
    )
    summary_col3.metric(
        "주요 원인 정답률",
        f"{user_cause_accuracy:.1f}%",
    )
    summary_col4.metric(
        "추가 원인 정답률",
        f"{user_secondary_accuracy:.1f}%",
    )

    ai_history_df = history_df[
        history_df[
            "ai_ready"
        ].fillna(False)
    ]

    if not ai_history_df.empty:
        st.markdown(
            "#### AI 비교 결과"
        )
        st.caption(
            "AI 분석을 실행한 문제만 집계합니다."
        )

        ai_metric_col1, ai_metric_col2 = (
            st.columns(2)
        )

        ai_metric_col1.metric(
            "AI 이상 시작 Lot 일치율",
            (
                f"{ai_history_df['ai_lot_correct'].fillna(False).mean() * 100:.1f}%"
            ),
        )
        ai_metric_col2.metric(
            "AI 주요 원인 일치율",
            (
                f"{ai_history_df['ai_cause_correct'].fillna(False).mean() * 100:.1f}%"
            ),
        )

    user_history_rows = []
    ai_history_rows = []
    download_rows = []

    for record in (
        st.session_state.history
    ):
        lot_result = (
            "일치"
            if record.get(
                "lot_correct",
                False,
            )
            else "불일치"
        )
        cause_result = (
            "일치"
            if record.get(
                "cause_correct",
                False,
            )
            else "불일치"
        )
        secondary_result = (
            "일치"
            if record.get(
                "secondary_correct",
                False,
            )
            else "불일치"
        )

        ai_ready = bool(
            record.get(
                "ai_ready",
                False,
            )
        )
        ai_lot_value = (
            (
                f"Lot {record.get('ai_fault_lot')}"
                if record.get(
                    "ai_fault_lot"
                )
                is not None
                else "찾지 못함"
            )
            if ai_ready
            else "분석 안 함"
        )
        ai_cause_value = (
            (
                record.get(
                    "ai_cause"
                )
                or "특정하지 못함"
            )
            if ai_ready
            else "분석 안 함"
        )
        ai_lot_result = (
            (
                "일치"
                if record.get(
                    "ai_lot_correct",
                    False,
                )
                else "불일치"
            )
            if ai_ready
            else "분석 안 함"
        )
        ai_cause_result = (
            (
                "일치"
                if record.get(
                    "ai_cause_correct",
                    False,
                )
                else "불일치"
            )
            if ai_ready
            else "분석 안 함"
        )

        user_row = {
            "풀이 시각": record.get(
                "timestamp",
                "",
            ),
            "재료": record.get(
                "material",
                "",
            ),
            "난이도": record.get(
                "difficulty",
                "",
            ),
            "문제 번호": record.get(
                "case_seed",
                "",
            ),
            "실제 이상 Lot": (
                f"Lot {record.get('actual_fault_lot')}"
            ),
            "내가 고른 Lot": (
                f"Lot {record.get('guessed_fault_lot')}"
            ),
            "Lot 판정": lot_result,
            "실제 주요 원인": record.get(
                "actual_cause",
                "",
            ),
            "내가 고른 주요 원인": record.get(
                "guessed_cause",
                "",
            ),
            "주요 원인 판정": cause_result,
            "실제 추가 원인": record.get(
                "actual_secondary_cause",
                "없음",
            ),
            "내가 고른 추가 원인": record.get(
                "guessed_secondary_cause",
                "없음",
            ),
            "추가 원인 판정": secondary_result,
        }

        ai_row = {
            "풀이 시각": record.get(
                "timestamp",
                "",
            ),
            "문제 번호": record.get(
                "case_seed",
                "",
            ),
            "실제 이상 Lot": (
                f"Lot {record.get('actual_fault_lot')}"
            ),
            "AI 추정 Lot": ai_lot_value,
            "AI Lot 판정": ai_lot_result,
            "실제 주요 원인": record.get(
                "actual_cause",
                "",
            ),
            "AI 추정 원인": ai_cause_value,
            "AI 원인 판정": ai_cause_result,
        }

        user_history_rows.append(
            user_row
        )
        ai_history_rows.append(
            ai_row
        )
        download_rows.append({
            **user_row,
            "AI 추정 Lot": ai_lot_value,
            "AI Lot 판정": ai_lot_result,
            "AI 추정 원인": ai_cause_value,
            "AI 원인 판정": ai_cause_result,
        })

    with st.expander(
        "지금까지의 풀이 기록 보기",
        expanded=False,
    ):
        user_tab, ai_tab = st.tabs(
            [
                "내 풀이",
                "AI 비교",
            ]
        )

        with user_tab:
            st.dataframe(
                pd.DataFrame(
                    user_history_rows
                ),
                hide_index=True,
                width="stretch",
            )

        with ai_tab:
            st.dataframe(
                pd.DataFrame(
                    ai_history_rows
                ),
                hide_index=True,
                width="stretch",
            )

        st.download_button(
            "전체 풀이 기록 CSV 다운로드",
            data=csv_bytes(
                pd.DataFrame(
                    download_rows
                )
            ),
            file_name=(
                "problem_solving_history.csv"
            ),
            mime="text/csv",
        )

else:
    st.info(
        "아직 완료한 문제가 없습니다. 진단을 제출하면 풀이 기록이 저장됩니다."
    )

# =========================================================
# 8. AI 모델 및 프로젝트 정보
# =========================================================
st.subheader("8. AI 모델 및 프로젝트 정보")

with st.expander(
    "프로젝트를 어떻게 설계했는지 확인하기",
    expanded=False,
):
    st.markdown(
        """
        1. 실제 반도체 생산 데이터는 개인 프로젝트에서 확보하기 어렵기 때문에 스퍼터 공정의 관계를 단순화한 합성 데이터를 설계했습니다.
        2. 전원 출력, Ar 유량, 챔버 압력과 타겟 상태 변화가 증착률, 두께, 균일도와 예상 면저항에 영향을 주도록 계산 규칙을 만들었습니다.
        3. 사용자는 생성된 Lot 데이터를 보고 이상 시작 시점과 원인을 직접 진단합니다.
        4. 규칙 기반 탐지는 사람이 정한 관리 범위와 변화 기준으로 이상 시점만 찾습니다.
        5. AI는 랜덤 포레스트 분류 모델로 정상과 네 가지 이상 원인을 학습하고, 사용자의 진단과 별도로 원인을 예측합니다.
        6. 난이도 정보는 실제 공정에서 얻을 수 없는 힌트이므로 AI 입력 변수에서 제외하고 평가 구분에만 사용합니다.
        7. 최종 목적은 높은 정확도 자체가 아니라 초기 이상, 미지 원인과 유사한 공정 신호에서 AI가 언제 왜 틀리는지 확인하는 것입니다.
        """
    )

with st.expander(
    "AI가 어떻게 사용되는지 확인하기",
    expanded=False,
):
    st.markdown(
        """
        이 앱의 AI는 **힌트 제공용이 아니라 사용자와 별도로 데이터를 분석하는 비교 대상**입니다.

        - AI는 프로그램이 자동으로 생성한 여러 공정 문제를 학습합니다.
        - 쉬움·보통 학습 문제에서는 Lot 1~2가 정상 기준으로 제공되고, 어려움·전문가에서는 초기 이상도 포함됩니다.
        - 문제 난이도 정보 자체는 AI 입력 변수로 사용하지 않습니다.
        - 사용자가 진단을 제출하기 전에는 AI 결과를 표시하지 않습니다.
        - 진단 제출 후 내 답과 실제 정답을 먼저 비교하고, 규칙 기반 탐지와 AI는 보조 분석으로 확인합니다.
        - AI가 표시하는 주요 변화는 예측 시점의 두께, 공정 변수, 증착률, 균일도와 예상 면저항 변화입니다.
        - 첫 Lot부터 이상인 문제와 학습에 없던 원인을 별도 평가해 모델이 언제 틀리거나 판단을 유보하는지 확인합니다.
        - AI 성능은 합성 데이터에 대한 평가 결과이며 실제 생산 라인의 진단 성능을 의미하지 않습니다.
        """
    )

with st.expander(
    "AI 모델 세부 성능과 고급 설정",
    expanded=False,
):
    st.caption(
        "일반 사용자는 이 설정을 조작하지 않아도 됩니다. "
        "첫 진단을 제출하면 1,000개의 혼합 난이도 문제로 학습한 기본 모델이 자동으로 준비됩니다. "
        "난이도는 성능 구분에만 사용하고 AI 입력에는 넣지 않습니다."
    )

    ai_setting_col1, ai_setting_col2, ai_setting_col3, ai_setting_col4 = (
        st.columns(4)
    )

    with ai_setting_col1:
        ai_case_count = st.selectbox(
            "AI 학습용 문제 수",
            options=[
                500,
                1000,
                2000,
                3000,
            ],
            index=1,
        )

    with ai_setting_col2:
        training_scope = st.selectbox(
            "AI 학습 데이터 구성",
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
        ai_dataset_seed = (
            st.number_input(
                "학습 데이터 생성 번호",
                min_value=1,
                max_value=999_999_999,
                value=DEFAULT_AI_DATASET_SEED,
                step=1,
            )
        )

    with ai_setting_col4:
        st.write("")
        st.write("")
        train_ai_button = (
            st.button(
                "AI 모델 다시 학습",
                type="primary",
                width="stretch",
            )
        )

    if train_ai_button:
        with st.spinner(
            "학습용 문제를 생성하고 AI 모델을 다시 학습하고 있습니다..."
        ):
            st.session_state.ai_bundle = (
                train_ai_model(
                    int(
                        ai_case_count
                    ),
                    int(
                        ai_dataset_seed
                    ),
                    str(
                        training_scope
                    ),
                )
            )

    render_ai_model_details(
        st.session_state.ai_bundle
    )

    if (
        st.session_state.ai_bundle
        is not None
    ):
        st.download_button(
            "AI 학습 데이터 CSV 내려받기",
            data=csv_bytes(
                st.session_state.ai_bundle[
                    "dataset"
                ]
            ),
            file_name=(
                "sputter_ai_training_features_"
                f"{st.session_state.ai_bundle['case_count']}_problems_"
                f"{st.session_state.ai_bundle['training_scope']}.csv"
            ),
            mime="text/csv",
        )
