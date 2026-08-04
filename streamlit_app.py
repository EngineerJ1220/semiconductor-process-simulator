import random
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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

APP_VERSION = "v1.2.2"
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


def initialize_case(material="Al", seed=None):
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999_999)

    st.session_state.active_material = material
    st.session_state.case_seed = int(seed)
    st.session_state.rng = np.random.default_rng(int(seed))
    st.session_state.current_lot = 0
    st.session_state.cause_key = None
    st.session_state.fault_start_lot = None
    st.session_state.fault_sign = 1
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
    if st.session_state.current_lot < 2:
        return False, "정상 기준을 만들기 위해 Lot을 2개 먼저 생산하세요."

    if st.session_state.cause_key is not None:
        return False, "이번 Case에는 이미 이상이 예약되어 있습니다."

    rng = st.session_state.rng
    st.session_state.cause_key = str(
        rng.choice(list(CAUSES.keys()))
    )
    st.session_state.fault_start_lot = (
        st.session_state.current_lot
        + int(rng.integers(1, 3))
    )
    st.session_state.fault_sign = int(
        rng.choice([-1, 1])
    )

    return (
        True,
        "랜덤 이상이 예약되었습니다. "
        "원인과 시작 Lot은 진단 제출 전까지 숨겨집니다.",
    )


def produce_lot():
    material = st.session_state.active_material
    material_props = MATERIALS[material]
    recipe = get_recipe(material)
    rng = st.session_state.rng

    st.session_state.current_lot += 1
    lot = st.session_state.current_lot

    fault_active = (
        st.session_state.cause_key is not None
        and lot >= st.session_state.fault_start_lot
    )

    progress = (
        lot - st.session_state.fault_start_lot + 1
        if fault_active
        else 0
    )

    rows = []

    for wafer in range(1, WAFERS_PER_LOT + 1):
        power_actual = rng.normal(
            recipe["power_w"],
            recipe["power_w"] * 0.0028,
        )
        ar_flow_actual = rng.normal(
            recipe["ar_flow_sccm"],
            0.08,
        )
        pressure_actual = rng.normal(
            recipe["pressure_mtorr"],
            0.03,
        )
        time_actual = rng.normal(
            recipe["deposition_time_s"],
            recipe["deposition_time_s"] * 0.0006,
        )

        processed_wafer_count = (
            (lot - 1) * WAFERS_PER_LOT + wafer
        )
        target_usage = 80 + processed_wafer_count * 0.02

        hidden_rate_effect = 0.0
        hidden_uniformity_effect = 0.0
        hidden_resistivity_effect = 0.0

        if fault_active:
            cause = st.session_state.cause_key

            if cause == "power_drift":
                power_actual += (
                    st.session_state.fault_sign
                    * recipe["power_w"]
                    * 0.012
                    * progress
                )

            elif cause == "flow_drop":
                ar_flow_actual -= 1.1 * progress
                pressure_actual -= 0.08 * progress

            elif cause == "pressure_rise":
                pressure_actual += 0.35 * progress
                hidden_uniformity_effect += 0.5 * progress
                hidden_resistivity_effect += 0.025 * progress

            elif cause == "target_wear":
                hidden_rate_effect -= (
                    recipe["base_rate_nm_s"]
                    * material_props["wear_sensitivity"]
                    * progress
                )
                hidden_uniformity_effect += 0.35 * progress
                hidden_resistivity_effect += 0.018 * progress

        relative_rate = (
            1.0
            + 0.0014 * (power_actual - recipe["power_w"])
            + 0.0080 * (
                ar_flow_actual - recipe["ar_flow_sccm"]
            )
            - 0.0200 * (
                pressure_actual - recipe["pressure_mtorr"]
            )
        )

        deposition_rate = (
            recipe["base_rate_nm_s"] * relative_rate
            - recipe["base_rate_nm_s"]
            * 0.0004
            * (target_usage - 80)
            + hidden_rate_effect
            + rng.normal(
                0,
                recipe["base_rate_nm_s"] * 0.0025,
            )
        )

        deposition_rate = max(
            deposition_rate,
            recipe["base_rate_nm_s"] * 0.25,
        )

        thickness = (
            deposition_rate * time_actual
            + rng.normal(0, 0.16)
        )

        uniformity = (
            1.15
            + 0.55
            * abs(
                pressure_actual
                - recipe["pressure_mtorr"]
            )
            + hidden_uniformity_effect
            + rng.normal(0, 0.10)
        )

        film_factor = (
            THIN_FILM_RESISTIVITY_FACTOR
            * (1 + hidden_resistivity_effect)
            * (1 + 0.015 * max(uniformity - 1.15, 0))
            * rng.normal(1.0, 0.008)
        )

        sheet_resistance = (
            10
            * material_props["bulk_resistivity_uohm_cm"]
            * film_factor
            / max(thickness, 1.0)
        )

        rows.append({
            "material": material,
            "lot": lot,
            "wafer": wafer,
            "power_set_w": round(recipe["power_w"], 2),
            "power_actual_w": round(power_actual, 2),
            "ar_flow_set_sccm": round(
                recipe["ar_flow_sccm"],
                2,
            ),
            "ar_flow_actual_sccm": round(
                ar_flow_actual,
                2,
            ),
            "pressure_set_mtorr": round(
                recipe["pressure_mtorr"],
                3,
            ),
            "pressure_actual_mtorr": round(
                pressure_actual,
                3,
            ),
            "time_set_s": round(
                recipe["deposition_time_s"],
                2,
            ),
            "time_actual_s": round(
                time_actual,
                2,
            ),
            "target_usage_h": round(
                target_usage,
                2,
            ),
            "deposition_rate_nm_s": round(
                deposition_rate,
                4,
            ),
            "thickness_nm": round(
                thickness,
                2,
            ),
            "uniformity_pct": round(
                max(uniformity, 0.1),
                2,
            ),
            "estimated_sheet_resistance_ohm_sq": round(
                sheet_resistance,
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
    "재료 선택, 면저항 추정, Lot별 이상 진단"
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("업데이트 내역", expanded=False):
        st.markdown(
            """
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
            """
        )

material_options = list(MATERIALS.keys())

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

if (
    selected_material
    != st.session_state.active_material
):
    st.warning(
        "재료 선택이 변경되었습니다. "
        "아래의 '새 Case 초기화'를 눌러야 "
        "새 재료가 적용됩니다."
    )

active_material = (
    st.session_state.active_material
)
material_props = MATERIALS[active_material]
recipe = get_recipe(active_material)

st.subheader(
    f"Case 01 · {active_material} 박막 "
    "DC Magnetron Sputtering"
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

st.caption(material_props["description"])
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
        "원인과 시작 Lot은 숨겨져 있습니다."
    )

raw_df = st.session_state.raw_data
summary_df = summarize(raw_df)

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

metric_col1.metric(
    "현재 재료",
    active_material,
)
metric_col2.metric(
    "누적 생산 Lot",
    st.session_state.current_lot,
)
metric_col3.metric(
    "누적 Wafer",
    len(raw_df),
)
metric_col4.metric(
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
st.subheader("5. 이상 진단")

with st.form("diagnosis_form"):
    guess_col1, guess_col2 = st.columns(2)

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
            "예상 원인",
            list(CAUSES.values()),
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
        actual_lot = int(
            st.session_state.fault_start_lot
        )
        automatic_lot = automatic_abnormal_lot(
            summary_df
        )

        lot_correct = (
            int(guessed_lot) == actual_lot
        )
        cause_correct = (
            guessed_cause == actual_cause
        )

        result = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "material": active_material,
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
            "both_correct": (
                lot_correct and cause_correct
            ),
            "auto_detected_lot": automatic_lot,
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

    result_col1, result_col2, result_col3 = (
        st.columns(3)
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

    st.markdown(
        f"**재료:** {result['material']}"
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
st.subheader("6. 누적 진단 이력")

if st.session_state.history:
    history_df = pd.DataFrame(
        st.session_state.history
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
