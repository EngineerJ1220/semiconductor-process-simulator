import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="가상 반도체 박막 공정 시뮬레이터",
    page_icon="🧪",
    layout="wide",
)

WAFERS_PER_LOT = 25
TARGET_THICKNESS = 100.0
LOWER_SPEC = 98.0
UPPER_SPEC = 102.0
UNIFORMITY_LIMIT = 3.0

SETPOINTS = {
    "power_w": 500.0,
    "ar_flow_sccm": 30.0,
    "pressure_mtorr": 5.0,
    "deposition_time_s": 100.0,
}

CAUSES = {
    "power_drift": "전원 공급계통 출력 드리프트",
    "flow_drop": "Ar 유량 제어 이상",
    "pressure_rise": "챔버 압력 제어 이상",
    "target_wear": "타겟 국부 소모",
}

CAUSE_EXPLANATIONS = {
    "power_drift": "Actual Power가 Lot 진행에 따라 변하고 증착률과 평균 두께도 같은 방향으로 이동합니다.",
    "flow_drop": "Actual Ar Flow와 Pressure가 함께 낮아지고 증착률과 평균 두께가 감소합니다.",
    "pressure_rise": "Actual Pressure가 상승하면서 증착률과 평균 두께가 감소하고 균일도가 악화됩니다.",
    "target_wear": "주요 제어값은 Setpoint 부근이지만 증착률이 서서히 낮아지고 균일도가 악화됩니다.",
}

RAW_COLUMNS = [
    "lot", "wafer",
    "power_set_w", "power_actual_w",
    "ar_flow_set_sccm", "ar_flow_actual_sccm",
    "pressure_set_mtorr", "pressure_actual_mtorr",
    "time_set_s", "time_actual_s",
    "target_usage_h", "deposition_rate_nm_s",
    "thickness_nm", "uniformity_pct",
]

SUMMARY_COLUMNS = [
    "lot", "mean_thickness_nm", "std_thickness_nm",
    "mean_uniformity_pct", "oos_count",
    "mean_power_w", "mean_ar_flow_sccm",
    "mean_pressure_mtorr", "mean_rate_nm_s",
]


def empty_raw():
    return pd.DataFrame(columns=RAW_COLUMNS)


def empty_summary():
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def initialize_case(seed=None):
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999_999)

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
        df.groupby("lot")
        .agg(
            mean_thickness_nm=("thickness_nm", "mean"),
            std_thickness_nm=("thickness_nm", "std"),
            mean_uniformity_pct=("uniformity_pct", "mean"),
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
        .groupby("lot")["failed"]
        .sum()
        .reset_index(name="oos_count")
    )

    summary = summary.merge(oos, on="lot")
    numeric_cols = [c for c in SUMMARY_COLUMNS if c not in ("lot", "oos_count")]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary[SUMMARY_COLUMNS]


def inject_fault():
    if st.session_state.current_lot < 2:
        return False, "정상 기준을 만들기 위해 Lot을 2개 먼저 생산하세요."

    if st.session_state.cause_key is not None:
        return False, "이번 Case에는 이미 이상이 예약되어 있습니다."

    rng = st.session_state.rng
    st.session_state.cause_key = str(rng.choice(list(CAUSES.keys())))
    st.session_state.fault_start_lot = (
        st.session_state.current_lot + int(rng.integers(1, 3))
    )
    st.session_state.fault_sign = int(rng.choice([-1, 1]))
    return True, "랜덤 이상이 예약되었습니다. 원인과 시작 Lot은 진단 제출 전까지 숨겨집니다."


def produce_lot():
    rng = st.session_state.rng
    st.session_state.current_lot += 1
    lot = st.session_state.current_lot

    fault_active = (
        st.session_state.cause_key is not None
        and lot >= st.session_state.fault_start_lot
    )
    progress = (
        lot - st.session_state.fault_start_lot + 1
        if fault_active else 0
    )

    rows = []

    for wafer in range(1, WAFERS_PER_LOT + 1):
        power_actual = rng.normal(SETPOINTS["power_w"], 1.4)
        ar_flow_actual = rng.normal(SETPOINTS["ar_flow_sccm"], 0.08)
        pressure_actual = rng.normal(SETPOINTS["pressure_mtorr"], 0.03)
        time_actual = rng.normal(SETPOINTS["deposition_time_s"], 0.06)

        processed_wafer_count = (lot - 1) * WAFERS_PER_LOT + wafer
        target_usage = 80 + processed_wafer_count * 0.02

        hidden_rate_effect = 0.0
        hidden_uniformity_effect = 0.0

        if fault_active:
            cause = st.session_state.cause_key

            if cause == "power_drift":
                power_actual += st.session_state.fault_sign * 6.0 * progress

            elif cause == "flow_drop":
                ar_flow_actual -= 1.1 * progress
                pressure_actual -= 0.08 * progress

            elif cause == "pressure_rise":
                pressure_actual += 0.35 * progress
                hidden_uniformity_effect += 0.5 * progress

            elif cause == "target_wear":
                hidden_rate_effect -= 0.008 * progress
                hidden_uniformity_effect += 0.35 * progress

        deposition_rate = (
            1.0
            + 0.0014 * (power_actual - SETPOINTS["power_w"])
            + 0.0080 * (ar_flow_actual - SETPOINTS["ar_flow_sccm"])
            - 0.0200 * (pressure_actual - SETPOINTS["pressure_mtorr"])
            - 0.0004 * (target_usage - 80)
            + hidden_rate_effect
            + rng.normal(0, 0.0025)
        )

        thickness = deposition_rate * time_actual + rng.normal(0, 0.16)

        uniformity = (
            1.15
            + 0.55 * abs(pressure_actual - SETPOINTS["pressure_mtorr"])
            + hidden_uniformity_effect
            + rng.normal(0, 0.10)
        )

        rows.append({
            "lot": lot,
            "wafer": wafer,
            "power_set_w": SETPOINTS["power_w"],
            "power_actual_w": round(power_actual, 2),
            "ar_flow_set_sccm": SETPOINTS["ar_flow_sccm"],
            "ar_flow_actual_sccm": round(ar_flow_actual, 2),
            "pressure_set_mtorr": SETPOINTS["pressure_mtorr"],
            "pressure_actual_mtorr": round(pressure_actual, 3),
            "time_set_s": SETPOINTS["deposition_time_s"],
            "time_actual_s": round(time_actual, 2),
            "target_usage_h": round(target_usage, 2),
            "deposition_rate_nm_s": round(deposition_rate, 4),
            "thickness_nm": round(thickness, 2),
            "uniformity_pct": round(max(uniformity, 0.1), 2),
        })

    new_df = pd.DataFrame(rows)
    st.session_state.raw_data = pd.concat(
        [st.session_state.raw_data, new_df],
        ignore_index=True,
    )
    return new_df


def automatic_abnormal_lot(summary):
    if len(summary) < 3:
        return None

    baseline = summary.iloc[:2]
    thickness_center = baseline["mean_thickness_nm"].mean()
    thickness_std = max(baseline["mean_thickness_nm"].std(ddof=1), 0.12)
    power_center = baseline["mean_power_w"].mean()
    flow_center = baseline["mean_ar_flow_sccm"].mean()
    pressure_center = baseline["mean_pressure_mtorr"].mean()
    rate_center = baseline["mean_rate_nm_s"].mean()

    for _, row in summary.iloc[2:].iterrows():
        signals = [
            abs(row["mean_thickness_nm"] - thickness_center) > max(3 * thickness_std, 0.8),
            abs(row["mean_power_w"] - power_center) > 4.5,
            abs(row["mean_ar_flow_sccm"] - flow_center) > 0.7,
            abs(row["mean_pressure_mtorr"] - pressure_center) > 0.20,
            abs(row["mean_rate_nm_s"] - rate_center) > 0.006,
            row["mean_uniformity_pct"] > 2.0,
        ]
        if sum(signals) >= 2:
            return int(row["lot"])

    return None


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


ensure_state()

st.title("가상 반도체 박막 공정 이상 진단 시뮬레이터")
st.caption("교육용 가상 데이터를 이용해 Sputter 공정의 이상 Lot과 원인을 추론하는 웹 애플리케이션입니다.")

with st.expander("프로젝트 안내와 한계"):
    st.markdown(
        """
        - 실제 기업 Recipe나 생산 데이터를 사용하지 않은 교육용 시뮬레이션입니다.
        - 공정 변수와 결과의 관계는 학습 목적에 맞게 단순화했습니다.
        - 데이터 생성 규칙과 진단 결과를 분리해 사용자가 정답을 모른 채 분석하도록 설계했습니다.
        """
    )

st.subheader("Case 01 · Al 박막 DC Magnetron Sputtering")

left, right = st.columns(2)

with left:
    st.markdown(
        f"""
        **생산 조건**

        - 기판: 300 mm Si Wafer
        - Lot 크기: {WAFERS_PER_LOT}장
        - 증착막: Al
        - 목표 두께: {TARGET_THICKNESS:.0f} nm
        - 두께 관리 범위: {LOWER_SPEC:.0f}~{UPPER_SPEC:.0f} nm
        - 균일도 기준: {UNIFORMITY_LIMIT:.1f}% 이하
        """
    )

with right:
    st.dataframe(
        pd.DataFrame({
            "공정 변수": [
                "DC Power",
                "Ar Flow",
                "Chamber Pressure",
                "Deposition Time",
            ],
            "Setpoint": [
                "500 W",
                "30 sccm",
                "5.0 mTorr",
                "100 s",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

seed_input = st.number_input(
    "재현용 Case Seed",
    min_value=1,
    max_value=999_999_999,
    value=int(st.session_state.case_seed),
    step=1,
)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("새 Case 초기화", use_container_width=True):
        initialize_case(int(seed_input))
        st.rerun()

with c2:
    if st.button("다음 Lot 생산", type="primary", use_container_width=True):
        new_lot = produce_lot()
        oos = int(
            (
                (new_lot["thickness_nm"] < LOWER_SPEC)
                | (new_lot["thickness_nm"] > UPPER_SPEC)
            ).sum()
        )
        uniformity_fail = int(
            (new_lot["uniformity_pct"] > UNIFORMITY_LIMIT).sum()
        )
        st.success(
            f"Lot {st.session_state.current_lot} 생산 완료 · "
            f"두께 이탈 {oos}장 · 균일도 초과 {uniformity_fail}장"
        )

with c3:
    if st.button("랜덤 이상 예약", use_container_width=True):
        ok, message = inject_fault()
        if ok:
            st.success(message)
        else:
            st.warning(message)

if st.session_state.cause_key is None:
    st.info("진행 순서: 정상 Lot 2개 생산 → 랜덤 이상 예약 → Lot 4~5개 추가 생산 → 진단 제출")
else:
    st.info("이상이 예약되어 있습니다. 원인과 시작 Lot은 숨겨져 있습니다.")

raw_df = st.session_state.raw_data
summary_df = summarize(raw_df)

m1, m2, m3 = st.columns(3)
m1.metric("누적 생산 Lot", st.session_state.current_lot)
m2.metric("누적 Wafer", len(raw_df))
m3.metric("Case Seed", st.session_state.case_seed)

if not summary_df.empty:
    st.subheader("1. 두께 추이")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(raw_df["lot"], raw_df["thickness_nm"], alpha=0.45, label="Wafer")
    ax.plot(
        summary_df["lot"],
        summary_df["mean_thickness_nm"],
        marker="o",
        linewidth=2,
        label="Lot mean",
    )
    ax.axhline(TARGET_THICKNESS, linestyle="--", label="Target")
    ax.axhline(LOWER_SPEC, linestyle=":", label="LSL")
    ax.axhline(UPPER_SPEC, linestyle=":", label="USL")
    ax.set_xlabel("Lot")
    ax.set_ylabel("Thickness (nm)")
    ax.set_xticks(summary_df["lot"])
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.subheader("2. Setpoint 대비 공정 변수 변화")

    process_dev = pd.DataFrame({
        "Lot": summary_df["lot"],
        "Power (%)": (
            (summary_df["mean_power_w"] - SETPOINTS["power_w"])
            / SETPOINTS["power_w"]
            * 100
        ),
        "Ar Flow (%)": (
            (summary_df["mean_ar_flow_sccm"] - SETPOINTS["ar_flow_sccm"])
            / SETPOINTS["ar_flow_sccm"]
            * 100
        ),
        "Pressure (%)": (
            (summary_df["mean_pressure_mtorr"] - SETPOINTS["pressure_mtorr"])
            / SETPOINTS["pressure_mtorr"]
            * 100
        ),
    }).set_index("Lot")

    st.line_chart(process_dev)

    st.subheader("3. Lot 요약")
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with st.expander("4. Wafer별 Raw Data"):
        st.dataframe(raw_df, hide_index=True, use_container_width=True)

    st.download_button(
        "Raw Data CSV 다운로드",
        data=csv_bytes(raw_df),
        file_name=f"sputter_raw_seed_{st.session_state.case_seed}.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("5. 이상 진단")

with st.form("diagnosis_form"):
    g1, g2 = st.columns(2)

    with g1:
        guessed_lot = st.number_input(
            "최초 이상 Lot",
            min_value=1,
            max_value=max(st.session_state.current_lot, 1),
            value=1,
            step=1,
        )

    with g2:
        guessed_cause = st.selectbox(
            "예상 원인",
            list(CAUSES.values()),
        )

    evidence = st.text_area(
        "판단 근거",
        placeholder="Lot별 Actual 변수, 증착률, 두께, 균일도의 변화를 연결해 작성하세요.",
        height=110,
    )

    additional_check = st.text_area(
        "추가 확인 항목",
        placeholder="예: 장비 로그, 센서 교정 이력, 밸브 상태",
        height=90,
    )

    corrective_action = st.text_area(
        "조치 방안",
        placeholder="예: 장비 Hold, 관련 계통 점검, Monitor Wafer 재검증",
        height=90,
    )

    submit = st.form_submit_button(
        "진단 제출 및 정답 비교",
        use_container_width=True,
    )

if submit:
    errors = []

    if st.session_state.cause_key is None:
        errors.append("먼저 랜덤 이상을 예약해야 합니다.")

    if (
        st.session_state.fault_start_lot is not None
        and st.session_state.current_lot < st.session_state.fault_start_lot
    ):
        errors.append("이상이 데이터에 나타날 때까지 Lot을 더 생산해야 합니다.")

    if len(evidence.strip()) < 30:
        errors.append("판단 근거를 30자 이상 작성하세요.")

    if len(additional_check.strip()) < 15:
        errors.append("추가 확인 항목을 15자 이상 작성하세요.")

    if len(corrective_action.strip()) < 15:
        errors.append("조치 방안을 15자 이상 작성하세요.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        actual_key = st.session_state.cause_key
        actual_cause = CAUSES[actual_key]
        actual_lot = int(st.session_state.fault_start_lot)
        auto_lot = automatic_abnormal_lot(summary_df)

        lot_correct = int(guessed_lot) == actual_lot
        cause_correct = guessed_cause == actual_cause

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case_seed": st.session_state.case_seed,
            "actual_fault_lot": actual_lot,
            "guessed_fault_lot": int(guessed_lot),
            "lot_correct": lot_correct,
            "actual_cause": actual_cause,
            "guessed_cause": guessed_cause,
            "cause_correct": cause_correct,
            "both_correct": lot_correct and cause_correct,
            "auto_detected_lot": auto_lot,
            "evidence": evidence.strip(),
            "additional_check": additional_check.strip(),
            "corrective_action": corrective_action.strip(),
        }

        st.session_state.history.append(result)
        st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result
    actual_key = next(
        key for key, value in CAUSES.items()
        if value == result["actual_cause"]
    )

    st.subheader("진단 결과")

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "이상 Lot",
        "일치" if result["lot_correct"] else "불일치",
        f"정답 Lot {result['actual_fault_lot']}",
    )
    r2.metric(
        "이상 원인",
        "일치" if result["cause_correct"] else "불일치",
        result["actual_cause"],
    )
    r3.metric(
        "통계 기반 자동 탐지",
        (
            f"Lot {result['auto_detected_lot']}"
            if result["auto_detected_lot"] is not None
            else "탐지 실패"
        ),
    )

    st.markdown(f"**정답 패턴:** {CAUSE_EXPLANATIONS[actual_key]}")
    st.markdown(f"**작성한 판단 근거:** {result['evidence']}")
    st.markdown(f"**추가 확인 항목:** {result['additional_check']}")
    st.markdown(f"**조치 방안:** {result['corrective_action']}")

st.divider()
st.subheader("6. 누적 진단 이력")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    total_cases = len(history_df)
    lot_accuracy = history_df["lot_correct"].mean() * 100
    cause_accuracy = history_df["cause_correct"].mean() * 100
    both_accuracy = history_df["both_correct"].mean() * 100

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("완료 Case", total_cases)
    h2.metric("이상 Lot 정확도", f"{lot_accuracy:.1f}%")
    h3.metric("원인 정확도", f"{cause_accuracy:.1f}%")
    h4.metric("동시 적중률", f"{both_accuracy:.1f}%")

    st.dataframe(history_df, hide_index=True, use_container_width=True)

    st.download_button(
        "진단 이력 CSV 다운로드",
        data=csv_bytes(history_df),
        file_name="diagnosis_history.csv",
        mime="text/csv",
    )
else:
    st.caption("아직 제출된 진단 이력이 없습니다.")
