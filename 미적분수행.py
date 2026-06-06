import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Matplotlib 한글 및 스타일 설정
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.unicode_minus"] = False

# 1. 페이지 설정
st.set_page_config(
    page_title="PK Simulator",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 커스텀 CSS 스타일 적용
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
html, body, .stApp { font-family: 'Noto Sans KR', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
.pk-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2d45 55%, #1a3a5c 100%);
    border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.pk-header h2 { margin: 0; color: #fff; font-size: 1.75rem; }
.pk-header p  { margin: 0.35rem 0 0; color: #7eb8e8; font-size: 0.88rem; }
.pk-header .badge {
    display: inline-block; background: rgba(126,184,232,0.15);
    border: 1px solid rgba(126,184,232,0.3); border-radius: 20px; padding: 0.15rem 0.7rem; font-size: 0.75rem; color: #7eb8e8; margin-right: 0.5rem;
}
.drug-card {
    border-radius: 14px; padding: 1.1rem 1.3rem; border-left: 5px solid; background: #fff; box-shadow: 0 2px 12px rgba(0,0,0,0.07); height: 100%;
}
.drug-card h4  { margin: 0 0 0.35rem; font-size: 1.05rem; }
.drug-card .sub { color: #666; font-size: 0.82rem; margin-bottom: 0.6rem; }
.drug-card p   { margin: 0.18rem 0; font-size: 0.83rem; color: #444; }
.alert { border-radius: 10px; padding: 0.65rem 1rem; margin-top: 0.7rem; font-size: 0.9rem; }
.alert-safe   { background:#edfaf1; border-left:4px solid #2ecc71; color:#1a6b3e; }
.alert-warn   { background:#fffbeb; border-left:4px solid #f39c12; color:#7d4a00; }
.alert-danger { background:#fff0f0; border-left:4px solid #e74c3c; color:#7a1515; }
.auc-grid { display:flex; gap:0.8rem; flex-wrap:wrap; margin-top:0.5rem; }
.auc-box {
    flex:1; min-width:140px; border-radius:12px; padding:0.9rem 1rem; text-align:center; background:#f8fafc; border:1px solid #e2e8f0;
}
.auc-box .aval { font-size:1.35rem; font-weight:700; }
.auc-box .albl { font-size:0.75rem; color:#718096; margin-top:0.15rem; }
.sec-title {
    font-size: 1.05rem; font-weight: 700; color: #2d3748; margin: 1.2rem 0 0.6rem; padding-bottom: 0.3rem; border-bottom: 2px solid #ebf0f7;
}
.param-tbl { width:100%; border-collapse:collapse; font-size:0.82rem; }
.param-tbl td { padding:0.3rem 0.6rem; border-bottom:1px solid #f0f0f0; }
.param-tbl td:first-child { color:#718096; }
.param-tbl td:last-child  { font-weight:600; color:#2d3748; }
</style>
""", unsafe_allow_html=True)

# 3. 데이터셋 정의
DRUGS = {
    "💉 Lidocaine (리도카인)": {
        "route": "IV", "Vc_kg": 0.52, "Vp_kg": 0.77, "k12": 1.65, "k21": 1.12, "kel0": 1.23, "ka": None, "F": 1.0,
        "dose_kg": 1.5, "t_end": 8.0, "dt": 0.001, "c_min": 1.5, "c_max": 5.0, "c_toxic": 9.0, "color": "#e74c3c",
        "desc_ko": "부정맥 치료제 / 국소마취제", "route_ko": "정맥주사 (IV Bolus)", "t_half_ko": "β기 반감기 ≈ 1.5~2시간",
        "note_ko": "경구 생체이용률 ~35% (초회통과효과 큼) → 반드시 IV",
    },
    "☕ Caffeine (카페인)": {
        "route": "PO", "Vc_kg": 0.60, "Vp_kg": 0.40, "k12": 0.30, "k21": 0.20, "kel0": 0.13, "ka": 2.0, "F": 1.0,
        "dose_kg": 3.0, "t_end": 24.0, "dt": 0.01, "c_min": None, "c_max": None, "c_toxic": 80.0, "color": "#7B3F00",
        "desc_ko": "중추신경계 자극제 / 아데노신 수용체 길항제", "route_ko": "경구투여 (PO)", "t_half_ko": "반감기 ≈ 3~5시간",
        "note_ko": "생체이용률 ~100%  |  간 CYP1A2 대사",
    },
}

LIVER = {
    "정상 (Normal)": {"factor": 1.0, "hex": "#27ae60"},
    "경도 장애 (Child-Pugh A)": {"factor": 0.70, "hex": "#f39c12"},
    "중등도 장애 (Child-Pugh B)": {"factor": 0.50, "hex": "#e67e22"},
    "중증 장애 (Child-Pugh C)": {"factor": 0.30, "hex": "#e74c3c"},
}

# 4. 핵심 시뮬레이션 및 분석 함수
def stimulate(drug: dict, weight: float, liver_factor: float, dose_mg: float) -> tuple:
    Vc = drug["Vc_kg"] * weight
    k12 = drug["k12"]
    k21 = drug["k21"]
    kel = drug["kel0"] * liver_factor
    dt = drug["dt"]
    t_end = drug["t_end"]
    N = int(t_end / dt) + 1

    t = np.linspace(0.0, t_end, N)
    Ac = np.zeros(N)
    Ap = np.zeros(N)
    Ag = np.zeros(N)

    if drug["route"] == "IV":
        Ac[0] = dose_mg
        for i in range(1, N):
            dAc = -(k12 + kel) * Ac[i-1] + k21 * Ap[i-1]
            dAp = k12 * Ac[i-1] - k21 * Ap[i-1]
            Ac[i] = max(0.0, Ac[i-1] + dAc * dt)
            Ap[i] = max(0.0, Ap[i-1] + dAp * dt)
    else:  # PO route
        ka = drug["ka"]
        F = drug["F"]
        Ag[0] = dose_mg * F
        for i in range(1, N):
            dAg = -ka * Ag[i-1]
            dAc = ka * Ag[i-1] - (k12 + kel) * Ac[i-1] + k21 * Ap[i-1]
            dAp = k12 * Ac[i-1] - k21 * Ap[i-1]
            Ag[i] = max(0.0, Ag[i-1] + dAg * dt)
            Ac[i] = max(0.0, Ac[i-1] + dAc * dt)
            Ap[i] = max(0.0, Ap[i-1] + dAp * dt)

    Cc = Ac / Vc
    return t, Ac, Ap, Cc, kel

def auc_total(t, Cc) -> float:
    return float(np.trapezoid(Cc, t))

def auc_zones(t, Cc, c_min, c_max) -> tuple:
    auc_th = float(np.trapezoid(np.where((Cc >= c_min) & (Cc <= c_max), Cc, 0), t))
    auc_sup = float(np.trapezoid(np.where(Cc > c_max, Cc - c_max, 0), t))
    auc_sub = float(np.trapezoid(np.where(Cc < c_min, Cc, 0), t))
    return auc_th, auc_sup, auc_sub

# 5. 사이드바 UI 설정
with st.sidebar:
    st.markdown("## ⚙️ 시뮬레이션 설정")
    st.divider()
    drug_name = st.selectbox("💊 약물 선택", list(DRUGS.keys()))
    drug = DRUGS[drug_name]
    st.divider()
    weight = st.slider("⚖️ 체중 (kg)", min_value=40, max_value=120, value=70, step=1)
    liver_name = st.selectbox("🏥 간 기능 상태", list(LIVER.keys()))
    st.divider()
    default_dose = drug["dose_kg"] * weight
    dose_mode = st.radio(
    "💉 용량 설정 방식",
    ["체중 기반 자동 계산", "직접 입력"],
    horizontal=True
    )

    if dose_mode == "체중 기반 자동 계산":
        dose = float(round(default_dose))
        st.info(f"현재 용량: {dose:.0f} mg = {drug['dose_kg']} mg/kg × {weight} kg")
    else:
        dose = st.number_input(
        "💉 투여 용량 (mg)",
        min_value=10.0,
        max_value=1000.0,
        value=float(round(default_dose)),
        step=5.0,
        help=f"기본값: {drug['dose_kg']} mg/kg × {weight} kg = {default_dose:.0f} mg",
    )
    st.divider()
    st.markdown("**📈 표시 옵션**")
    show_compartments = st.checkbox("Ac / Ap 구획별 약물량 표시", value=False)
    compare_mode = st.checkbox("간 기능 비교 모드 (4가지 비교)", value=False)
    log_scale = st.checkbox("로그 스케일 (Cc 축)", value=False)
    st.divider()
    st.caption("📐 수학적 방법")
    st.caption(f"• 오일러 방법  Δt = {drug['dt']} hr")
    st.caption("• 구분구적법 (사다리꼴 공식)")
    st.caption(f"• 시뮬레이션 시간: 0 → {drug['t_end']} hr")

# 6. 메인 헤더 레이아웃
st.markdown("""
<div class="pk-header">
  <h2>💊 2구획 약물동태학 시뮬레이터</h2>
  <p>
    <span class="badge">2-Compartment Open Model</span>
    <span class="badge">Euler's Method</span>
    <span class="badge">구분구적법 · AUC</span>
  </p>
</div>
""", unsafe_allow_html=True)

# 7. 데이터 계산 및 메트릭 바인딩
liver_info = LIVER[liver_name]
liver_factor = liver_info["factor"]

t, Ac, Ap, Cc, kel_eff = stimulate(drug, weight, liver_factor, dose)
auc = auc_total(t, Cc)
cmax = float(np.max(Cc))
tmax = float(t[np.argmax(Cc)])
t_half_eff = 0.693 / kel_eff if kel_eff > 0 else float('inf')

col_card, col_metrics = st.columns([2, 3], gap="medium")

with col_card:
    th_str = f"<p>🎯 <b>치료 범위:</b> {drug['c_min']}–{drug['c_max']} mg/L</p>" if drug["c_min"] else ""
    col = drug["color"]
    st.markdown(f"""
    <div class="drug-card" style="border-color:{col};">
      <h4 style="color:{col};">{drug_name}</h4>
      <div class="sub">{drug['desc_ko']}</div>
      <p>🚀 <b>투여 경로:</b> {drug['route_ko']}</p>
      <p>⏱ <b>{drug['t_half_ko']}</b></p>
      <p>🧪 <b>Vc:</b> {drug['Vc_kg']*weight:.1f} L &nbsp;|&nbsp; <b>Vp:</b> {drug['Vp_kg']*weight:.1f} L</p>
      <p>💊 <b>투여용량:</b> {dose:.0f} mg ({dose/weight:.2f} mg/kg)</p>
      {th_str}
      <p style="color:#888;font-style:italic;font-size:0.78rem;margin-top:0.5rem;">ℹ {drug['note_ko']}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("📈 Cmax", f"{cmax:.3f} mg/L")
    r1c2.metric("⏰ Tmax", f"{tmax:.2f} hr")
    r1c3.metric("📐 AUC (0→t_end)", f"{auc:.2f} mg·hr/L")

    r2c1, r2c2, r2c3 = st.columns(3)
    delta_lbl = f"×{liver_factor} 간 기능 저하" if liver_factor < 1 else "정상"
    delta_col = "inverse" if liver_factor < 1 else "off"
    r2c1.metric("🔁 kel (보정)", f"{kel_eff:.4f} /hr", delta=delta_lbl, delta_color=delta_col)
    r2c2.metric("⏳ 유효 반감기", f"{t_half_eff:.2f} hr")
    r2c3.metric("⚖️ Vc (체중보정)", f"{drug['Vc_kg']*weight:.1f} L")

# 8. 경고 메시지 출력
if drug["c_min"]:
    if cmax > drug["c_toxic"]:
        st.markdown(f'<div class="alert alert-danger">⚠️ <b>독성 위험:</b> Cmax <b>{cmax:.2f} mg/L</b>  ›  독성 한계 {drug["c_toxic"]} mg/L</div>', unsafe_allow_html=True)
    elif cmax > drug["c_max"]:
        st.markdown(f'<div class="alert alert-warn">⚠️ <b>과량 주의:</b> Cmax <b>{cmax:.2f} mg/L</b>  ›  치료 최대 {drug["c_max"]} mg/L</div>', unsafe_allow_html=True)
    elif cmax >= drug["c_min"]:
        st.markdown(f'<div class="alert alert-safe">✅ <b>치료 범위 내:</b> Cmax <b>{cmax:.2f} mg/L</b>  ∈  [{drug["c_min"]}, {drug["c_max"]}] mg/L</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert alert-warn">⚠️ <b>효과 미달:</b> Cmax <b>{cmax:.2f} mg/L</b>  ‹  최소 치료농도 {drug["c_min"]} mg/L</div>', unsafe_allow_html=True)

st.divider()

# 9. 시각화 그래프 섹션
st.markdown('<div class="sec-title">📉 혈중농도 (Cc) 시뮬레이션 결과</div>', unsafe_allow_html=True)

if compare_mode:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafbfc")
    auc_dict = {}

    for lname, ldata in LIVER.items():
        _, _, _, Cc_c, _ = stimulate(drug, weight, ldata["factor"], dose)
        auc_c = auc_total(t, Cc_c)
        auc_dict[lname] = auc_c
        short = lname.split("(")[0].strip()
        ax_l.plot(t, Cc_c, color=ldata["hex"], lw=2.2, label=f"{short}  |  AUC = {auc_c:.1f}")

    if drug["c_min"]:
        ax_l.axhspan(drug["c_min"], drug["c_max"], alpha=0.08, color="#2ecc71")
        ax_l.axhline(drug["c_min"], color="#27ae60", ls="--", lw=1.3, alpha=0.7)
        ax_l.axhline(drug["c_max"], color="#e67e22", ls="--", lw=1.3, alpha=0.7)

    ax_l.set(xlabel="Time (hr)", ylabel="Cc (mg/L)", title="간 기능에 따른 혈중농도 변화", xlim=(0, drug["t_end"]), ylim=(0, None))
    if log_scale: ax_l.set_yscale("log")
    ax_l.legend(fontsize=8.5); ax_l.grid(alpha=0.22)

    labels = [k.split("(")[0].strip() for k in auc_dict]
    values = list(auc_dict.values())
    colors_ = [LIVER[k]["hex"] for k in auc_dict]
    bars = ax_r.bar(range(len(labels)), values, color=colors_, alpha=0.88, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax_r.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.012, f"{val:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax_r.set_xticks(range(len(labels)))
    ax_r.set_xticklabels(labels, rotation=12, ha="right", fontsize=8.5)
    ax_r.set(xlabel="간 기능 상태", ylabel="AUC (mg·hr/L)", title="간 기능에 따른 AUC 비교")
    ax_r.grid(alpha=0.22, axis="y")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)

else:
    n_rows = 2 if show_compartments else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 * n_rows), facecolor="#fafbfc")
    ax = axes[0] if n_rows > 1 else axes

    if drug["c_min"]:
        c_min_d, c_max_d = drug["c_min"], drug["c_max"]
        ax.fill_between(t, 0, np.minimum(Cc, c_min_d), color="#a0aec0", alpha=0.18, label="Sub-therapeutic")
        ax.fill_between(t, c_min_d, np.clip(Cc, c_min_d, c_max_d), where=Cc >= c_min_d, color="#27ae60", alpha=0.22, label=f"Therapeutic  ({c_min_d}–{c_max_d} mg/L)")
        ax.fill_between(t, c_max_d, Cc, where=Cc > c_max_d, color="#f39c12", alpha=0.28, label="Supra-therapeutic")
        ax.axhline(c_min_d, color="#27ae60", ls="--", lw=1.3, alpha=0.75)
        ax.axhline(c_max_d, color="#e67e22", ls="--", lw=1.3, alpha=0.75)
        ax.axhline(drug["c_toxic"], color="#e74c3c", ls=":", lw=1.3, alpha=0.65, label=f"Toxic  ({drug['c_toxic']} mg/L)")
    else:
        ax.fill_between(t, 0, Cc, color=drug["color"], alpha=0.15, label="AUC")

    ax.plot(t, Cc, color=drug["color"], lw=2.5, label="Cc  (Central Concentration)")
    ax.annotate(f"Cmax = {cmax:.3f} mg/L\n(t = {tmax:.2f} hr)", xy=(tmax, cmax), xytext=(tmax + drug["t_end"] * 0.07, cmax * 0.80),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.92))
    # 약물별 축 범위 조정
    if "Lidocaine" in drug_name:
        x_limit = (0, 4) 
        if log_scale:
            y_limit = (0.1, 12)
        else:
            y_limit = (0, 10)
    else:
        x_limit = (0, drug["t_end"])
        y_limit = (None if log_scale else 0)
    
    plot_drug_name = drug_name.replace("💉 ", "").replace("☕ ", "")

    ax.set(
        xlabel="Time (hr)",
        ylabel="Cc (mg/L)",
        title=(f"{plot_drug_name}  |  2-Compartment Open Model\nWeight: {weight} kg  |  {liver_name}  |  AUC = {auc:.2f} mg·hr/L"),
        xlim=x_limit,
        ylim=y_limit
    )
    if log_scale: ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.22)

    if show_compartments:
        ax2 = axes[1]
        ax2.plot(t, Ac, color="#3182ce", lw=2.3, label="Ac  (Central compartment — mg)")
        ax2.plot(t, Ap, color="#9f7aea", lw=2.3, ls="--", label="Ap  (Peripheral compartment — mg)")
        ax2.set(xlabel="Time (hr)", ylabel="Amount (mg)", title="Ac & Ap  |  구획별 약물량 변화", xlim=(0, drug["t_end"]), ylim=(0, None))
        ax2.legend(fontsize=9); ax2.grid(alpha=0.22)

    plt.tight_layout(pad=1.8)
    st.pyplot(fig)

# 10. AUC 하단 분석 결과 및 수학적 원리 정보
st.markdown('<div class="sec-title">📊 AUC 분석 — 구분구적법 결과</div>', unsafe_allow_html=True)

if drug["c_min"]:
    auc_th, auc_sup, auc_sub = auc_zones(t, Cc, drug["c_min"], drug["c_max"])
    eff_pct = min(100.0, auc_th / auc * 100) if auc > 0 else 0.0
    st.markdown(f"""
    <div class="auc-grid">
      <div class="auc-box"><div class="aval" style="color:#2d3748;">{auc:.2f}</div><div class="albl">총 AUC (mg·hr/L)</div></div>
      <div class="auc-box"><div class="aval" style="color:#27ae60;">{auc_th:.2f}</div><div class="albl">✅ 치료 구간 AUC</div></div>
      <div class="auc-box"><div class="aval" style="color:#e67e22;">{auc_sup:.2f}</div><div class="albl">⚠️ 과량 구간 AUC</div></div>
      <div class="auc-box"><div class="aval" style="color:#718096;">{auc_sub:.2f}</div><div class="albl">📉 치료 미달 AUC</div></div>
      <div class="auc-box"><div class="aval" style="color:#3182ce;">{eff_pct:.1f}%</div><div class="albl">🎯 치료 효율</div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="auc-grid">
      <div class="auc-box"><div class="aval" style="color:#2d3748;">{auc:.2f}</div><div class="albl">총 AUC (mg·hr/L)</div></div>
      <div class="auc-box"><div class="aval" style="color:#7B3F00;">{cmax:.3f}</div><div class="albl">Cmax (mg/L)</div></div>
      <div class="auc-box"><div class="aval" style="color:#555;">{t_half_eff:.2f} hr</div><div class="albl">유효 반감기</div></div>
      <div class="auc-box"><div class="aval" style="color:#3182ce;">{tmax:.2f} hr</div><div class="albl">Tmax</div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

with st.expander("📐 수학적 원리 보기 (오일러 방법 + 구분구적법)"):
    eq_col, param_col = st.columns([3, 2], gap="large")
    with eq_col:
        st.markdown("**2구획 미분방정식**")
        if drug["route"] == "IV":
            st.latex(r"\frac{dA_c}{dt} = -(k_{12}+k_{el})\,A_c + k_{21}\,A_p")
            st.latex(r"\frac{dA_p}{dt} = k_{12}\,A_c - k_{21}\,A_p")
        else:
            st.latex(r"\frac{dA_g}{dt} = -k_a\,A_g")
            st.latex(r"\frac{dA_c}{dt} = k_a\,A_g -(k_{12}+k_{el})\,A_c + k_{21}\,A_p")
            st.latex(r"\frac{dA_p}{dt} = k_{12}\,A_c - k_{21}\,A_p")
        st.markdown("**오일러 점화식 (컴퓨터 근사)**")
        st.latex(r"A_c[i{+}1] = A_c[i] + \frac{dA_c}{dt}\bigg|_i \cdot \Delta t")
        st.latex(r"C_c[i] = \frac{A_c[i]}{V_c}")
        st.markdown("**구분구적법 — AUC (사다리꼴 공식)**")
        st.latex(r"AUC \;=\; \sum_{i=0}^{N-1} \frac{C_c[i]+C_c[i{+}1]}{2}\,\Delta t")
    with param_col:
        st.markdown("**현재 파라미터**")
        ka_row = f"<tr><td>ka</td><td>{drug['ka']} /hr</td></tr>" if drug["ka"] else ""
        st.markdown(f"""
        <table class="param-tbl">
          <tr><td>Δt</td><td>{drug['dt']} hr</td></tr>
          <tr><td>k12</td><td>{drug['k12']} /hr</td></tr>
          <tr><td>k21</td><td>{drug['k21']} /hr</td></tr>
          {ka_row}
          <tr><td>kel₀</td><td>{drug['kel0']} /hr</td></tr>
          <tr><td>kel (보정)</td><td>{kel_eff:.4f} /hr</td></tr>
          <tr><td>간기능 계수</td><td>× {liver_factor}</td></tr>
          <tr><td>Vc</td><td>{drug['Vc_kg']*weight:.2f} L</td></tr>
          <tr><td>Vp</td><td>{drug['Vp_kg']*weight:.2f} L</td></tr>
          <tr><td>체중</td><td>{weight} kg</td></tr>
          <tr><td>투여용량</td><td>{dose:.0f} mg</td></tr>
        </table>
        """, unsafe_allow_html=True)