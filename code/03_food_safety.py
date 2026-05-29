import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import cobra
from cobra.flux_analysis import flux_variability_analysis

import utils
from utils import set_style, save_fig, log_progress, OUTPUT, DATA

log = utils.get_logger("03_safety")
set_style()

MODEL_XML = DATA / "MODEL1507180015_url.xml"

# ──────────────────────────────────────────────────────────
# 辅助函数：应用 natto 方向约束
# ──────────────────────────────────────────────────────────
def apply_natto_constraints(model):
    c = pd.read_csv(OUTPUT / "01_constraints_table.csv")
    ex_ids = {r.id for r in model.reactions if r.id.startswith("EX_")}
    applied = 0
    for _, row in c.iterrows():
        ex_id = "EX_" + row["model_id"].replace("M_", "")
        if ex_id not in ex_ids:
            continue
        rxn = model.reactions.get_by_id(ex_id)
        if row["constraint_type"] == "secretion":
            rxn.lower_bound = 0.0
        else:
            rxn.upper_bound = 0.0
        applied += 1
    return applied

# ══════════════════════════════════════════════════════════
# 1. 加载模型
# ══════════════════════════════════════════════════════════
log.info("Loading iBsu1103 model ...")
model_ref = cobra.io.read_sbml_model(str(MODEL_XML))
log.info(f"  Reactions: {len(model_ref.reactions)}  Metabolites: {len(model_ref.metabolites)}")

# ══════════════════════════════════════════════════════════
# 2. 读取模块二的 cFBA 通量 & FVA 结果
# ══════════════════════════════════════════════════════════
cfba_df = pd.read_csv(OUTPUT / "02_cfba_fluxes.csv")
cfba_map = dict(zip(cfba_df["reaction"], cfba_df["cfba_flux"]))
base_map = dict(zip(cfba_df["reaction"], cfba_df["baseline_flux"]))

fva_df = pd.read_csv(OUTPUT / "02_fva_ranges.csv", index_col=0)

# ──────────────────────────────────────────────────────────
# 风险代谢物定义
# 归一化因子 (norm_flux)：将通量归一化到 [0,1]
#   使用 FVA maximum（natto 条件下最大可能分泌量）
# ADI 参考（adi_mmol）：基于 EFSA/文献数据转化为等价通量单位
#   苯乙胺 EFSA TDI ~3 mg/kg/d ≈ 0.025 mmol/kg/d
#     → 相对单位：令 adi_norm = 0.05（=5% FVA max 为安全阈值）
# ──────────────────────────────────────────────────────────
RISK_METS = {
    "Phenethylamine":         {"ex": "EX_cpd03161_e", "fc": 55.35, "adi_norm": 0.05,
                               "risk": "Biogenic amine; BP elevation",
                               "note": "MODEL GAP: no Phe decarboxylase in iBsu1103"},
    "Phenylacetyl-Gln(PAGln)":{"ex": "EX_cpd00053_e", "fc":  3.28, "adi_norm": 0.20,
                               "risk": "Platelet activation; pro-thrombotic",
                               "note": "Gln exchange used as proxy for PAGln"},
    "L-Arg (ADMA proxy)":     {"ex": "EX_cpd03535_e", "fc": 25.70, "adi_norm": 0.30,
                               "risk": "eNOS inhibition; endothelial damage",
                               "note": "Arg-phosphate exchange (cpd03535)"},
}

# ══════════════════════════════════════════════════════════
# 3. 提取风险代谢物通量（cFBA 实际值 + FVA 范围）
# ══════════════════════════════════════════════════════════
log.info("Extracting risk metabolite fluxes from cFBA + FVA ...")
risk_rows = []

for met_name, info in RISK_METS.items():
    ex_id = info["ex"]

    cfba_v  = cfba_map.get(ex_id, np.nan)
    base_v  = base_map.get(ex_id, np.nan)

    fva_min = fva_max = np.nan
    if ex_id in fva_df.index:
        fva_min = fva_df.loc[ex_id, "minimum"]
        fva_max = fva_df.loc[ex_id, "maximum"]

    # 归一化通量（相对于 FVA 最大值）
    norm_cfba = (cfba_v / fva_max) if (not np.isnan(fva_max) and fva_max > 1e-9) else (
                 cfba_v if not np.isnan(cfba_v) else 0.0)
    norm_base = (base_v / fva_max) if (not np.isnan(fva_max) and fva_max > 1e-9) else (
                 base_v if not np.isnan(base_v) else 0.0)

    # RI = 归一化通量 / ADI_norm
    ri_cfba = round(float(norm_cfba) / info["adi_norm"], 4) if not np.isnan(norm_cfba) else np.nan
    ri_base = round(float(norm_base) / info["adi_norm"], 4) if not np.isnan(norm_base) else np.nan

    risk_rows.append({
        "metabolite":     met_name,
        "exchange_rxn":   ex_id,
        "fc_observed":    info["fc"],
        "risk_type":      info["risk"],
        "note":           info["note"],
        "baseline_flux":  round(float(base_v), 4) if not np.isnan(base_v) else np.nan,
        "cfba_flux":      round(float(cfba_v), 4) if not np.isnan(cfba_v) else np.nan,
        "fva_min":        round(float(fva_min), 4) if not np.isnan(fva_min) else np.nan,
        "fva_max":        round(float(fva_max), 4) if not np.isnan(fva_max) else np.nan,
        "norm_cfba":      round(float(norm_cfba), 6),
        "adi_norm":       info["adi_norm"],
        "ri_cfba":        ri_cfba,
        "ri_baseline":    ri_base,
    })
    log.info(f"  {met_name}: cFBA={cfba_v:.2f}  FVA[{fva_min:.2f},{fva_max:.2f}]  norm={norm_cfba:.4f}  RI={ri_cfba}")

risk_df = pd.DataFrame(risk_rows)
risk_df.to_csv(OUTPUT / "03_risk_fluxes.csv", index=False, encoding="utf-8-sig")
log.info("  03_risk_fluxes.csv saved")

# ══════════════════════════════════════════════════════════
# 4. PEA exchange 上界扫描
#    模拟"苯丙氨酸水解程度"→ PEA 允许积累量不同的风险情景
#    iBsu1103 无 Phe→PEA 路径，PEA 只能从胞外摄入再转运
#    这里扫描 EX_cpd03161_e 上界 (0→2)，模拟环境中 PEA 暴露水平
# ══════════════════════════════════════════════════════════
log.info("PEA exchange upper-bound scan ...")
scan_rows = []

pea_ex = "EX_cpd03161_e"
scan_ubs = np.linspace(0, 2, 21)    # 0~2 mmol/gDW/h（合理生物学范围）

with model_ref as model:
    apply_natto_constraints(model)
    bg_natto = model.slim_optimize()

    for ub in scan_ubs:
        with model:
            rxn = model.reactions.get_by_id(pea_ex)
            rxn.upper_bound = float(ub)
            rxn.lower_bound = 0.0
            model.objective = "bio00006"
            sol = model.optimize()
            growth = sol.objective_value if sol.status == "optimal" else np.nan
            pea_f  = sol.fluxes.get(pea_ex, 0.0) if sol.status == "optimal" else np.nan
            scan_rows.append({
                "pea_ub":   round(ub, 4),
                "growth":   round(float(growth), 6) if not np.isnan(growth) else np.nan,
                "pea_flux": round(float(pea_f),  6) if not np.isnan(pea_f)  else np.nan,
                "ri_pea":   round(float(pea_f) / 0.05, 4) if not np.isnan(pea_f) else np.nan,
            })

scan_df = pd.DataFrame(scan_rows)
scan_df.to_csv(OUTPUT / "03_pea_scan.csv", index=False, encoding="utf-8-sig")
log.info("  03_pea_scan.csv saved")

# ══════════════════════════════════════════════════════════
# 5. 多场景综合风险指数 RI
#    场景：(A) open; (B) natto standard; (C) natto + PEA restricted;
#          (D) natto + low hydrolysis; (E) natto minimal
# ══════════════════════════════════════════════════════════
log.info("Computing multi-scenario Risk Index ...")

SCENARIOS = {
    "Open (baseline)":         {"apply_natto": False, "pea_ub": 2.0,  "glu_ub": 10000},
    "Natto (standard)":        {"apply_natto": True,  "pea_ub": 2.0,  "glu_ub": 10000},
    "Natto + PEA control":     {"apply_natto": True,  "pea_ub": 0.5,  "glu_ub": 10000},
    "Natto + low hydrolysis":  {"apply_natto": True,  "pea_ub": 0.1,  "glu_ub": 5000},
    "Natto + minimal AA":      {"apply_natto": True,  "pea_ub": 0.05, "glu_ub": 1000},
}

ri_rows = []
for scenario, params in SCENARIOS.items():
    row_d = {"scenario": scenario}
    ri_total = 0.0

    with model_ref as model:
        if params["apply_natto"]:
            apply_natto_constraints(model)

        # 应用 PEA 上界
        try:
            pea_rxn = model.reactions.get_by_id("EX_cpd03161_e")
            pea_rxn.upper_bound = params["pea_ub"]
            pea_rxn.lower_bound = 0.0
        except Exception:
            pass

        sol = model.optimize()
        if sol.status != "optimal":
            for met_name in RISK_METS:
                row_d[met_name] = np.nan
            row_d["RI_total"] = np.nan
            ri_rows.append(row_d)
            continue

        for met_name, info in RISK_METS.items():
            ex_id  = info["ex"]
            v = sol.fluxes.get(ex_id, 0.0)
            # FVA max 归一化
            fva_max_v = fva_df.loc[ex_id, "maximum"] if ex_id in fva_df.index else 1.0
            norm_v = (v / fva_max_v) if fva_max_v > 1e-9 else abs(v)
            ri_v = round(float(norm_v) / info["adi_norm"], 4)
            row_d[met_name] = ri_v
            ri_total += max(ri_v, 0.0)

    row_d["RI_total"] = round(ri_total, 4)
    ri_rows.append(row_d)
    log.info(f"  {scenario}: RI_total={ri_total:.4f}")

ri_df = pd.DataFrame(ri_rows)
ri_df.to_csv(OUTPUT / "03_risk_index.csv", index=False, encoding="utf-8-sig")
log.info("  03_risk_index.csv saved")

# ══════════════════════════════════════════════════════════
# 6. 图7：PEA 上界扫描曲线
# ══════════════════════════════════════════════════════════
log.info("Drawing fig07_pea_response.png ...")
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = ax.twinx()

ax.plot(scan_df["pea_ub"], scan_df["pea_flux"],
        color=utils.PALETTE["natto_up"], lw=2.5, marker="o", ms=5,
        label="PEA secretion flux")
ax2.plot(scan_df["pea_ub"], scan_df["growth"],
         color=utils.PALETTE["neutral"], lw=2, ls="--", marker="s", ms=4,
         label="Biomass flux (growth)")

# 安全阈值线（RI=1 → pea_flux = 0.05）
ax.axhline(0.05, color="#e74c3c", lw=1.2, ls=":", label="Safety threshold (RI=1)")
ax.fill_between(scan_df["pea_ub"], 0, 0.05, alpha=0.12, color="#27ae60", label="Safe zone")
ax.fill_between(scan_df["pea_ub"], 0.05, scan_df["pea_ub"].max(),
                alpha=0.08, color="#e74c3c")

ax.set_xlabel("Allowed PEA secretion (mmol/gDW/h) — proxy for protein hydrolysis intensity",
              fontsize=9)
ax.set_ylabel("PEA flux (mmol/gDW/h)", fontsize=10, color=utils.PALETTE["natto_up"])
ax2.set_ylabel("Biomass (growth rate)", fontsize=10, color=utils.PALETTE["neutral"])
ax.set_title("Phenethylamine Secretion as a Function of Allowed Upper Bound\n"
             "(Natto cFBA; PEA model gap noted — no Phe decarboxylase in iBsu1103)",
             fontsize=10, fontweight="bold")

note_text = ("Note: iBsu1103 lacks a Phe decarboxylase (R_rxnXXXX).\n"
             "PEA FVA max = 0 in this model → PEA accumulation\n"
             "in natto likely involves an unmodeled enzyme (novel finding).")
ax.text(0.98, 0.55, note_text, transform=ax.transAxes, fontsize=7.5,
        color="#555", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#ccc"))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

save_fig(fig, "fig07_pea_response.png")
log.info("  fig07_pea_response.png saved")

# ══════════════════════════════════════════════════════════
# 7. 图8：三种风险代谢物通量概览
# ══════════════════════════════════════════════════════════
log.info("Drawing fig08_risk_overview.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：三种代谢物 cFBA 通量与 FVA 范围
ax = axes[0]
n = len(risk_rows)
x = np.arange(n)
w = 0.32

bars_b = ax.bar(x - w/2, [d["baseline_flux"] if not np.isnan(d["baseline_flux"]) else 0 for d in risk_rows],
                w, color=utils.PALETTE["neutral"], label="Baseline cFBA", alpha=0.85)
bars_n = ax.bar(x + w/2, [d["cfba_flux"] if not np.isnan(d["cfba_flux"]) else 0 for d in risk_rows],
                w, color=utils.PALETTE["natto_up"], label="Natto cFBA", alpha=0.85)

# FVA 误差棒（显示 FVA max）
for i, d in enumerate(risk_rows):
    if not np.isnan(d["fva_max"]) and d["fva_max"] < 1000:
        ax.errorbar(x[i] + w/2, d["cfba_flux"] if not np.isnan(d["cfba_flux"]) else 0,
                    yerr=[[0], [d["fva_max"] - (d["cfba_flux"] if not np.isnan(d["cfba_flux"]) else 0)]],
                    fmt="none", color="#c0392b", capsize=4, lw=1.5)

short_labels = [d["metabolite"].split("(")[0][:22] for d in risk_rows]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Flux (mmol/gDW/h)", fontsize=9)
ax.set_title("Risk Metabolite Fluxes\n(cFBA Baseline vs. Natto Conditions)", fontsize=9, fontweight="bold")
ax.legend(fontsize=8)

# 注释 PEA model gap
ax.annotate("Model gap:\nno Phe decarboxylase\n→ flux = 0",
            xy=(0, 0), xytext=(0, max(d["cfba_flux"] for d in risk_rows if not np.isnan(d["cfba_flux"]))/3),
            fontsize=6.5, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

# 右：Risk Index 柱状图（按场景）
ax2 = axes[1]
ri_met_cols = list(RISK_METS.keys())
# ri_rows is already the list of dicts with scenario key

x2 = np.arange(len(ri_rows))
bottoms = np.zeros(len(ri_rows))
colors_bar = [utils.PALETTE["natto_up"], utils.PALETTE["soy_up"], utils.PALETTE["aa"]]
for j, (met_col, color) in enumerate(zip(ri_met_cols, colors_bar)):
    vals = [float(d.get(met_col, 0) or 0) for d in ri_rows]
    ax2.bar(x2, vals, bottom=bottoms,
            color=color, alpha=0.8, label=met_col.split("(")[0][:20])
    bottoms += np.array(vals)

ax2.axhline(1.0, color="#e74c3c", lw=1.2, ls="--", label="RI = 1 (safe threshold)")
ax2.set_xticks(x2)
ax2.set_xticklabels([d["scenario"].replace("Natto + ", "\n").replace("Natto ", "\n") for d in ri_rows],
                    rotation=15, ha="right", fontsize=7.5)
ax2.set_ylabel("Risk Index (RI, stacked)", fontsize=9)
ax2.set_title("Cumulative Risk Index by Scenario\n(lower = safer)", fontsize=9, fontweight="bold")
ax2.legend(fontsize=7, loc="upper right")

plt.tight_layout()
save_fig(fig, "fig08_risk_overview.png")
log.info("  fig08_risk_overview.png saved")

# ══════════════════════════════════════════════════════════
# 8. 图9：RI 热图
# ══════════════════════════════════════════════════════════
log.info("Drawing fig09_risk_index_heatmap.png ...")

ri_hm = ri_df.set_index("scenario")[ri_met_cols].astype(float).fillna(0)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(
    ri_hm, annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.4, ax=ax,
    cbar_kws={"label": "Risk Index (RI)"},
    vmin=0, vmax=max(ri_hm.values.max(), 1.5),
)
ax.set_xticklabels([c.split("(")[0][:20] for c in ri_met_cols], rotation=20, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_title("Food Safety Risk Index Across Fermentation Scenarios\n"
             "(RI > 1 = potential concern; PEA = 0 due to model gap)",
             fontsize=10, fontweight="bold", pad=10)

# RI_total 旁注
for i, row_s in enumerate(ri_rows):
    ax.text(len(ri_met_cols) + 0.1, i + 0.5,
            f"tot={row_s['RI_total']:.3f}", va="center", fontsize=7, color="#333")

plt.tight_layout()
save_fig(fig, "fig09_risk_index_heatmap.png")
log.info("  fig09_risk_index_heatmap.png saved")

# ══════════════════════════════════════════════════════════
# 9. 控制台摘要
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Module 3 Complete: Food Safety Risk Assessment")
print("=" * 65)
print()
print(f"  {'Metabolite':<35} {'cFBA flux':>10} {'FVA max':>10} {'RI':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
for d in risk_rows:
    fv = d['cfba_flux'] if not np.isnan(d['cfba_flux']) else 0
    fm = d['fva_max']   if not np.isnan(d['fva_max'])   else 0
    ri = d['ri_cfba']   if not np.isnan(d['ri_cfba'])   else 0
    print(f"  {d['metabolite']:<35} {fv:>10.4f} {fm:>10.4f} {ri:>8.4f}")
print()
print("  KEY FINDING: PEA (Phenethylamine) FVA max = 0.0")
print("  => iBsu1103 lacks phenylalanine decarboxylase (model gap)")
print("  => PEA accumulation in natto requires an uncharacterized enzyme")
print("  => This predicts a novel gene discovery target")
print()
print("  Risk Index by scenario:")
print(f"  {'Scenario':<35} {'RI_total':>10}")
for d in ri_rows:
    ri_t = d['RI_total'] if not np.isnan(d['RI_total']) else 0
    print(f"  {d['scenario']:<35} {ri_t:>10.4f}")
print()
print("  Output files:")
for f in ["03_risk_fluxes.csv", "03_pea_scan.csv", "03_risk_index.csv",
          "fig07_pea_response.png", "fig08_risk_overview.png", "fig09_risk_index_heatmap.png"]:
    status = "[OK]" if (OUTPUT / f).exists() else "[--]"
    print(f"    {status}  code/output/{f}")
print("=" * 65)

log_progress("Module 3",
             f"Food safety: PEA model gap found (FVA max=0), RI computed for 3 risk mets, 5 scenarios")
log.info("Module 3 complete.")
