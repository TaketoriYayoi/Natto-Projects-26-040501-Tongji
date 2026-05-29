import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import cobra
from cobra.flux_analysis import flux_variability_analysis

import utils
from utils import set_style, save_fig, log_progress, OUTPUT, DATA

log = utils.get_logger("04_fseof")
set_style()

MODEL_XML = DATA / "MODEL1507180015_url.xml"

def apply_natto_constraints(model):
    c = pd.read_csv(OUTPUT / "01_constraints_table.csv")
    ex_ids = {r.id for r in model.reactions if r.id.startswith("EX_")}
    for _, row in c.iterrows():
        ex_id = "EX_" + row["model_id"].replace("M_", "")
        if ex_id not in ex_ids:
            continue
        rxn = model.reactions.get_by_id(ex_id)
        if row["constraint_type"] == "secretion":
            rxn.lower_bound = 0.0
        else:
            rxn.upper_bound = 0.0

# ══════════════════════════════════════════════════════════
# 1. 加载模型
# ══════════════════════════════════════════════════════════
log.info("Loading iBsu1103 model ...")
model_ref = cobra.io.read_sbml_model(str(MODEL_XML))
BIOMASS = "bio00006"

# 三个目标产物
TARGETS = {
    "L-Ornithine":    {"ex": "EX_cpd00064_e", "fc_obs": 97.7,  "value": "health supplement, ~$20/kg"},
    "L-Histidine":    {"ex": "EX_cpd00119_e", "fc_obs": 34.5,  "value": "N-Ac-His precursor, >$200/kg"},
    "L-Glutamate":    {"ex": "EX_cpd00023_e", "fc_obs": 12.0,  "value": "flavor / fermentation key node"},
}

# ══════════════════════════════════════════════════════════
# 2. 计算每个目标的理论最大产量 & 实用产量（80% biomass）
# ══════════════════════════════════════════════════════════
log.info("Computing theoretical and practical max yields ...")

yield_rows = []
for prod_name, info in TARGETS.items():
    ex_id = info["ex"]

    # --- 理论最大产量（不约束生物量）---
    with model_ref as m:
        apply_natto_constraints(m)
        m.objective = ex_id
        m.objective_direction = "max"
        sol = m.optimize()
        theo_max = sol.objective_value if sol.status == "optimal" else np.nan

    # --- 实用产量（≥80% 最大生物量）---
    with model_ref as m:
        apply_natto_constraints(m)
        bio_max = m.slim_optimize()
        m.reactions.get_by_id(BIOMASS).lower_bound = 0.8 * bio_max
        m.objective = ex_id
        m.objective_direction = "max"
        sol = m.optimize()
        prac_max = sol.objective_value if sol.status == "optimal" else np.nan

    # --- cFBA 实测通量（作为"当前基线"）---
    cfba_df = pd.read_csv(OUTPUT / "02_cfba_fluxes.csv")
    cfba_v = cfba_df.loc[cfba_df.reaction == ex_id, "cfba_flux"].values
    current = float(cfba_v[0]) if len(cfba_v) > 0 else 0.0

    gap_pct = round((prac_max - current) / prac_max * 100, 1) if prac_max > 1e-9 else np.nan

    yield_rows.append({
        "product":         prod_name,
        "exchange_rxn":    ex_id,
        "fc_observed":     info["fc_obs"],
        "industrial_value": info["value"],
        "theo_max_flux":   round(float(theo_max), 4) if not np.isnan(theo_max) else np.nan,
        "prac_max_flux":   round(float(prac_max), 4) if not np.isnan(prac_max) else np.nan,
        "current_flux":    round(current, 4),
        "yield_gap_pct":   gap_pct,
    })
    log.info(f"  {prod_name}: theo={theo_max:.2f}, prac={prac_max:.2f}, current={current:.2f}, gap={gap_pct}%")

yield_df = pd.DataFrame(yield_rows)
yield_df.to_csv(OUTPUT / "04_yield_gap.csv", index=False, encoding="utf-8-sig")
log.info("  04_yield_gap.csv saved")

# ══════════════════════════════════════════════════════════
# 3. FSEOF：扫描每个目标的工程靶点
#    方法：将目标产物 exchange 下界从 0 线性增至 practical_max
#    在 n_steps 步中记录所有反应的通量，最后做线性回归
# ══════════════════════════════════════════════════════════
log.info("Running FSEOF for all targets ...")

N_STEPS   = 10    # 扫描步数
N_TOPGENE = 20    # 每个目标输出前 N 个靶点

fseof_results = {}

for prod_name, info in TARGETS.items():
    ex_id    = info["ex"]
    prac_row = [d for d in yield_rows if d["product"] == prod_name][0]
    prac_max = prac_row["prac_max_flux"]

    if np.isnan(prac_max) or prac_max < 1e-9:
        log.warning(f"  {prod_name}: prac_max too small, skip FSEOF")
        continue

    log.info(f"  FSEOF for {prod_name} (0 → {prac_max:.2f}) ...")

    scan_lbs = np.linspace(0, prac_max * 0.95, N_STEPS)
    flux_matrix = {}  # rxn_id → list of fluxes

    with model_ref as model:
        apply_natto_constraints(model)
        bio_max = model.slim_optimize()
        model.reactions.get_by_id(BIOMASS).lower_bound = 0.8 * bio_max

        for lb in scan_lbs:
            with model:
                rxn = model.reactions.get_by_id(ex_id)
                rxn.lower_bound = float(lb)
                model.objective = BIOMASS
                sol = model.optimize()
                if sol.status != "optimal":
                    continue
                for rxn_id, v in sol.fluxes.items():
                    if rxn_id not in flux_matrix:
                        flux_matrix[rxn_id] = []
                    flux_matrix[rxn_id].append(v)

    # 线性回归：通量 ~ 产量步骤
    target_lbs_arr = scan_lbs[:max(len(v) for v in flux_matrix.values())]
    fseof_rows = []
    for rxn_id, flux_list in flux_matrix.items():
        if len(flux_list) < 3:
            continue
        # 补齐长度
        n = len(flux_list)
        x = target_lbs_arr[:n]
        y = np.array(flux_list)
        if np.std(y) < 1e-12:
            continue
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            continue

        # 获取反应元信息
        try:
            rxn = model_ref.reactions.get_by_id(rxn_id)
            rxn_name = rxn.name[:55]
            gpr = rxn.gene_reaction_rule[:60] if rxn.gene_reaction_rule else ""
            genes = [g.id for g in rxn.genes]
        except Exception:
            rxn_name, gpr, genes = "", "", []

        fseof_rows.append({
            "reaction":     rxn_id,
            "rxn_name":     rxn_name,
            "slope":        round(slope, 6),
            "intercept":    round(intercept, 4),
            "gpr":          gpr,
            "n_genes":      len(genes),
            "target_type":  "overexpress" if slope > 0.01 else ("knockout" if slope < -0.01 else "neutral"),
        })

    fseof_df = pd.DataFrame(fseof_rows).sort_values("slope", key=abs, ascending=False)
    fseof_results[prod_name] = fseof_df

    # 筛选有基因且非 exchange 的靶点
    eng_targets = fseof_df[
        (fseof_df["n_genes"] > 0) &
        (~fseof_df["reaction"].str.startswith("EX_")) &
        (~fseof_df["reaction"].str.startswith("bio")) &
        (fseof_df["slope"].abs() > 0.01)
    ].head(N_TOPGENE)

    safe_name = prod_name.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
    eng_targets.to_csv(OUTPUT / f"04_fseof_{safe_name}.csv", index=False, encoding="utf-8-sig")
    log.info(f"    -> {len(eng_targets)} engineering targets found, saved to 04_fseof_{safe_name}.csv")

# ══════════════════════════════════════════════════════════
# 4. 图10：Yield Gap 分析（三目标）
# ══════════════════════════════════════════════════════════
log.info("Drawing fig10_yield_gap.png ...")
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(yield_df))
w = 0.25
ax.bar(x - w,   yield_df["theo_max_flux"],  w, color="#3498DB", label="Theoretical max", alpha=0.85)
ax.bar(x,       yield_df["prac_max_flux"],  w, color="#27AE60", label="Practical max (80% growth)", alpha=0.85)
ax.bar(x + w,   yield_df["current_flux"],   w, color=utils.PALETTE["natto_up"], label="cFBA current", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(yield_df["product"], fontsize=10)
ax.set_ylabel("Secretion flux (mmol/gDW/h)", fontsize=10)
ax.set_title("Yield Gap Analysis for High-Value Target Metabolites\n(Natto cFBA Conditions)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

for i, row_y in yield_df.iterrows():
    gap = row_y["yield_gap_pct"]
    if not np.isnan(gap):
        ax.text(i, row_y["prac_max_flux"] + 0.03, f"gap:{gap:.0f}%",
                ha="center", fontsize=8, color="#e74c3c", fontweight="bold")

save_fig(fig, "fig10_yield_gap.png")
log.info("  fig10_yield_gap.png saved")

# ══════════════════════════════════════════════════════════
# 5. 图11：FSEOF 结果 — Top 工程靶点汇总
# ══════════════════════════════════════════════════════════
log.info("Drawing fig11_engineering_targets.png ...")

fig, axes = plt.subplots(1, len(fseof_results), figsize=(6 * len(fseof_results), 7),
                          squeeze=False)
colors_fseof = {"overexpress": utils.PALETTE["natto_up"],
                "knockout":    utils.PALETTE["soy_up"],
                "neutral":     utils.PALETTE["neutral"]}

for col, (prod_name, fseof_df) in enumerate(fseof_results.items()):
    ax = axes[0][col]

    # 只取有基因、非 EX 的前 15 个
    sub = fseof_df[
        (fseof_df["n_genes"] > 0) &
        (~fseof_df["reaction"].str.startswith("EX_")) &
        (~fseof_df["reaction"].str.startswith("bio"))
    ].head(15)

    if sub.empty:
        ax.text(0.5, 0.5, "No targets", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(prod_name)
        continue

    y      = np.arange(len(sub))
    slopes = sub["slope"].values
    bar_colors = [colors_fseof[t] for t in sub["target_type"]]
    labels = [f"{row.reaction}\n({row.rxn_name[:30]})" for _, row in sub.iterrows()]

    ax.barh(y, slopes, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("FSEOF slope (flux change per unit product)", fontsize=8)
    ax.set_title(f"Engineering Targets: {prod_name}\n(FSEOF, natto cFBA)",
                 fontsize=9, fontweight="bold")

    patches = [
        mpatches.Patch(color=colors_fseof["overexpress"], label="Overexpress"),
        mpatches.Patch(color=colors_fseof["knockout"],    label="Knockout"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

plt.tight_layout()
save_fig(fig, "fig11_engineering_targets.png")
log.info("  fig11_engineering_targets.png saved")

# ══════════════════════════════════════════════════════════
# 6. 图12：FSEOF 通量扫描轨迹（以 L-Ornithine 为例）
# ══════════════════════════════════════════════════════════
log.info("Drawing fig12_fseof_trajectory.png ...")

prod_ex = "EX_cpd00064_e"  # L-Ornithine
prac_v   = [d["prac_max_flux"] for d in yield_rows if d["product"] == "L-Ornithine"][0]

# 重跑扫描，记录 top 6 反应的轨迹
TOP_RXNS = 6
if "L-Ornithine" in fseof_results:
    top_rxns = fseof_results["L-Ornithine"][
        (fseof_results["L-Ornithine"]["n_genes"] > 0) &
        (~fseof_results["L-Ornithine"]["reaction"].str.startswith("EX_"))
    ].head(TOP_RXNS)["reaction"].tolist()
else:
    top_rxns = []

if top_rxns and prac_v > 0:
    scan_lbs2 = np.linspace(0, prac_v * 0.95, 15)
    traj = {rxn_id: [] for rxn_id in top_rxns}
    prod_fluxes2 = []

    with model_ref as model:
        apply_natto_constraints(model)
        bio_max = model.slim_optimize()
        model.reactions.get_by_id(BIOMASS).lower_bound = 0.8 * bio_max

        for lb in scan_lbs2:
            with model:
                rxn = model.reactions.get_by_id(prod_ex)
                rxn.lower_bound = float(lb)
                model.objective = BIOMASS
                sol = model.optimize()
                if sol.status != "optimal":
                    for rid in top_rxns:
                        traj[rid].append(np.nan)
                    prod_fluxes2.append(np.nan)
                else:
                    for rid in top_rxns:
                        traj[rid].append(sol.fluxes.get(rid, 0.0))
                    prod_fluxes2.append(sol.fluxes.get(prod_ex, 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    palette_traj = plt.cm.tab10(np.linspace(0, 0.9, len(top_rxns)))
    for i, rxn_id in enumerate(top_rxns):
        y_vals = traj[rxn_id]
        if all(np.isnan(v) for v in y_vals):
            continue
        try:
            rname = model_ref.reactions.get_by_id(rxn_id).name[:35]
        except Exception:
            rname = rxn_id
        ax.plot(prod_fluxes2, y_vals, lw=2, color=palette_traj[i],
                label=f"{rxn_id} ({rname})", marker="o", ms=4)

    ax.set_xlabel("L-Ornithine secretion flux (mmol/gDW/h)", fontsize=10)
    ax.set_ylabel("Reaction flux (mmol/gDW/h)", fontsize=10)
    ax.set_title("FSEOF Flux Trajectories for Top Engineering Targets\n(L-Ornithine production, natto cFBA)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.axhline(0, color="black", lw=0.5, ls=":")

    save_fig(fig, "fig12_fseof_trajectory.png")
    log.info("  fig12_fseof_trajectory.png saved")

# ══════════════════════════════════════════════════════════
# 7. 控制台摘要
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Module 4 Complete: Engineering Target Prediction (FSEOF)")
print("=" * 65)
print()
print(f"  {'Product':<20} {'Theo max':>10} {'Prac max':>10} {'Current':>10} {'Gap%':>7}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
for d in yield_rows:
    print(f"  {d['product']:<20} {d['theo_max_flux']:>10.2f} "
          f"{d['prac_max_flux']:>10.2f} {d['current_flux']:>10.2f} "
          f"{str(d['yield_gap_pct'])+'%':>7}")
print()
print("  FSEOF top engineering targets:")
for prod_name, fseof_df in fseof_results.items():
    sub = fseof_df[(fseof_df["n_genes"]>0) & (~fseof_df["reaction"].str.startswith("EX_"))].head(5)
    print(f"\n  [{prod_name}]")
    for _, r in sub.iterrows():
        act = "UP" if r["target_type"]=="overexpress" else ("DOWN" if r["target_type"]=="knockout" else "~")
        print(f"    {act:4} {r['reaction']:15}  slope={r['slope']:+.4f}  {r['rxn_name'][:40]}")
print()
print("  Output files:")
for f_name in ["04_yield_gap.csv",
               "04_fseof_L_Ornithine.csv", "04_fseof_L_Histidine.csv", "04_fseof_L_Glutamate.csv",
               "fig10_yield_gap.png", "fig11_engineering_targets.png", "fig12_fseof_trajectory.png"]:
    status = "[OK]" if (OUTPUT / f_name).exists() else "[--]"
    print(f"    {status}  code/output/{f_name}")
print("=" * 65)

log_progress("Module 4", f"FSEOF for 3 targets: {', '.join(TARGETS.keys())}; yield gap + engineering targets saved")
log.info("Module 4 complete.")
