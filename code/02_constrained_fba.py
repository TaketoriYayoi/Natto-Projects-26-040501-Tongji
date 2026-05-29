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
from utils import (
    load_model_metadata, set_style, save_fig, log_progress, OUTPUT, DATA,
)

log = utils.get_logger("02_cfba")
set_style()

MODEL_XML = DATA / "MODEL1507180015_url.xml"

# ══════════════════════════════════════════════════════════
# 1. 加载模型
# ══════════════════════════════════════════════════════════
log.info("加载 iBsu1103 模型 ...")
model = cobra.io.read_sbml_model(str(MODEL_XML))
log.info(f"  反应: {len(model.reactions)}  代谢物: {len(model.metabolites)}  基因: {len(model.genes)}")

BIOMASS_RXN = "bio00006"

# ══════════════════════════════════════════════════════════
# 2. 基线 FBA（标准条件）
# ══════════════════════════════════════════════════════════
log.info("运行基线 FBA ...")
with model:
    model.objective = BIOMASS_RXN
    baseline_sol = model.optimize()

log.info(f"  基线生物量: {baseline_sol.objective_value:.4f}")
baseline_fluxes = baseline_sol.fluxes.rename("baseline_flux").reset_index()
baseline_fluxes.columns = ["reaction", "baseline_flux"]
baseline_fluxes.to_csv(OUTPUT / "02_baseline_fluxes.csv", index=False, encoding="utf-8-sig")
log.info("  02_baseline_fluxes.csv 已保存")

# ══════════════════════════════════════════════════════════
# 3. 应用纳豆发酵约束
# ══════════════════════════════════════════════════════════
log.info("加载约束表并应用到模型 ...")
constraints = pd.read_csv(OUTPUT / "01_constraints_table.csv")

# 将 model_id（如 M_cpd00119_e）转成 exchange 反应 ID（EX_cpd00119_e）
# 只有 _e（胞外）代谢物才有 exchange 反应；_c 代谢物找对应的转运反应或跳过
ex_rxn_ids = {r.id for r in model.reactions if r.id.startswith("EX_")}
applied_count = 0
skipped = []

with model:
    # ── 方向约束策略（Direction-only cFBA）──
    # 只限制 exchange 方向，不强制通量幅度。
    # 纳豆上调（secretion）→ lb = 0（允许分泌，禁止摄取）
    # 大豆上调（uptake）   → ub = 0（允许摄取，禁止分泌）
    # 这与文献中标准 cFBA 做法一致，避免过度约束导致 infeasible。
    for _, row in constraints.iterrows():
        met_raw   = row["model_id"]          # M_cpd00119_e
        met_id    = met_raw.replace("M_", "") # cpd00119_e
        ex_id     = "EX_" + met_id           # EX_cpd00119_e
        direction = row["constraint_type"]   # "secretion" or "uptake"

        if ex_id not in ex_rxn_ids:
            skipped.append(ex_id)
            continue

        rxn = model.reactions.get_by_id(ex_id)
        if direction == "secretion":
            # natto_up → 仅允许分泌方向（lb 升至 0）
            if rxn.lower_bound < 0:
                rxn.lower_bound = 0.0
            applied_count += 1
        else:
            # soy_up → 仅允许摄取方向（ub 降至 0）
            if rxn.upper_bound > 0:
                rxn.upper_bound = 0.0
            applied_count += 1

    log.info(f"  成功应用 exchange 方向约束: {applied_count} 个")
    if skipped:
        log.info(f"  跳过（无 EX_ 反应）: {len(skipped)} 个")

    # 求解约束型 FBA
    log.info("运行约束型 FBA (cFBA) ...")
    cfba_sol = model.optimize()
    log.info(f"  cFBA 生物量: {cfba_sol.objective_value:.4f}  状态: {cfba_sol.status}")

    cfba_fluxes = cfba_sol.fluxes.rename("cfba_flux").reset_index()
    cfba_fluxes.columns = ["reaction", "cfba_flux"]

    # ══════════════════════════════════════════════════════
    # 4. FVA（在 cFBA 约束下，保留 90% 最大生物量）
    # ══════════════════════════════════════════════════════
    log.info("运行 FVA (fraction=0.90) ...")
    try:
        fva_result = flux_variability_analysis(
            model, fraction_of_optimum=0.90,
            processes=1,   # Windows 多进程不稳定，用单进程
        )
        log.info("  FVA 完成")
    except Exception as e:
        log.warning(f"  FVA 失败: {e}，跳过")
        fva_result = None

# ══════════════════════════════════════════════════════════
# 5. 保存通量结果
# ══════════════════════════════════════════════════════════
# 合并基线与 cFBA 通量
flux_df = baseline_fluxes.merge(cfba_fluxes, on="reaction", how="outer")
flux_df["delta_flux"] = flux_df["cfba_flux"] - flux_df["baseline_flux"]
flux_df["abs_delta"]  = flux_df["delta_flux"].abs()
flux_df.to_csv(OUTPUT / "02_cfba_fluxes.csv", index=False, encoding="utf-8-sig")
log.info("  02_cfba_fluxes.csv 已保存")

if fva_result is not None:
    fva_result["range"] = fva_result["maximum"] - fva_result["minimum"]
    fva_result.to_csv(OUTPUT / "02_fva_ranges.csv", encoding="utf-8-sig")
    log.info("  02_fva_ranges.csv 已保存")

# ══════════════════════════════════════════════════════════
# 6. 三条核心通量故事定量分析
# ══════════════════════════════════════════════════════════
log.info("提取三条核心通量故事 ...")

# ── Story A: TCA 循环关键反应 ──
# 注：iBsu1103 无直接衣康酸合成反应（模型 gap），用 TCA 中间体通量代替
tca_reactions = {
    "Citrate lyase (rxn00256)":           "rxn00256",   # citrate → oxaloacetate + acetate
    "Isocitrate dehydrogenase (rxn01387)": "rxn01387",  # isocitrate → αKG
    "Isocitrate hydro-lyase (rxn01388)":   "rxn01388",  # isocitrate → 2-methylisocitrate
    "Succinate:CoA ligase (rxn00285)":     "rxn00285",  # succinyl-CoA → succinate
    "Malate dehydratase (rxn00799)":       "rxn00799",  # malate → fumarate
    "Acetolactate synthase (rxn02185)":    "rxn02185",  # 2-acetolactate synthesis
    "GABA transaminase (rxn01204)":        "rxn01204",  # GABA → succinate semialdehyde
}

# ── Story B: 氨基酸代谢 ──
aa_reactions = {
    "Phe decarboxylase/PEA synth (rxn11405-t)": "rxn11405",  # phenethylamine transport
    "Phenethylamine oxidase (rxn01903)":          "rxn01903",
    "Ornithine carbamoyltransferase (rxn00573)":  "rxn00573",
    "Lysine degradation (rxn00460)":              "rxn00460",
    "His biosynthesis (rxn01977)":                "rxn01977",
}

# ── Story C: GABA 悖论 ──
gaba_reactions = {
    "GABA transaminase (rxn01204)":              "rxn01204",
    "Succinate-semialdehyde DH (rxn00509)":      "rxn00509",
    "Butyrate-CoA transferase (rxn00875)":        "rxn00875",
}

all_story_rxns = {**tca_reactions, **aa_reactions, **gaba_reactions}

story_rows = []
for label, rxn_id in all_story_rxns.items():
    try:
        base_v = float(baseline_sol.fluxes.get(rxn_id, np.nan))
        cfba_v = float(cfba_sol.fluxes.get(rxn_id, np.nan))
        story_rows.append({
            "label":        label,
            "reaction":     rxn_id,
            "baseline_flux": round(base_v, 6),
            "cfba_flux":    round(cfba_v, 6),
            "delta":        round(cfba_v - base_v, 6),
            "fold_change":  round(cfba_v / base_v, 4) if abs(base_v) > 1e-9 else np.nan,
            "story":        "TCA" if rxn_id in tca_reactions.values()
                            else ("AA" if rxn_id in aa_reactions.values() else "GABA"),
        })
    except Exception:
        pass

story_df = pd.DataFrame(story_rows)
story_df.to_csv(OUTPUT / "02_flux_stories.csv", index=False, encoding="utf-8-sig")
log.info("  02_flux_stories.csv 已保存")

# ══════════════════════════════════════════════════════════
# 7. 图4：通量差异热图（Top 40 变化最大的反应）
# ══════════════════════════════════════════════════════════
log.info("绘制 fig04_flux_heatmap.png ...")

top40 = (
    flux_df
    .dropna(subset=["baseline_flux", "cfba_flux"])
    .nlargest(40, "abs_delta")
    .set_index("reaction")
)
# 排除生物量反应本身
top40 = top40[~top40.index.str.contains("bio")]
top_plot = top40[["baseline_flux", "cfba_flux"]].head(30)

fig, ax = plt.subplots(figsize=(10, 10))

# 添加反应名称
rxn_names = {}
for rid in top_plot.index:
    try:
        rxn_names[rid] = model.reactions.get_by_id(rid).name[:45]
    except Exception:
        rxn_names[rid] = rid
top_plot.index = [f"{rid}\n({rxn_names.get(rid,'')[:35]})" for rid in top_plot.index]

sns.heatmap(
    top_plot,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.3,
    ax=ax,
    cbar_kws={"label": "Flux (mmol/gDW/h)"},
)
ax.set_xticklabels(["Baseline FBA", "Constrained FBA\n(Natto conditions)"], fontsize=10)
ax.set_title("Top 30 Reactions with Largest Flux Change\nBaseline vs. Constrained FBA",
             fontsize=11, fontweight="bold", pad=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)

save_fig(fig, "fig04_flux_heatmap.png")
log.info("  fig04_flux_heatmap.png 已保存")

# ══════════════════════════════════════════════════════════
# 8. 图5：TCA 循环通量对比
# ══════════════════════════════════════════════════════════
log.info("绘制 fig05_tca_fluxes.png ...")

tca_story = story_df[story_df["story"] == "TCA"].copy()
tca_story = tca_story.dropna(subset=["baseline_flux", "cfba_flux"])

if len(tca_story) > 0:
    x = np.arange(len(tca_story))
    w = 0.35
    labels = [lab.split("(")[0].strip() for lab in tca_story["label"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, tca_story["baseline_flux"], w,
                   color=utils.PALETTE["neutral"], label="Baseline FBA", alpha=0.85)
    bars2 = ax.bar(x + w/2, tca_story["cfba_flux"],    w,
                   color=utils.PALETTE["natto_up"], label="Natto cFBA", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Flux (mmol/gDW/h)", fontsize=10)
    ax.set_title("TCA Cycle & Related Reaction Fluxes\nBaseline vs. Natto Constrained FBA",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.5)

    # 数值标注
    for bar in bars1:
        h = bar.get_height()
        if abs(h) > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005 if h >= 0 else h - 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, color="#555")
    for bar in bars2:
        h = bar.get_height()
        if abs(h) > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005 if h >= 0 else h - 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, color="#c0392b")

    save_fig(fig, "fig05_tca_fluxes.png")
    log.info("  fig05_tca_fluxes.png 已保存")

# ══════════════════════════════════════════════════════════
# 9. 图6：氨基酸代谢 & GABA 通量
# ══════════════════════════════════════════════════════════
log.info("绘制 fig06_aa_gaba_fluxes.png ...")

aa_gaba = story_df[story_df["story"].isin(["AA", "GABA"])].copy().dropna(subset=["cfba_flux"])

if len(aa_gaba) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, story_tag, title, color in [
        (axes[0], "AA",   "Amino Acid Metabolism",  utils.PALETTE["aa"]),
        (axes[1], "GABA", "GABA Consumption",       utils.PALETTE["tca"]),
    ]:
        sub = aa_gaba[aa_gaba["story"] == story_tag]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        labels = [lab.split("(")[0].strip() for lab in sub["label"]]
        x = np.arange(len(sub))
        w = 0.35

        ax.bar(x - w/2, sub["baseline_flux"], w,
               color=utils.PALETTE["neutral"], label="Baseline", alpha=0.85)
        ax.bar(x + w/2, sub["cfba_flux"],     w,
               color=color, label="Natto cFBA", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Flux (mmol/gDW/h)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.5)

    fig.suptitle("Amino Acid & GABA Pathway Fluxes: Baseline vs. Natto cFBA",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig06_aa_gaba_fluxes.png")
    log.info("  fig06_aa_gaba_fluxes.png 已保存")

# ══════════════════════════════════════════════════════════
# 10. FVA 瓶颈 & 弹性节点统计
# ══════════════════════════════════════════════════════════
if fva_result is not None:
    bottlenecks = fva_result[fva_result["range"] < 0.01].sort_values("range")
    flexible    = fva_result[fva_result["range"] > 10].sort_values("range", ascending=False)
    log.info(f"  FVA 瓶颈反应（range<0.01）: {len(bottlenecks)} 个")
    log.info(f"  高弹性反应（range>10）:     {len(flexible)} 个")

# ══════════════════════════════════════════════════════════
# 11. 控制台摘要
# ══════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  模块二完成：约束型 FBA 通量重建")
print("═" * 60)
print(f"  模型规模: {len(model.reactions)} 反应 / {len(model.metabolites)} 代谢物")
print(f"  基线 FBA 生物量:   {baseline_sol.objective_value:.4f}")
print(f"  cFBA 生物量:       {cfba_sol.objective_value:.4f}")
print(f"  生物量变化率:      {(cfba_sol.objective_value/baseline_sol.objective_value-1)*100:.1f}%")
print(f"  成功施加 exchange 约束: {applied_count} 个")
print()
print("  三条核心通量故事（cFBA）：")
for _, r in story_df.iterrows():
    cfba_v = r['cfba_flux']
    base_v = r['baseline_flux']
    delta_str = f"Δ={r['delta']:+.4f}"
    print(f"    [{r['story']}] {r['label'][:45]:<45}  base={base_v:.4f}  cFBA={cfba_v:.4f}  {delta_str}")

print()
print("  输出文件：")
for f in ["02_baseline_fluxes.csv", "02_cfba_fluxes.csv",
          "02_fva_ranges.csv", "02_flux_stories.csv",
          "fig04_flux_heatmap.png", "fig05_tca_fluxes.png", "fig06_aa_gaba_fluxes.png"]:
    p = OUTPUT / f
    status = "[OK]" if p.exists() else "[--]"
    print(f"    {status}  code/output/{f}")
print("═" * 60)

log_progress("模块二", f"cFBA 完成: 生物量={cfba_sol.objective_value:.2f}, 施加{applied_count}个约束, 输出4个CSV+3张图")
log.info("模块二全部完成。")
