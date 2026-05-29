import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import cobra

import utils
from utils import OUTPUT, DATA

log = utils.get_logger("07_extended")

# ── 全局配色（Paul Tol Bright，与主图一致）──────────────────
R   = "#BB5566"
B   = "#004488"
OR  = "#DDAA33"
GR  = "#117733"
PU  = "#882255"
LB  = "#4477AA"
GY  = "#AAAAAA"
TEAL= "#44AA99"

def setup():
    import matplotlib as mpl; mpl.rcdefaults()
    plt.rcParams.update({
        "font.family":              ["DejaVu Sans"],
        "mathtext.fontset":         "dejavusans",
        "axes.unicode_minus":       False,
        "savefig.dpi":              300,
        "savefig.bbox":             "tight",
        "savefig.pad_inches":       0.06,
        "figure.facecolor":         "white",
        "axes.facecolor":           "white",
        "axes.edgecolor":           "#333333",
        "axes.linewidth":           0.9,
        "axes.spines.top":          False,
        "axes.spines.right":        False,
        "axes.grid":                False,
        "xtick.direction":          "out",
        "ytick.direction":          "out",
        "xtick.major.width":        0.8,
        "ytick.major.width":        0.8,
        "xtick.major.size":         3.5,
        "ytick.major.size":         3.5,
        "xtick.labelsize":          8,
        "ytick.labelsize":          8,
        "axes.labelsize":           9,
        "axes.titlesize":           9.5,
        "axes.titleweight":         "bold",
        "axes.labelpad":            4,
        "legend.fontsize":          7.5,
        "legend.frameon":           True,
        "legend.framealpha":        0.9,
        "legend.edgecolor":         "#CCCCCC",
        "lines.linewidth":          1.5,
        "patch.linewidth":          0.4,
        "figure.constrained_layout.use": True,
    })

setup()

def plbl(ax, letter, x=-0.14, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color="#111111")

def save(fig, name):
    p = OUTPUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  {name} saved ({p.stat().st_size//1024} KB)")

MODEL_XML = DATA / "MODEL1507180015_url.xml"
BIOMASS   = "bio00006"

log.info("Loading iBsu1103 model ...")
model_ref = cobra.io.read_sbml_model(str(MODEL_XML))

constraints_df = pd.read_csv(OUTPUT / "01_constraints_table.csv")

def apply_natto_constraints(model, fc_thresh=0.0):
    """
    应用纳豆方向约束。fc_thresh: 只对 |logFC| 对应的 FC 超过此阈值的代谢物施加约束。
    fc_thresh=0 表示应用所有约束（原始行为）。
    """
    ex_ids = {r.id for r in model.reactions if r.id.startswith("EX_")}
    applied = 0
    for _, row in constraints_df.iterrows():
        # 可选FC过滤
        if fc_thresh > 0:
            fc_val = float(row.get("fc", 1.0))
            fc_abs = fc_val if fc_val >= 1 else (1/fc_val if fc_val > 0 else 1)
            if fc_abs < fc_thresh:
                continue
        ex_id = "EX_" + str(row["model_id"]).replace("M_", "")
        if ex_id not in ex_ids:
            continue
        rxn = model.reactions.get_by_id(ex_id)
        if row["constraint_type"] == "secretion":
            rxn.lower_bound = 0.0
        else:
            rxn.upper_bound = 0.0
        applied += 1
    return applied

# ══════════════════════════════════════════════════════════════
# TASK 1：FSEOF 多目标扩展 + 热图
# ══════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 1: FSEOF multi-target analysis ...")
log.info("=" * 60)

FSEOF_TARGETS = {
    "L-Histidine":  {"ex": "EX_cpd00119_e", "fc_obs": 34.5,
                     "note": "N-Ac-His precursor; 1455x up"},
    "L-Glutamate":  {"ex": "EX_cpd00023_e", "fc_obs": 12.0,
                     "note": "TCA/AA hub; flavor core"},
    "L-Serine":     {"ex": "EX_cpd00054_e", "fc_obs": 8.5,
                     "note": "One-carbon metabolism; Nattokinase residue"},
    "L-Valine":     {"ex": "EX_cpd00156_e", "fc_obs": 76.4,
                     "note": "BCAA; natto aroma precursor"},
    "L-Leucine":    {"ex": "EX_cpd00107_e", "fc_obs": 76.4,
                     "note": "BCAA; acetolactate pathway"},
    "Glycine":      {"ex": "EX_cpd00033_e", "fc_obs": 26.0,
                     "note": "Purine/heme precursor; natto-up 26x"},
}

N_STEPS    = 12   # FSEOF 扫描步数（多一点提高线性回归质量）
N_TOP      = 20   # 每个目标取前 N 工程靶点

fseof_all = {}    # target → fseof_df

for prod_name, info in FSEOF_TARGETS.items():
    ex_id = info["ex"]
    log.info(f"  [{prod_name}] computing practical max ...")

    # 计算实用最大产量（80% biomass）
    with model_ref as m:
        apply_natto_constraints(m)
        bio_max = m.slim_optimize()
        if bio_max is None or bio_max < 1e-9:
            log.warning(f"    {prod_name}: biomass infeasible, skip")
            continue
        m.reactions.get_by_id(BIOMASS).lower_bound = 0.8 * bio_max
        m.objective = ex_id
        m.objective_direction = "max"
        sol = m.optimize()
        prac_max = sol.objective_value if sol.status == "optimal" else 0.0

    if prac_max < 1e-6:
        log.warning(f"    {prod_name}: prac_max ≈ 0 (model gap), skip FSEOF")
        continue
    log.info(f"    prac_max = {prac_max:.2f} mmol/gDW/h")

    scan_lbs = np.linspace(0, prac_max * 0.92, N_STEPS)
    flux_matrix = {}

    with model_ref as model:
        apply_natto_constraints(model)
        bio_max2 = model.slim_optimize()
        model.reactions.get_by_id(BIOMASS).lower_bound = 0.8 * bio_max2

        for lb in scan_lbs:
            with model:
                model.reactions.get_by_id(ex_id).lower_bound = float(lb)
                model.objective = BIOMASS
                sol = model.optimize()
                if sol.status != "optimal":
                    continue
                for rxn_id, v in sol.fluxes.items():
                    flux_matrix.setdefault(rxn_id, []).append(v)

    if not flux_matrix:
        log.warning(f"    {prod_name}: no feasible solutions found")
        continue

    # 线性回归：通量 ~ 产量约束步骤索引
    n_valid = max(len(v) for v in flux_matrix.values())
    x_arr = np.arange(n_valid, dtype=float)

    fseof_rows = []
    for rxn_id, flux_list in flux_matrix.items():
        if len(flux_list) < 4:
            continue
        y = np.array(flux_list[:n_valid])
        x = x_arr[:len(y)]
        if np.std(y) < 1e-12:
            continue
        slope, intercept = np.polyfit(x, y, 1)

        try:
            rxn   = model_ref.reactions.get_by_id(rxn_id)
            rname = rxn.name[:55]
            gpr   = rxn.gene_reaction_rule[:60] if rxn.gene_reaction_rule else ""
            ngenes= len(rxn.genes)
        except Exception:
            rname, gpr, ngenes = "", "", 0

        fseof_rows.append({
            "reaction":    rxn_id,
            "rxn_name":    rname,
            "slope":       round(slope, 6),
            "intercept":   round(intercept, 4),
            "gpr":         gpr,
            "n_genes":     ngenes,
            "target_type": ("overexpress" if slope > 0.01
                            else ("knockout" if slope < -0.01 else "neutral")),
        })

    fseof_df = pd.DataFrame(fseof_rows).sort_values("slope", key=abs, ascending=False)

    # 筛选有基因注释、非 exchange、非 biomass 的靶点
    eng_df = fseof_df[
        (fseof_df["n_genes"] > 0) &
        (~fseof_df["reaction"].str.startswith("EX_")) &
        (~fseof_df["reaction"].str.startswith("bio")) &
        (fseof_df["slope"].abs() > 0.05)
    ].head(N_TOP).copy()
    eng_df["target_product"] = prod_name

    if eng_df.empty:
        log.warning(f"    {prod_name}: no targets after filtering (all neutral/EX/no-gene), skip")
        continue

    # 对斜率做 per-unit 归一化：除以 prac_max，消除不同目标量级差异
    eng_df["slope_norm"] = eng_df["slope"] / prac_max

    fseof_all[prod_name] = eng_df
    safe = prod_name.replace("-","_").replace(" ","_")
    eng_df.to_csv(OUTPUT / f"07_fseof_{safe}.csv", index=False, encoding="utf-8-sig")
    log.info(f"    -> {len(eng_df)} targets | top slope = {eng_df['slope'].iloc[0]:.3f} "
             f"(norm={eng_df['slope_norm'].iloc[0]:.4f})")

# 合并所有结果
if fseof_all:
    combined = pd.concat(fseof_all.values(), ignore_index=True)
    combined.to_csv(OUTPUT / "07_fseof_multitarget.csv", index=False, encoding="utf-8-sig")
    log.info(f"  07_fseof_multitarget.csv saved ({len(combined)} rows)")

# ── 图 9：多目标 FSEOF 热图 ────────────────────────────────────
log.info("Drawing fig9_fseof_heatmap.png ...")

if fseof_all:
    # 找各目标 Top-10 反应的并集（有基因，非EX，非bio）
    top_rxns_per_target = {}
    for prod, df in fseof_all.items():
        top_rxns_per_target[prod] = list(df.head(10)["reaction"])

    union_rxns = list(dict.fromkeys(
        rxn for lst in top_rxns_per_target.values() for rxn in lst
    ))[:30]  # 最多 30 行

    # 构建热图矩阵 (rxn × target → slope_norm，归一化消除量级差异)
    targets_list = list(fseof_all.keys())
    heat_data = pd.DataFrame(index=union_rxns, columns=targets_list, dtype=float)
    heat_data[:] = np.nan

    for prod, df in fseof_all.items():
        col_key = "slope_norm" if "slope_norm" in df.columns else "slope"
        for _, row in df.iterrows():
            if row["reaction"] in heat_data.index:
                heat_data.loc[row["reaction"], prod] = row[col_key]

    # 反应名称映射（用于 y 轴标签）
    rxn_label_map = {}
    for rid in union_rxns:
        try:
            name = model_ref.reactions.get_by_id(rid).name[:40]
            rxn_label_map[rid] = f"{rid}\n({name})" if name else rid
        except Exception:
            rxn_label_map[rid] = rid

    heat_data.index = [rxn_label_map.get(r, r) for r in heat_data.index]

    # 热图：行 = 反应，列 = 目标代谢物
    fig_h = max(8, len(heat_data) * 0.38)
    fig, ax = plt.subplots(figsize=(len(targets_list)*1.8 + 1.5, fig_h))

    # 用 seaborn heatmap（diverging colormap）
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    vabs = np.nanmax(np.abs(heat_data.values.astype(float))) * 1.05
    if vabs < 0.01:
        vabs = 1.0

    mask = heat_data.isna()
    sns.heatmap(
        heat_data.astype(float),
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-vabs, vmax=vabs,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 6.5},
        linewidths=0.4, linecolor="#DDDDDD",
        cbar_kws={"label": "FSEOF Slope", "shrink": 0.6},
    )
    ax.set_xlabel("Target Metabolite", fontsize=9, labelpad=6)
    ax.set_ylabel("Reaction (Reaction ID + Name)", fontsize=9, labelpad=6)
    ax.set_title(
        "Multi-Target FSEOF Engineering Targets\n"
        "(Slope: reaction flux change per unit of target metabolite secretion;\n"
        " Blue = knockout target; Red = overexpression target)",
        fontsize=9.5, fontweight="bold", pad=10
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)

    save(fig, "fig9_fseof_heatmap.png")
else:
    log.warning("No FSEOF results available for heatmap")

# ══════════════════════════════════════════════════════════════
# TASK 2：多情景敏感性分析
# ══════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 2: Sensitivity analysis ...")
log.info("=" * 60)

RISK_METS = {
    "Phenethylamine":    {"ex": "EX_cpd03161_e", "adi_norm": 0.05},
    "PAGln (proxy)":     {"ex": "EX_cpd00053_e", "adi_norm": 0.20},
    "L-Arg (ADMA proxy)":{"ex": "EX_cpd03535_e", "adi_norm": 0.30},
}
BIOMASS_FRACS = [0.50, 0.75, 0.90, 1.00]   # biomass 约束强度
FC_THRESHOLDS = [2.0, 5.0, 10.0]           # 施加约束的最低 FC 阈值

sens_rows = []

for bio_frac in BIOMASS_FRACS:
    for fc_thresh in FC_THRESHOLDS:
        log.info(f"  biomass_frac={bio_frac:.2f}  fc_thresh={fc_thresh:.0f} ...")

        with model_ref as m:
            apply_natto_constraints(m, fc_thresh=fc_thresh)
            bio_max = m.slim_optimize()
            if bio_max is None or bio_max < 1e-9:
                log.warning("    infeasible, skip")
                continue
            m.reactions.get_by_id(BIOMASS).lower_bound = bio_frac * bio_max
            m.objective = BIOMASS
            sol = m.optimize()
            if sol.status != "optimal":
                log.warning("    not optimal, skip")
                continue

            ri_total = 0.0
            ri_met_vals = {}
            for met_name, minfo in RISK_METS.items():
                ex_id = minfo["ex"]
                cfba_v = sol.fluxes.get(ex_id, 0.0)

                # FVA max 在此情景下
                try:
                    from cobra.flux_analysis import flux_variability_analysis as fva_fn
                    fva_res = fva_fn(m, [m.reactions.get_by_id(ex_id)],
                                     fraction_of_optimum=bio_frac,
                                     processes=1)
                    fva_max = fva_res.loc[ex_id, "maximum"]
                except Exception:
                    fva_max = cfba_v if cfba_v > 0 else 10000.0

                norm = (cfba_v / fva_max) if fva_max > 1e-9 else (1.0 if cfba_v > 0 else 0.0)
                ri = norm / minfo["adi_norm"]
                ri_met_vals[met_name] = round(ri, 4)
                ri_total += ri

            n_constraints = apply_natto_constraints(m, fc_thresh=fc_thresh)

        row = {
            "biomass_frac":  bio_frac,
            "fc_threshold":  fc_thresh,
            "n_constraints": n_constraints,
            "RI_total":      round(ri_total, 4),
        }
        row.update({f"RI_{k}": v for k, v in ri_met_vals.items()})
        sens_rows.append(row)
        log.info(f"    n_constraints={n_constraints}  RI_total={ri_total:.3f}")

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUTPUT / "07_sensitivity.csv", index=False, encoding="utf-8-sig")
log.info(f"  07_sensitivity.csv saved ({len(sens_df)} rows)")

# ── 图 10：敏感性折线图 ────────────────────────────────────────
log.info("Drawing fig10_sensitivity.png ...")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# Panel A：固定 fc_thresh=5，RI_total vs biomass_frac（分 FC 组）
ax = axes[0]
met_colors_sens = {
    "Phenethylamine":    GY,
    "PAGln (proxy)":     B,
    "L-Arg (ADMA proxy)":R,
}
for fc_t, ls in zip(FC_THRESHOLDS, ["-", "--", ":"]):
    sub = sens_df[sens_df["fc_threshold"] == fc_t].sort_values("biomass_frac")
    if sub.empty: continue
    ax.plot(sub["biomass_frac"] * 100, sub["RI_total"],
            marker="o", ms=5.5, lw=1.8, ls=ls,
            color=OR if fc_t == 2 else (B if fc_t == 5 else R),
            label=f"FC ≥ {fc_t:.0f} constraint")
ax.axhline(1.0, color="#AAAAAA", lw=0.9, ls="--", zorder=1, label="RI = 1.0 threshold")
ax.set_xlabel("Biomass Constraint (% of max growth)", fontsize=9)
ax.set_ylabel("Composite Risk Index (RI$_{total}$)", fontsize=9)
ax.set_title("Sensitivity: RI$_{total}$ vs. Biomass Constraint\nfor Different FC Thresholds", fontsize=9.5)
ax.legend(fontsize=7.5, framealpha=0.9)
ax.set_xticks([50, 75, 90, 100])
plbl(ax, "A")

# Panel B：固定 biomass_frac=0.9，各代谢物 RI vs FC 阈值
ax2 = axes[1]
sub90 = sens_df[sens_df["biomass_frac"] == 0.90].sort_values("fc_threshold")
ri_cols = [c for c in sens_df.columns if c.startswith("RI_") and "total" not in c.lower()]
for rc, col in zip(ri_cols,
                   [GY, B, R, OR, GR, PU]):
    if rc not in sub90.columns: continue
    met_short = rc.replace("RI_", "").replace(" (proxy)","").replace(" (ADMA proxy)","")
    ax2.plot(sub90["fc_threshold"], sub90[rc],
             marker="s", ms=5.5, lw=1.8, color=col,
             label=met_short)
ax2.axhline(1.0, color="#AAAAAA", lw=0.9, ls="--", zorder=1, label="RI = 1.0")
ax2.set_xlabel("FC Threshold for Applying Constraints", fontsize=9)
ax2.set_ylabel("Individual Risk Index (RI)", fontsize=9)
ax2.set_title("Sensitivity: Individual RI vs. FC Threshold\n(Biomass constraint = 90%)", fontsize=9.5)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.set_xticks(FC_THRESHOLDS)
plbl(ax2, "B")

save(fig, "fig10_sensitivity.png")

# ══════════════════════════════════════════════════════════════
# TASK 3：锚定代谢物通路富集气泡图
# ══════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 3: Pathway enrichment bubble chart ...")
log.info("=" * 60)

mapped = pd.read_csv(OUTPUT / "01_mapped_metabolites.csv")

# ── 展开 pathways_s2（pipe-separated），同时用 cls 作为备用分类 ──
# 去掉 "- Glycine max (soybean)" 后缀
def clean_pathway(s):
    s = s.strip()
    s = s.replace(" - Glycine max (soybean)", "").replace(" - Bacillus subtilis", "")
    # 去掉括号里的数字 "(12)"
    import re
    s = re.sub(r'\s*\(\d+\)\s*$', '', s)
    return s.strip()

def simplify_cls(c):
    """将复合 cls 字符串（分号分隔）取主类"""
    if not isinstance(c, str): return "Others"
    parts = [p.strip() for p in c.split(";")]
    # 优先返回含 amino acid 的类
    for p in parts:
        if "Amino acid" in p: return "Amino acids & derivatives"
    return parts[0]

path_rows = []
for _, row in mapped.iterrows():
    pathways_raw = str(row.get("pathways_s2", ""))
    cls_main = simplify_cls(str(row.get("cls", "")))
    fc       = float(row["fc"]) if not pd.isna(row.get("fc", np.nan)) else 1.0
    direction= str(row.get("direction", ""))
    name     = str(row.get("name", ""))

    if pathways_raw and pathways_raw not in ("nan", ""):
        pathways = [clean_pathway(p) for p in pathways_raw.split("|") if p.strip()]
        # 去掉"Metabolic pathways"这种通太宽的注释
        pathways = [p for p in pathways
                    if p not in ("Metabolic pathways", "Biosynthesis of secondary metabolites")]
    else:
        pathways = []

    # 若无通路信息，用 cls 作为通路名称
    if not pathways:
        pathways = [cls_main]

    for pw in pathways:
        if not pw or len(pw) < 4: continue
        path_rows.append({
            "pathway":    pw,
            "metabolite": name,
            "fc":         fc,
            "log2fc":     np.log2(fc) if fc > 0 else 0.0,
            "direction":  direction,
            "cls_main":   cls_main,
        })

path_long = pd.DataFrame(path_rows)

# 统计每个通路
pw_stats = path_long.groupby("pathway").agg(
    n_metabolites = ("metabolite", "nunique"),
    mean_log2fc   = ("log2fc", "mean"),
    mean_fc       = ("fc", "mean"),
    n_natto_up    = ("direction", lambda x: (x == "natto_up").sum()),
    n_soy_up      = ("direction", lambda x: (x == "soy_up").sum()),
).reset_index()

pw_stats["dominant"] = pw_stats.apply(
    lambda r: "natto_up" if r["n_natto_up"] >= r["n_soy_up"] else "soy_up", axis=1
)
pw_stats["pct_natto"] = pw_stats["n_natto_up"] / pw_stats["n_metabolites"]

# 过滤：至少 2 个代谢物命中
pw_stats = pw_stats[pw_stats["n_metabolites"] >= 2].copy()
# 按代谢物数降序
pw_stats = pw_stats.sort_values("mean_log2fc", ascending=False).reset_index(drop=True)

pw_stats.to_csv(OUTPUT / "07_pathway_enrichment.csv", index=False, encoding="utf-8-sig")
log.info(f"  07_pathway_enrichment.csv saved ({len(pw_stats)} pathways)")

# ── 图 11：通路富集气泡图 ─────────────────────────────────────
log.info("Drawing fig11_pathway_bubble.png ...")

# 取前 20 个通路（按平均 log2FC 排序）
pw_plot = pw_stats.head(20).copy()
# 缩短通路名
def shorten_pw(s, maxlen=42):
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

pw_plot["pw_label"] = pw_plot["pathway"].apply(shorten_pw)
pw_plot = pw_plot.sort_values("mean_log2fc", ascending=True)  # 最高FC在顶部

fig, ax = plt.subplots(figsize=(9.5, 7.5))

# 气泡颜色：主要方向
bubble_colors = [R if d == "natto_up" else B for d in pw_plot["dominant"]]
# 气泡大小：代谢物数
bubble_sizes  = (pw_plot["n_metabolites"] * 55).clip(80, 900)

sc = ax.scatter(
    pw_plot["mean_log2fc"],
    range(len(pw_plot)),
    s=bubble_sizes,
    c=bubble_colors,
    alpha=0.78,
    edgecolors="white",
    linewidths=0.7,
    zorder=5,
)

# 数值标注
for i, row in pw_plot.reset_index(drop=True).iterrows():
    ax.text(row["mean_log2fc"] + 0.12, i,
            f"n={row['n_metabolites']}  log₂FC={row['mean_log2fc']:.1f}",
            va="center", fontsize=6.8, color="#333333")

ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
ax.set_yticks(range(len(pw_plot)))
ax.set_yticklabels(pw_plot["pw_label"].tolist(), fontsize=7.8)
ax.set_xlabel("Mean log₂(Fold Change) of Anchor Metabolites", fontsize=9)
ax.set_title(
    "KEGG Pathway Enrichment of Anchor Metabolites\n"
    "(Bubble size = number of metabolites; Red = Natto-enriched; Blue = Soy-enriched)",
    fontsize=9.5, fontweight="bold", pad=10
)

# 图例
legend_sizes = [2, 4, 8]
for sz in legend_sizes:
    ax.scatter([], [], s=sz*55, c=GY, alpha=0.65, label=f"n = {sz} metabolites",
               edgecolors="white", linewidths=0.6)
ax.scatter([], [], c=R, s=120, alpha=0.78, label="Natto-enriched dominant",
           edgecolors="white", linewidths=0.6)
ax.scatter([], [], c=B, s=120, alpha=0.78, label="Soy-enriched dominant",
           edgecolors="white", linewidths=0.6)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
ax.set_axisbelow(True)

save(fig, "fig11_pathway_bubble.png")

# ══════════════════════════════════════════════════════════════
# 复制到 paper/figures/
# ══════════════════════════════════════════════════════════════
import shutil
PAPER_FIG = OUTPUT.parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

new_figs = ["fig9_fseof_heatmap.png", "fig10_sensitivity.png", "fig11_pathway_bubble.png"]
copied = 0
for fn in new_figs:
    src = OUTPUT / fn
    if src.exists():
        shutil.copy(src, PAPER_FIG / fn)
        copied += 1

log.info(f"Done — {copied} new figures copied to paper/figures/")

# ══════════════════════════════════════════════════════════════
# 控制台摘要
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Extended Analysis Complete (Tasks 1-3)")
print("=" * 65)
print()

print("  [Task 1] FSEOF Multi-target Results:")
for prod, df in fseof_all.items():
    print(f"    {prod:15s}: {len(df):2d} targets | "
          f"top = {df['reaction'].iloc[0]} (slope={df['slope'].iloc[0]:.3f})")
print()

print("  [Task 2] Sensitivity Analysis:")
print(f"    Scenarios: {len(sens_df)} combinations "
      f"({len(BIOMASS_FRACS)} biomass × {len(FC_THRESHOLDS)} FC thresholds)")
if not sens_df.empty:
    ri_range = f"{sens_df['RI_total'].min():.3f} – {sens_df['RI_total'].max():.3f}"
    print(f"    RI_total range: {ri_range}")
print()

print("  [Task 3] Pathway Enrichment:")
print(f"    Qualified pathways (n_met ≥ 2): {len(pw_stats)}")
if not pw_stats.empty:
    top3 = pw_stats.head(3)
    for _, r in top3.iterrows():
        print(f"    {r['pathway'][:45]:45s}  "
              f"n={r['n_metabolites']}  log2FC={r['mean_log2fc']:.2f}")
print()

print("  Output files:")
for fn in ["07_fseof_multitarget.csv", "07_sensitivity.csv",
           "07_pathway_enrichment.csv",
           "fig9_fseof_heatmap.png", "fig10_sensitivity.png",
           "fig11_pathway_bubble.png"]:
    status = "[OK]" if (OUTPUT / fn).exists() else "[--]"
    print(f"    {status}  code/output/{fn}")
print("=" * 65)
