"""
08_redraw_figs.py — 重绘 fig9/10/11 + 新增 fig12 通路流程图
Times New Roman字体 · 鲜艳配色 · 紧凑无留白
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import utils
from utils import OUTPUT

# ── 全局字体 & 样式 ────────────────────────────────────────────
def setup():
    import matplotlib as mpl; mpl.rcdefaults()
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#222222",
        "axes.linewidth":   0.9,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        False,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.labelsize":  8.5,
        "ytick.labelsize":  8.5,
        "axes.labelsize":   9.5,
        "axes.titlesize":   10,
        "axes.titleweight": "bold",
        "legend.fontsize":  8,
        "legend.frameon":   True,
        "legend.framealpha":0.9,
        "legend.edgecolor": "#CCCCCC",
        "lines.linewidth":  1.8,
        "patch.linewidth":  0.5,
    })

setup()

# 鲜艳配色（science research palette）
RED   = "#E63946"
BLUE  = "#1D6FA4"
ORG   = "#F4A261"
GRN   = "#2A9D8F"
PUR   = "#7B2D8B"
YLW   = "#E9C46A"
GY    = "#999999"
DBLUE = "#264653"

def save(fig, name):
    p = OUTPUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {name}  ({p.stat().st_size//1024} KB)")

def plbl(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")

# ══════════════════════════════════════════════════════════════
# FIG 9：FSEOF 多目标热图（紧凑版，TNR字体）
# ══════════════════════════════════════════════════════════════
print("Redrawing fig9 ...")
df_all = pd.read_csv(OUTPUT / "07_fseof_multitarget.csv")

targets_list = [t for t in df_all["target_product"].unique() if t != "Glycine"]
top_rxns_per_target = {}
for prod in targets_list:
    sub = df_all[df_all["target_product"]==prod]
    top_rxns_per_target[prod] = list(sub.head(8)["reaction"])

union_rxns = list(dict.fromkeys(
    r for lst in top_rxns_per_target.values() for r in lst
))[:22]

import cobra
model_ref = cobra.io.read_sbml_model(str(utils.DATA / "MODEL1507180015_url.xml"))

heat = pd.DataFrame(index=union_rxns, columns=targets_list, dtype=float)
heat[:] = np.nan
for prod in targets_list:
    sub = df_all[df_all["target_product"]==prod]
    col_key = "slope_norm" if "slope_norm" in sub.columns else "slope"
    for _, row in sub.iterrows():
        if row["reaction"] in heat.index:
            heat.loc[row["reaction"], prod] = row[col_key]

# y 轴标签：reaction ID → 简短名称
RXN_SHORT = {
    "rxn05617": "Mannitol PTS",
    "rxn00546": "Mannitol-1P DH",
    "rxn05569": "GlcN PTS",
    "rxn01236": "Butyrate kinase",
    "rxn00994": "Butyryl-CoA transferase",
    "rxn00871": "Phosphotransbutyrylase",
    "rxn00988": "Acetoacetate-CoA ligase",
    "rxn05559": "Formate transporter",
    "rxn05298": "Glu transporter (GLUt4i)",
    "rxn05500": "Arabinose transporter",
    "rxn04082": "L-Ribulose-5P epimerase",
    "rxn01292": "Arabinose isomerase",
    "rxn01763": "Ribulokinase",
    "rxn05568": "GlcN6P uniport",
    "rxn00555": "GlcN6P synthase",
    "rxn05216": "Gln Na$^+$ symporter",
    "rxn05582": "Gly transporter",
    "rxn05307": "Ser transporter (SERt2)",
    "rxn05672": "D-Glu transporter",
    "rxn00193": "Glu racemase",
    "rxn00692": "SHMT",
    "rxn00908": "Glycine synthase",
}

def rxn_label(rid):
    if rid in RXN_SHORT:
        return RXN_SHORT[rid]
    try:
        nm = model_ref.reactions.get_by_id(rid).name
        if nm:
            return nm[:24] if len(nm) > 24 else nm
    except:
        pass
    return rid

heat.index = [rxn_label(r) for r in heat.index]

fig_h = max(6.0, len(heat)*0.33)
fig, ax = plt.subplots(figsize=(len(targets_list)*1.7+1.2, fig_h))

cmap = sns.diverging_palette(220, 15, s=90, l=45, as_cmap=True)
vabs = float(np.nanmax(np.abs(heat.values.astype(float)))) * 1.05
if vabs < 0.01: vabs = 0.15

mask = heat.isna()
sns.heatmap(heat.astype(float), ax=ax,
            cmap=cmap, center=0, vmin=-vabs, vmax=vabs,
            mask=mask,
            annot=True, fmt=".2f", annot_kws={"size":7, "fontfamily":"Times New Roman"},
            linewidths=0.35, linecolor="#CCCCCC",
            cbar_kws={"label":"Normalized FSEOF Slope", "shrink":0.55,
                      "aspect":20})

ax.set_xlabel("Target Metabolite", fontsize=9.5, labelpad=5)
ax.set_ylabel("Reaction", fontsize=9.5, labelpad=5)
ax.set_title("Multi-Target FSEOF: Engineering Targets Across Six Amino Acids\n"
             "Blue = Knockout  |  Red = Overexpression  |  Grey = Not applicable",
             fontsize=9.5, pad=6)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)
# colorbar字体
ax.collections[0].colorbar.ax.tick_params(labelsize=8)
ax.collections[0].colorbar.ax.yaxis.label.set_fontsize(8.5)

save(fig, "fig9_fseof_heatmap.png")

# ══════════════════════════════════════════════════════════════
# FIG 10：敏感性折线图（紧凑版，TNR字体）
# ══════════════════════════════════════════════════════════════
print("Redrawing fig10 ...")
sens = pd.read_csv(OUTPUT / "07_sensitivity.csv")
FC_THRESHOLDS = [2.0, 5.0, 10.0]
fc_colors  = {2.0: ORG, 5.0: BLUE, 10.0: RED}
fc_markers = {2.0: "o",  5.0: "s",  10.0: "D"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Panel A：RI_total vs biomass_frac，分 FC 组
for fc_t in FC_THRESHOLDS:
    sub = sens[sens["fc_threshold"]==fc_t].sort_values("biomass_frac")
    ax1.plot(sub["biomass_frac"]*100, sub["RI_total"],
             marker=fc_markers[fc_t], ms=6, lw=1.8,
             color=fc_colors[fc_t], label=f"FC ≥ {fc_t:.0f}")
ax1.axhline(1.0, color=GY, lw=1.0, ls="--", zorder=1, label="RI = 1.0 (threshold)")
ax1.fill_between([48, 102], 0, 1.0, color="#E8F5E9", alpha=0.4, zorder=0)
ax1.set_xlim(48, 102)
ax1.set_xticks([50,75,90,100])
ax1.set_xlabel("Biomass Constraint (% of max)")
ax1.set_ylabel("Composite Risk Index (RI$_{total}$)")
ax1.set_title("RI$_{total}$ vs. Biomass Constraint\n(colored by FC threshold)")
ax1.legend(fontsize=8, loc="upper right")
plbl(ax1, "A")

# Panel B：各代谢物 RI vs FC 阈值（biomass=90%）
sub90 = sens[sens["biomass_frac"]==0.90].sort_values("fc_threshold")
ri_cols = [c for c in sens.columns if c.startswith("RI_") and "total" not in c.lower()]
met_colors = [GY, BLUE, RED]
met_labels = ["PEA (biogenic amine)", "PAGln (pro-thrombotic)", "L-Arg / ADMA (eNOS)"]
for rc, col, lbl in zip(ri_cols, met_colors, met_labels):
    if rc not in sub90.columns: continue
    ax2.plot(sub90["fc_threshold"], sub90[rc],
             marker="o", ms=6, lw=1.8, color=col, label=lbl)
ax2.axhline(1.0, color=GY, lw=1.0, ls="--", zorder=1)
ax2.fill_between([1.5, 10.5], 0, 1.0, color="#E8F5E9", alpha=0.4, zorder=0)
ax2.set_xlim(1.5, 10.5)
ax2.set_xticks([2,5,10])
ax2.set_xlabel("FC Threshold for Applying Constraints")
ax2.set_ylabel("Individual Risk Index (RI)")
ax2.set_title("Individual RI vs. FC Threshold\n(Biomass constraint = 90%)")
ax2.legend(fontsize=8, loc="upper right")
ax2.annotate("PEA: always 0\n(model gap)", xy=(2, 0), xytext=(3.5, 0.5),
             fontsize=7.5, color=GY,
             arrowprops=dict(arrowstyle="->", color=GY, lw=0.8))
plbl(ax2, "B")

plt.tight_layout(pad=0.8, w_pad=1.5)
save(fig, "fig10_sensitivity.png")

# ══════════════════════════════════════════════════════════════
# FIG 11：通路富集气泡图（紧凑版，TNR字体）
# ══════════════════════════════════════════════════════════════
print("Redrawing fig11 ...")
pw = pd.read_csv(OUTPUT / "07_pathway_enrichment.csv")
pw_plot = pw.head(18).copy()

PW_SHORT = {
    "Tropane, piperidine and pyridine alkaloid biosynthesis": "Tropane/piperidine alkaloids",
    "Valine, leucine and isoleucine degradation":             "Val/Leu/Ile degradation",
    "Valine, leucine and isoleucine biosynthesis":            "Val/Leu/Ile biosynthesis",
    "Biosynthesis of secondary metabolites - unclassified":   "2° metabolite biosynthesis",
    "Benzene and substituted derivatives":                    "Benzene derivatives",
    "2-Oxocarboxylic acid metabolism":                        "2-Oxocarboxylic acid metab.",
    "Arginine and proline metabolism":                        "Arg/Pro metabolism",
    "Biosynthesis of amino acids":                            "Amino acid biosynthesis",
    "Aminoacyl-tRNA biosynthesis":                            "Aminoacyl-tRNA biosyn.",
    "Cyanoamino acid metabolism":                             "Cyanoamino acid metab.",
    "Glucosinolate biosynthesis":                             "Glucosinolate biosyn.",
    "Porphyrin and chlorophyll metabolism":                   "Porphyrin/chlorophyll metab.",
}

def shorten(s, n=30):
    if s in PW_SHORT:
        return PW_SHORT[s]
    return s if len(s) <= n else s[:n]

pw_plot["lbl"] = pw_plot["pathway"].apply(shorten)
pw_plot = pw_plot.sort_values("mean_log2fc", ascending=True)

fig, ax = plt.subplots(figsize=(9.5, 7.0))

colors = [RED if d=="natto_up" else BLUE for d in pw_plot["dominant"]]
sizes  = (pw_plot["n_metabolites"] * 60).clip(90, 850)

ax.scatter(pw_plot["mean_log2fc"], range(len(pw_plot)),
           s=sizes, c=colors, alpha=0.82,
           edgecolors="white", linewidths=0.8, zorder=5)

xmax = pw_plot["mean_log2fc"].max()
for i, row in pw_plot.reset_index(drop=True).iterrows():
    ax.text(row["mean_log2fc"]+0.15, i,
            f"n={int(row['n_metabolites'])}  ({row['mean_log2fc']:.1f})",
            va="center", fontsize=7, color="#222222")

ax.axvline(0, color=GY, lw=0.8, ls="--", zorder=1)
ax.set_xlim(pw_plot["mean_log2fc"].min() - 0.5, xmax + 1.8)
ax.set_yticks(range(len(pw_plot)))
ax.set_yticklabels(pw_plot["lbl"].tolist(), fontsize=8)
ax.set_xlabel("Mean log$_2$(Fold Change) of Anchor Metabolites")
ax.set_title("KEGG Pathway Enrichment of 76 Anchor Metabolites\n"
             r"Bubble size $\propto$ number of metabolites hit  |  Red=Natto-up, Blue=Soy-up",
             fontsize=9.5, pad=6)

# 图例
for sz, lbl in [(2,90),(4,240),(8,480)]:
    ax.scatter([],[], s=sz*60, c=GY, alpha=0.6,
               label=f"n = {sz}", edgecolors="white")
ax.scatter([],[], c=RED, s=140, alpha=0.85, label="Natto-enriched", edgecolors="white")
ax.scatter([],[], c=BLUE, s=140, alpha=0.85, label="Soy-enriched", edgecolors="white")
ax.legend(fontsize=8, loc="lower right", framealpha=0.92,
          handletextpad=0.4, labelspacing=0.3)
ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)
save(fig, "fig11_pathway_bubble.png")

# ══════════════════════════════════════════════════════════════
# FIG 12：分析流程总结图（cFBA pipeline overview）
# ══════════════════════════════════════════════════════════════
print("Drawing fig12 pipeline overview ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.0),
                          gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

# ── Panel A: 分析流程六步骤 ──────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 11.5)
ax.axis("off")
ax.set_title("(A)  cFBA Analysis Pipeline Overview",
             fontsize=10.5, fontweight="bold", pad=8, loc="left")

steps = [
    # (y_center, box_color, border_color, title, subtitle)
    (10.5, "#EBF5FB", "#2471A3",
     "① Metabolomics Input",
     "569 compounds · 160 sig. (p<0.05)\n125 natto-up · 36 soy-up"),
    (8.8,  "#EAF4EA", "#1A7A4A",
     "② Two-Level Mapping → iBsu1103",
     "76 anchor metabolites (47.5%)\n61 secretion + 14 uptake constraints"),
    (7.1,  "#FEF9E7", "#D4A017",
     "③ Constrained FBA (cFBA)",
     "75 directional constraints applied\nBiomass cost: 7.6% (1229.7 → 1136.5)"),
    (5.4,  "#FDEDEC", "#C0392B",
     "④ FVA + Risk Assessment",
     "397 bottleneck / 932 flexible reactions\nRI: PEA=0 (gap) · PAGln=0.68 · L-Arg=3.26"),
    (3.7,  "#F4ECF7", "#7B2D8B",
     "⑤ FSEOF Engineering Targets",
     "14 L-His targets · ptsG KO (−0.132)\nFormate OE (+0.133) · Mannitol PTS KO (−0.148)"),
    (2.0,  "#FDFEFE", "#5D6D7E",
     "⑥ Essential Gene Screen",
     "202 standard · 230 natto essential\n27 natto-specific (3 metabolic modules)"),
]

arrow_col = "#888888"
for i, (yc, fc, ec, title, sub) in enumerate(steps):
    # box
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((0.3, yc - 0.72), 9.4, 1.44,
                       boxstyle="round,pad=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(p)
    ax.text(0.75, yc + 0.22, title, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=ec, zorder=4)
    ax.text(0.75, yc - 0.22, sub, ha="left", va="center",
            fontsize=8.0, color="#333333", zorder=4)
    # arrow to next step
    if i < len(steps) - 1:
        y_next = steps[i+1][0]
        ax.annotate("", xy=(5.0, y_next + 0.72 + 0.08),
                    xytext=(5.0, yc - 0.72 - 0.08),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_col,
                                   lw=1.4, mutation_scale=14), zorder=2)

ax.text(-0.05, 1.02, "A", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

# ── Panel B: 三类风险代谢物机制对比 ─────────────────────────
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 11.5)
ax2.axis("off")
ax2.set_title("(B)  Risk Metabolite Mechanism Classification",
              fontsize=10.5, fontweight="bold", pad=8, loc="left")

# 三列标题
col_xs = [1.5, 5.0, 8.5]
col_titles = ["PEA\n(FC = 55×)", "PAGln\n(FC = 3.3×)", "L-Arg\n(FC = 25×)"]
col_colors = [RED, BLUE, ORG]
for cx, ct, cc in zip(col_xs, col_titles, col_colors):
    p2 = FancyBboxPatch((cx - 1.35, 9.8), 2.7, 1.3,
                        boxstyle="round,pad=0.1",
                        facecolor=cc, edgecolor=cc, linewidth=1.2, zorder=3)
    ax2.add_patch(p2)
    ax2.text(cx, 10.45, ct, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white", zorder=4)

# 行标签 + 内容
rows = [
    ("Mechanism",
     "Model gap\n(no AAAD gene)", "Substrate-driven\n(Gln supply)", "Substrate-driven\n(Arg from soy)"),
    ("RI (baseline)",
     "0  (undefined)", "5.00", "3.33"),
    ("RI (natto cFBA)",
     "0  (model gap)", "0.68", "3.26"),
    ("Intervention",
     "Gene-level\n(AAAD KO)", "Process control\n(protease activity)", "Process control\n(hydrolysis depth)"),
    ("Sensitivity",
     "Stable = 0\n(all 12 scenarios)", "High\n(FC threshold)", "Stable\n(2.26–3.33)"),
]
row_ys   = [8.2, 6.8, 5.6, 4.2, 2.8]
row_bgs  = ["#F8F9FA", "#FFFFFF", "#F8F9FA", "#FFFFFF", "#F8F9FA"]
for (rlbl, v1, v2, v3), ry, rbg in zip(rows, row_ys, row_bgs):
    # row background
    ax2.add_patch(FancyBboxPatch((0.1, ry - 0.55), 9.8, 1.1,
                                 boxstyle="square,pad=0",
                                 facecolor=rbg, edgecolor="#DDDDDD",
                                 linewidth=0.5, zorder=1))
    ax2.text(0.55, ry, rlbl, ha="center", va="center",
             fontsize=8.0, fontweight="bold", color="#555555", zorder=3)
    for cx, val, cc in zip(col_xs, [v1, v2, v3], col_colors):
        ax2.text(cx, ry, val, ha="center", va="center",
                 fontsize=8.0, color=cc if "0" not in val or "gap" in val.lower() else "#222222",
                 zorder=3)

# 分隔线
ax2.plot([0.1, 9.9], [9.8, 9.8], color="#CCCCCC", lw=0.8)
for ry in row_ys:
    ax2.plot([0.1, 9.9], [ry + 0.55, ry + 0.55], color="#EEEEEE", lw=0.5)

ax2.text(-0.05, 1.02, "B", transform=ax2.transAxes,
         fontsize=14, fontweight="bold", va="top")

plt.tight_layout(pad=0.5, w_pad=0.8)
save(fig, "fig12_pathway_diagram.png")

# ══════════════════════════════════════════════════════════════
# FIG 15：AAAD候选基因诊断评分图
# ══════════════════════════════════════════════════════════════
print("Drawing fig15 AAAD scoring ...")

candidates = [
    ("BSNT_RS03085\n(PLP-decarboxylase)", 17),
    ("SpeA\n(Arg decarboxylase)",           4),
    ("BsdC\n(BSNT_RS02105)",                3),
    ("PadC\n(BSNT_RS18045)",                1),
]
feat_rows = [
    ("Protein length ~480 aa",              [True,  False, False, False]),
    ("Schiff-base Lys (K303)",              [True,  False, False, False]),
    ("Substrate Asp (D271)",                [True,  False, False, False]),
    ("GxxGxxG PLP-binding motif",           [True,  False, False, False]),
    ("SxxxK phosphate contact",             [True,  False, False, False]),
    ("Natto-specific genomic insertion",    [True,  False, False, False]),
    ("Upstream locus transcribed (natto)",  [True,  False, False, False]),
    ("Aromatic amino acid substrate",       [True,  False, True,  False]),
    ("Complete PLP fold-type I site",       [True,  False, False, False]),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                gridspec_kw={"width_ratios": [1, 1.3]})

names  = [c[0] for c in candidates]
scores = [c[1] for c in candidates]
bar_cols = [RED if s >= 15 else (ORG if s >= 8 else GY) for s in scores]
bars = ax1.barh(range(len(candidates)), scores, color=bar_cols,
                height=0.55, edgecolor="white", linewidth=0.5)
ax1.set_yticks(range(len(candidates)))
ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlabel("Diagnostic Score (out of 20)")
ax1.set_title("AAAD Candidate Scoring\n(BEST195 genome-wide decarboxylase screen)")
ax1.axvline(15, color=RED, lw=1.0, ls="--", alpha=0.7, label="Strong candidate (≥15/20)")
for bar, sc in zip(bars, scores):
    ax1.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             f"{sc}/20", va="center", fontsize=9.5, fontweight="bold")
ax1.set_xlim(0, 23)
ax1.legend(fontsize=8, loc="lower right")
plbl(ax1, "A")

feat_names = [f[0] for f in feat_rows]
cand_short = ["BSNT_RS03085", "SpeA", "BsdC", "PadC"]
mat = np.array([[1 if v else 0 for v in f[1]] for f in feat_rows], dtype=float)
cmap2 = matplotlib.colors.ListedColormap(["#FFCCCC", "#CCEECC"])
ax2.imshow(mat, cmap=cmap2, vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(4))
ax2.set_xticklabels(cand_short, fontsize=9, rotation=25, ha="right")
ax2.set_yticks(range(len(feat_names)))
ax2.set_yticklabels(feat_names, fontsize=8)
ax2.set_title("Diagnostic Feature Comparison\n(Green = present, Red = absent)")
for i in range(len(feat_names)):
    for j in range(4):
        sym = "✓" if mat[i, j] else "✗"
        col = "#1A5C1A" if mat[i, j] else "#8B0000"
        ax2.text(j, i, sym, ha="center", va="center", fontsize=11, color=col,
                 fontfamily="DejaVu Sans")
plbl(ax2, "B")

plt.tight_layout(pad=0.6, w_pad=1.2)
save(fig, "fig15_aaad_scoring.png")

# ── 复制到 paper/figures/ ──────────────────────────────────────
PAPER_FIG = OUTPUT.parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

for fn in ["fig9_fseof_heatmap.png","fig10_sensitivity.png",
           "fig11_pathway_bubble.png","fig12_pathway_diagram.png",
           "fig15_aaad_scoring.png"]:
    src = OUTPUT / fn
    if src.exists():
        shutil.copy(src, PAPER_FIG / fn)

print("\nDone — fig9/10/11/12 redrawn and copied to paper/figures/")
