import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import utils
from utils import OUTPUT, DATA

log = utils.get_logger("06_figures")

# ═══════════════════════════════════════════════════════════════
#  全局样式  ——  Nature / Cell 期刊风格
# ═══════════════════════════════════════════════════════════════
def setup():
    import matplotlib as mpl; mpl.rcdefaults()
    plt.rcParams.update({
        "font.family":               ["Times New Roman"],
        "mathtext.fontset":          "stix",
        "axes.unicode_minus":        False,
        "savefig.dpi":               300,
        "savefig.bbox":              "tight",
        "savefig.pad_inches":        0.06,
        "figure.facecolor":          "white",
        "axes.facecolor":            "white",
        "axes.edgecolor":            "#333333",
        "axes.linewidth":            0.9,
        "axes.spines.top":           False,
        "axes.spines.right":         False,
        "axes.grid":                 False,
        "xtick.direction":           "out",
        "ytick.direction":           "out",
        "xtick.major.width":         0.8,
        "ytick.major.width":         0.8,
        "xtick.major.size":          3.5,
        "ytick.major.size":          3.5,
        "xtick.labelsize":           8,
        "ytick.labelsize":           8,
        "axes.labelsize":            9,
        "axes.titlesize":            9.5,
        "axes.titleweight":          "bold",
        "axes.labelpad":             4,
        "legend.fontsize":           7.5,
        "legend.frameon":            True,
        "legend.framealpha":         0.9,
        "legend.edgecolor":          "#CCCCCC",
        "legend.handlelength":       1.5,
        "lines.linewidth":           1.5,
        "patch.linewidth":           0.4,
        "figure.constrained_layout.use": True,
    })

setup()

# ── 配色（Paul Tol Bright，色盲友好）──────────────────
R  = "#BB5566"
B  = "#004488"
OR = "#DDAA33"
GR = "#117733"
PU = "#882255"
LB = "#4477AA"
GY = "#AAAAAA"
TEAL = "#44AA99"

def plbl(ax, letter, x=-0.13, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color="#111111")

def save(fig, name):
    p = OUTPUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  {name} saved")

# ═══════════════════════════════════════════════════════════════
# FIG 1  ——  Venn + 化学类别条图
# ═══════════════════════════════════════════════════════════════
log.info("Fig 1 ...")
from matplotlib_venn import venn3
from utils import load_s1, load_s2, load_model_metadata

s1      = load_s1(sig_only=True)
s2      = load_s2()
sp_df,_ = load_model_metadata()
mapped  = pd.read_csv(OUTPUT / "01_mapped_metabolites.csv")

s1_n = set(s1["name"].str.lower().str.split(";").str[0].str.strip())
s2_c = set()
for _, row in s2.iterrows():
    for c in row["compound_list"]:
        nm = (c.split(" ",1)[1].strip() if " " in c else c.strip()).lower()
        s2_c.add(nm)
mn = set(sp_df["name_short"].dropna())
sub = (len(s1_n-s2_c-mn), len(s2_c-s1_n-mn),
       len((s1_n&s2_c)-mn), len(mn-s1_n-s2_c),
       len((s1_n&mn)-s2_c), len((s2_c&mn)-s1_n),
       len(s1_n&s2_c&mn))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

v = venn3(subsets=sub,
          set_labels=("Table S1\n(161 sig. metabolites)",
                      "Table S2\n(52 KEGG pathways)",
                      "iBsu1103\n(1,381 species)"),
          set_colors=(R, B, GR), alpha=0.36, ax=ax1)
for p in v.patches:
    if p: p.set_edgecolor("white"); p.set_linewidth(1.8)
for t in v.set_labels:
    if t: t.set_fontsize(9.5); t.set_fontweight("bold")
for t in v.subset_labels:
    if t: t.set_fontsize(9.5); t.set_fontweight("bold")
ax1.annotate(f"Anchor set\nS1 $\\cap$ iBsu1103 = {sub[4]+sub[6]} metabolites",
             xy=(0.97,0.18), xycoords="axes fraction", fontsize=8.5,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.9))
ax1.set_title("Three-Source Data Integration")
plbl(ax1, "A")

# 化学类别分布
cls_abbr = {
    "Amino acid and derivatives;amino acids":"Amino acids",
    "Amino acid and derivatives":"Amino acids",
    "Nucleotide and its derivates":"Nucleotides",
    "Organooxygen compounds":"Organooxygen",
    "Keto acids and derivatives":"Keto acids",
    "Benzene and substituted derivatives":"Aromatics",
    "Benzene and substituted deri":"Aromatics",
    "Flavonoids":"Flavonoids",
    "Carbohydrates":"Carbohydrates",
    "Cholines":"Cholines",
    "Organic acids":"Organic acids",
    "Alkaloids":"Alkaloids",
    "phytohormone":"Phytohormones",
    "Imidazopyrimidines":"Imidazopyr.",
    "Carboxylic acids and derivat":"Carboxylic acids",
    "Fatty Acyls":"Fatty Acyls",
    "Benzoic acid derivatives":"Benzoates",
    "Phenols;Phenylpropanoic acid":"Phenolics",
    "Organic acids and derivative":"Org. acids",
}
mapped["cls2"] = mapped["cls"].apply(
    lambda x: next((v for k,v in cls_abbr.items() if k in str(x)), "Others"))
ct = (mapped.groupby(["cls2","direction"])
      .size().unstack(fill_value=0)
      .sort_values("natto_up" if "natto_up" in mapped["direction"].values else mapped["direction"].unique()[0],
                   ascending=True))
ys = range(len(ct))
w  = 0.38
for i, (cls, row) in enumerate(ct.iterrows()):
    nu = row.get("natto_up",0)
    su = row.get("soy_up",0)
    ax2.barh(i+w/2, nu, w, color=R, alpha=0.85, edgecolor="white")
    ax2.barh(i-w/2, su, w, color=B, alpha=0.85, edgecolor="white")
ax2.set_yticks(list(ys))
ax2.set_yticklabels(ct.index.tolist(), fontsize=8.5)
ax2.set_xlabel("Number of anchor metabolites")
ax2.set_title("Chemical Class Distribution of 76 Anchor Metabolites")
ax2.legend(handles=[mpatches.Patch(color=R,label="Natto-enriched (61)"),
                    mpatches.Patch(color=B,label="Soy-enriched (15)")],
           loc="lower right", fontsize=8)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plbl(ax2, "B")
save(fig, "fig1_venn_class.png")

# ═══════════════════════════════════════════════════════════════
# FIG 2  ——  火山图（高质量）
# ═══════════════════════════════════════════════════════════════
log.info("Fig 2 ...")
s1a   = load_s1(sig_only=False)
m_low = set(mapped["name"].str.lower().str.split(";").str[0].str.strip())
is_map= s1a["name"].str.lower().str.split(";").str[0].str.strip().isin(m_low)
yv    = -np.log10(s1a["pval"].clip(lower=1e-10))

fig, ax = plt.subplots(figsize=(6.0, 4.8))
ns = s1a["pval"] >= 0.05
ax.scatter(s1a.loc[ns,"logfc"], yv[ns], c=GY, s=9, alpha=0.35,
           linewidths=0, label=f"NS (n={ns.sum()})", rasterized=True, zorder=2)
ms = (s1a["pval"]<0.05)&(s1a["fc"]<1)&~is_map
ax.scatter(s1a.loc[ms,"logfc"], yv[ms], c=B, s=16, alpha=0.78,
           linewidths=0, label=f"Soy-up (n={ms.sum()})", zorder=3)
mn2 = (s1a["pval"]<0.05)&(s1a["fc"]>1)&~is_map
ax.scatter(s1a.loc[mn2,"logfc"], yv[mn2], c=R, s=16, alpha=0.78,
           linewidths=0, label=f"Natto-up (n={mn2.sum()})", zorder=3)
ax.scatter(s1a.loc[is_map,"logfc"], yv[is_map], c=OR, s=55, alpha=0.96,
           edgecolors="white", linewidths=0.6, zorder=6,
           label=f"Anchor nodes (n={is_map.sum()})")

top = pd.concat([mapped.nlargest(7,"fc"), mapped.nsmallest(5,"fc")]).drop_duplicates("name")
ymax_data = -np.log10(max(s1a["pval"].min(), 1e-10))
for i, (_, r) in enumerate(top.iterrows()):
    nm = r["name"].split(";")[0].strip()
    if len(nm)>20: nm = nm[:19]+"."
    xp = r["logfc"]; yp = -np.log10(max(r["pval"],1e-10))
    dx = (1.2 + 0.4*(i % 3)) if xp>0 else -(1.2 + 0.4*(i % 3))
    dy = 0.4 + 0.25*(i % 4)
    if yp > ymax_data * 0.65:
        dy = -(dy + 0.5)
    ax.annotate(nm, xy=(xp,yp), xytext=(xp+dx, yp+dy),
                fontsize=6.2, color="#222222",
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.55),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))

ax.axhline(-np.log10(0.05), color="#999999", lw=0.9, ls="--")
ax.axvline(0, color="#CCCCCC", lw=0.7, ls=":")
xl = ax.get_xlim()
ax.text(xl[1]*0.96, -np.log10(0.05)+0.12, "p = 0.05", fontsize=7.5, color="#888888", ha="right")
ax.set_xlabel(r"$\log_2$ Fold Change  [Natto / Soybean]", fontsize=9.5)
ax.set_ylabel(r"$-\log_{10}$  p-value", fontsize=9.5)
ax.set_title("Differential Metabolite Volcano Plot: Natto vs. Soybean")
ax.legend(loc="upper left", markerscale=1.4, framealpha=0.9, edgecolor="#CCCCCC")
save(fig, "fig2_volcano.png")

# ═══════════════════════════════════════════════════════════════
# FIG 3  ——  重设计：瀑布图（A）+ Baseline vs cFBA 散点图（B）
# ═══════════════════════════════════════════════════════════════
log.info("Fig 3 — Waterfall + Scatter ...")
cf = pd.read_csv(OUTPUT / "02_cfba_fluxes.csv")
internal = cf[~cf["reaction"].str.startswith("EX_") &
              ~cf["reaction"].str.contains("bio")].copy()
clean = internal[(internal["abs_delta"]>5) &
                 (internal["abs_delta"]<7000) &
                 (internal["baseline_flux"].abs()<6900) &
                 (internal["cfba_flux"].abs()<6900)].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.5))

# ── Panel A: 瀑布图 top24 ──────────────────────────────
top24 = clean.nlargest(24, "abs_delta").copy()
top24 = top24.sort_values("delta_flux", ascending=True)
top24["label"] = top24["reaction"].apply(lambda x: x[:16])
top24["col"] = top24["delta_flux"].apply(lambda x: R if x > 0 else B)

ys = range(len(top24))
for i, (_, row) in enumerate(top24.iterrows()):
    ax1.barh(i, row["delta_flux"], color=row["col"],
             height=0.68, alpha=0.88, edgecolor="white", linewidth=0.4)
ax1.set_yticks(list(ys))
ax1.set_yticklabels(top24["label"].tolist(), fontsize=8)
ax1.axvline(0, color="#333333", lw=1.0)
ax1.set_xlabel(r"$\Delta$v  =  v$_{\mathrm{cFBA}}$ $-$ v$_{\mathrm{baseline}}$  (rel. units)")
ax1.set_title("Top 24 Flux Changes (Internal Reactions)")
ax1.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
ax1.set_axisbelow(True)
p1 = mpatches.Patch(color=R, label="Natto-increased")
p2 = mpatches.Patch(color=B, label="Natto-decreased")
ax1.legend(handles=[p1, p2], fontsize=8, loc="lower right")
plbl(ax1, "A")

# ── Panel B: 散点图 Baseline vs cFBA（全202条clean反应）──
ax2.scatter(clean["baseline_flux"], clean["cfba_flux"],
            c=GY, s=18, alpha=0.45, linewidths=0,
            label=f"All reactions (n={len(clean)})", zorder=2)

# 高亮 top24
for _, row in top24.iterrows():
    col = R if row["delta_flux"]>0 else B
    ax2.scatter(row["baseline_flux"], row["cfba_flux"],
                c=col, s=45, alpha=0.92, linewidths=0.4,
                edgecolors="white", zorder=5)

# 1:1 对角线
vmin_ = min(clean["baseline_flux"].min(), clean["cfba_flux"].min())
vmax_ = max(clean["baseline_flux"].max(), clean["cfba_flux"].max())
diag = np.linspace(vmin_, vmax_, 100)
ax2.plot(diag, diag, color="#444444", lw=1.2, ls="--",
         alpha=0.6, label="y = x (no change)")

ax2.set_xlabel("Baseline FBA flux (rel. units)")
ax2.set_ylabel("Natto cFBA flux (rel. units)")
ax2.set_title("Flux Redistribution: Baseline vs. Natto cFBA")
ax2.legend(handles=[
    plt.Line2D([0],[0], color="#444444", ls="--", lw=1.2, label="y = x"),
    mpatches.Patch(color=GY, label=f"All clean reactions (n={len(clean)})"),
    mpatches.Patch(color=R, label="Top-24 natto-increased"),
    mpatches.Patch(color=B, label="Top-24 natto-decreased"),
], fontsize=7.5, loc="upper left")
plbl(ax2, "B")

save(fig, "fig3_flux_waterfall.png")

# ═══════════════════════════════════════════════════════════════
# FIG 4  ——  风险RI + FSEOF
# ═══════════════════════════════════════════════════════════════
log.info("Fig 4 ...")
rf = pd.read_csv(OUTPUT / "03_risk_fluxes.csv")
fs = pd.read_csv(OUTPUT / "04_fseof_L_Histidine.csv")
fs["slope"] = pd.to_numeric(fs["slope"], errors="coerce")
fs = fs.dropna(subset=["slope"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

mets_s = rf["metabolite"].tolist()
n = len(mets_s); x = np.arange(n); w = 0.32
ax1.bar(x-w/2, rf["ri_baseline"].abs(), w, color=GY, alpha=0.85,
        label="Baseline FBA", edgecolor="white", linewidth=0.4)
ax1.bar(x+w/2, rf["ri_cfba"].abs(), w, color=R, alpha=0.88,
        label="Natto cFBA", edgecolor="white", linewidth=0.4)
ax1.axhline(1.0, color="#333333", lw=1.3, ls="--", label="RI = 1.0 (safety threshold)")
for i, v in enumerate(rf["ri_baseline"].abs()):
    ax1.text(i-w/2, v+0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#555555")
for i, v in enumerate(rf["ri_cfba"].abs()):
    ax1.text(i+w/2, v+0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
             color=R if v>1 else "#555555", fontweight="bold" if v>1 else "normal")
ax1.set_xticks(x)
ax1.set_xticklabels(["Phenethylamine\n(PEA)", "Phenylacetyl-\nGln (PAGln)", "L-Arg\n(ADMA proxy)"], fontsize=8.5)
ax1.set_ylabel("Risk Index (RI)")
ax1.set_title("Food Safety Risk Index per Metabolite")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_ylim(0, rf["ri_baseline"].abs().max()*1.32)
ax1.annotate("Model Gap:\nno Phe\ndecarboxylase",
             xy=(0-w/2, 0.1), xytext=(0.42, 4.2),
             fontsize=7, color=R, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=R, lw=0.9),
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E7", ec=OR, lw=0.9))
plbl(ax1, "A")

top_oe = fs.nlargest(7, "slope")
top_ko = fs.nsmallest(7, "slope")
show = pd.concat([top_ko, top_oe]).drop_duplicates("reaction").sort_values("slope")

def sname(n):
    m = {"formate transport in via proton symport":  "Formate transporter",
         "H2Ot5":                                    "H$_2$O transport",
         "ATP:D-fructose-6-phosphate 1-phosphotransferase": "PFK (Fru6P→Fru1,6P)",
         "ATP:D-fructose-1-phosphate 6-phosphotransferase": "Fru1P kinase",
         "D-fructose transport via PEP:Pyr PTS":     "Fructose PTS (ptsG)",
         "ATP:pyruvate O2-phosphotransferase":        "Pyruvate kinase",
         "Acetaldehyde:NAD+ oxidoreductase":          "Acetaldehyde DH",
         "N-Acetyl-D-glucosamine-6-phosphate amidohydrolase": "GlcNAc6P deacetylase",
         "(S)-Lactate:NAD+ oxidoreductase":           "L-Lactate DH",
         "(S)-Lactate acetaldehyde-lyase":            "Lactate lyase",
         "N-Acetyl-D-glucosamine transport via PEP:Pyr PTS": "GlcNAc PTS",
         "ATP:alpha-D-glucose-1-phosphate adenyltransferase": "Glc1P adenylase",
         "glycogen synthase (ADPGlc)":                "Glycogen synthase",
         "cpd00155 phosphorylase":                    "Starch phosphorylase",
         "Salicin 6-phosphate glucohydrolase":        "Salicin-6P glucohydrolase",
         "mannitol transport via PEP:Pyr PTS":        "Mannitol PTS",
         "D-Mannitol-1-phosphate:NAD+ 5-oxidoreductase": "Mannitol-1P DH",
         "D-glucosamine transport via PEP:Pyr PTS":   "GlcN PTS",
         "ATP:butyrate 1-phosphotransferase":         "Butyrate kinase",
         "Butanoyl-CoA:acetoacetate CoA-transferase": "Butyryl-CoA transferase",
         "Butanoyl-CoA:orthophosphate butanoyltransferase": "Phosphotransbutyrylase",
         "Acetoacetate:CoA ligase (AMP-forming)":     "Acetoacetate-CoA ligase",
         "L-arabinose transport via proton symport":  "Arabinose transporter",
         "L-ribulose-5-phosphate 4-epimerase":        "L-Ribulose-5P epimerase",
         "L-Arabinose ketol-isomerase":               "Arabinose isomerase",
         "ATP:L-ribulose 5-phosphotransferase":       "Ribulokinase",
         "D-glucosamine 6-phosphate reversible uniport": "GlcN6P uniport",
         "L-Glutamine:D-fructose-6-phosphate aminotransferase (hexose": "GlcN6P synthase",
         "Na+ Glutamine symporter":                   "Gln Na$^+$ symporter",
         "glycine transport in/out via proton symport": "Gly transporter",
         "D-glutamate transport in via proton symport": "D-Glu transporter",
         "L-Glutamate racemase":                      "Glu racemase",
         "5,10-Methylenetetrahydrofolate:glycine hydroxymethyltransferase": "SHMT",
         "glycine synthase":                          "Glycine synthase",
         }
    s = n.strip()
    if s in m:
        return m[s]
    # fallback: keep ≤22 chars, no truncation marker
    return s[:22] if len(s) > 22 else s

show["short"] = show["rxn_name"].apply(sname)
show["col"] = show["slope"].apply(lambda x: GR if x>0 else PU)

for i, (_, r) in enumerate(show.iterrows()):
    ax2.hlines(i, 0, r["slope"], color=r["col"], lw=2.0, alpha=0.9)
    ax2.scatter(r["slope"], i, color=r["col"], s=60, zorder=5,
                edgecolors="white", linewidths=0.7)
ax2.set_yticks(range(len(show)))
ax2.set_yticklabels(show["short"].tolist(), fontsize=8)
ax2.axvline(0, color="#333333", lw=0.9)
ax2.set_xlabel("Flux-Production Regression Slope")
ax2.set_title("L-Histidine FSEOF Engineering Targets")
ax2.grid(axis="x", color="#EEEEEE", lw=0.6)
ax2.set_axisbelow(True)
ax2.legend(handles=[mpatches.Patch(color=GR, label="Overexpression (positive slope)"),
                    mpatches.Patch(color=PU, label="Knockout (negative slope)")], fontsize=8)
plbl(ax2, "B")
save(fig, "fig4_risk_fseof.png")

# ═══════════════════════════════════════════════════════════════
# FIG 5  ——  重设计：28基因通路分组 + 关联反应数排名
# ═══════════════════════════════════════════════════════════════
log.info("Fig 5 — Essential genes redesigned ...")
ko_all  = pd.read_csv(OUTPUT / "05_essential_natto.csv")
ko_std  = pd.read_csv(OUTPUT / "05_essential_standard.csv")
ko_spec = pd.read_csv(OUTPUT / "05_natto_specific_essential.csv")

# 手动标注每个基因的通路分组
pathway_map = {
    "SPONTANEOUS":   "Spontaneous/Misc",
    "peg_DOT_2193":  "Amino sugar metab.",
    "peg_DOT_2266":  "Phe/Tyr metabolism",
    "peg_DOT_2267":  "Phe/Tyr metabolism",
    "peg_DOT_2268":  "Phe/Tyr metabolism",
    "peg_DOT_2269":  "Phe/Tyr metabolism",
    "peg_DOT_2270":  "Phe/Tyr metabolism",
    "peg_DOT_2271":  "Nucleotide metab.",
    "peg_DOT_2272":  "Nucleotide metab.",
    "peg_DOT_2343":  "Central carbon",
    "peg_DOT_2828":  "Acetolactate pathway",
    "peg_DOT_2829":  "Acetolactate pathway",
    "peg_DOT_2830":  "Acetolactate pathway",
    "peg_DOT_2831":  "Acetolactate pathway",
    "peg_DOT_2832":  "Acetolactate pathway",
    "peg_DOT_2833":  "Acetolactate pathway",
    "peg_DOT_2834":  "Acetolactate pathway",
    "peg_DOT_2965":  "Vitamin/cofactor",
    "peg_DOT_3492":  "His biosynthesis",
    "peg_DOT_3493":  "His biosynthesis",
    "peg_DOT_3494":  "His biosynthesis",
    "peg_DOT_3495":  "His biosynthesis",
    "peg_DOT_3496":  "His biosynthesis",
    "peg_DOT_3497":  "His biosynthesis",
    "peg_DOT_3498":  "His biosynthesis",
    "peg_DOT_3499":  "His biosynthesis",
    "peg_DOT_653":   "Nucleotide metab.",
    "peg_DOT_75":    "Nucleotide metab.",
}
pathway_colors = {
    "Acetolactate pathway": "#BB5566",
    "His biosynthesis":     "#DDAA33",
    "Phe/Tyr metabolism":   "#004488",
    "Nucleotide metab.":    "#117733",
    "Amino sugar metab.":   "#44AA99",
    "Central carbon":       "#882255",
    "Vitamin/cofactor":     "#4477AA",
    "Spontaneous/Misc":     "#AAAAAA",
}

ko_spec = ko_spec[ko_spec["gene_id"] != "SPONTANEOUS"].copy()
ko_spec["pathway"] = ko_spec["gene_id"].map(pathway_map).fillna("Others")
ko_spec["col"] = ko_spec["pathway"].map(pathway_colors).fillna("#AAAAAA")
ko_spec_sorted = ko_spec.sort_values(["pathway","n_reactions"], ascending=[True,False])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2))

# ── Panel A: 通路分组水平棒图（按基因关联反应数）──────
ys = range(len(ko_spec_sorted))
for i, (_, row) in enumerate(ko_spec_sorted.iterrows()):
    ax1.barh(i, row["n_reactions"], color=row["col"],
             height=0.72, alpha=0.88, edgecolor="white", linewidth=0.4)
ax1.set_yticks(list(ys))
ax1.set_yticklabels(ko_spec_sorted["gene_id"].str.replace("peg_DOT_","peg.").tolist(),
                    fontsize=8)
ax1.set_xlabel("Number of associated reactions")
ax1.set_title("Natto-Specific Essential Genes\n(Grouped by Metabolic Pathway)")
ax1.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
ax1.set_axisbelow(True)
# 图例
legend_handles = [mpatches.Patch(color=c, label=p)
                  for p, c in pathway_colors.items()
                  if p in ko_spec_sorted["pathway"].values]
ax1.legend(handles=legend_handles, fontsize=7, loc="lower right",
           framealpha=0.92, ncol=1)
# 右侧标注通路名
prev_path = None
for i, (_, row) in enumerate(ko_spec_sorted.iterrows()):
    if row["pathway"] != prev_path:
        ax1.text(ax1.get_xlim()[1]*0.02 if ax1.get_xlim()[1]>0 else 0.1,
                 i, row["pathway"],
                 fontsize=6.5, va="center", alpha=0.7, color=row["col"])
        prev_path = row["pathway"]
plbl(ax1, "A")

# ── Panel B: 标准 vs 纳豆必需基因数量对比 + 生长率密度 ──
ax2_twin = ax2
# 先画分组条
categories = ["Standard\ncondition\nessential",
              "Natto\ncondition\nessential",
              "Natto-\nspecific"]
vals = [len(ko_std), len(ko_all), len(ko_spec)]
cols_bar = [B, R, OR]
bars = ax2_twin.bar(categories, vals, color=cols_bar, width=0.5,
                    edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, vals):
    pct = val/1103*100
    ax2_twin.text(bar.get_x()+bar.get_width()/2,
                  bar.get_height()+3, f"{val}\n({pct:.1f}%)",
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2_twin.set_ylabel("Number of genes")
ax2_twin.set_title("Essential Gene Count: Standard vs. Natto Conditions\n(1,103 total genes in iBsu1103)")
ax2_twin.set_ylim(0, max(vals)*1.22)

# 在条图上叠加趋势箭头
ax2_twin.annotate("", xy=(1, vals[1]+5), xytext=(0, vals[0]+5),
                  arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
ax2_twin.annotate(f"+{vals[1]-vals[0]} genes\n(+{(vals[1]-vals[0])/vals[0]*100:.0f}%)",
                  xy=(0.5, (vals[0]+vals[1])/2+15), fontsize=8.5,
                  ha="center", color="#555555", style="italic")
plbl(ax2_twin, "B")

save(fig, "fig5_essential_genes.png")

# ═══════════════════════════════════════════════════════════════
# FIG 6  ——  FVA 弹性饼图 + 度分布 log-log
# ═══════════════════════════════════════════════════════════════
log.info("Fig 6 ...")
fva = pd.read_csv(OUTPUT / "02_fva_ranges.csv", index_col=0)
nd  = pd.read_csv(OUTPUT / "05_network_nodes.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

n_b = (fva["range"] <  0.01).sum()
n_m = ((fva["range"]>=0.01)&(fva["range"]<=100)).sum()
n_f = (fva["range"] > 100).sum()
tot = n_b + n_m + n_f

patches_pie, texts, autotexts = ax1.pie(
    [n_b, n_m, n_f],
    labels=[f"Bottleneck\nrange<0.01\n(n={n_b})",
            f"Moderate\n0.01-100\n(n={n_m})",
            f"Flexible\nrange>100\n(n={n_f})"],
    colors=[R, OR, GR], explode=(0.06,0,0),
    autopct="%1.1f%%", pctdistance=0.72,
    startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=1.8),
    textprops={"fontsize":8.5})
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")
ax1.set_title("FVA Flux Flexibility Classification\n(Natto cFBA, 90% optimality, 1,681 reactions)")
plbl(ax1, "A", x=-0.06)

degrees = nd["degree"].values
dc = pd.Series(degrees).value_counts().sort_index()
xv = dc.index.values.astype(float); yc = dc.values.astype(float)
mask = (xv>0) & (yc>0)

ax2.scatter(xv[mask], yc[mask], c=LB, s=30, alpha=0.82,
            edgecolors="white", linewidths=0.4, label="Observed", zorder=5)

if mask.sum()>4:
    xl_ = np.log10(xv[mask]); yl_ = np.log10(yc[mask])
    slope_, icept_, r_, *_ = stats.linregress(xl_, yl_)
    xfit = np.logspace(xl_.min(), xl_.max(), 100)
    ax2.plot(xfit, 10**icept_ * xfit**slope_, color=R, lw=2.5, zorder=4,
             label=f"Power-law: $\\gamma$={abs(slope_):.2f}, $R^2$={r_**2:.3f}")

anch = nd[nd["is_anchor"]==True]
for _, row in anch.iterrows():
    d = row["degree"]; fc_v = dc.get(d,1)
    ax2.scatter(d, fc_v, c=OR, s=65, zorder=7, edgecolors="white", linewidths=0.6)
ax2.scatter([],[],c=OR,s=65,label=f"Anchor metabolites (n={len(anch)})")

top3 = nd.nlargest(3,"degree")
for idx_i, (_, row) in enumerate(top3.iterrows()):
    d = row["degree"]; fc_v = dc.get(d,1)
    nm = row["name"].split("_")[0][:8]
    ax2.annotate(nm, xy=(d,fc_v), xytext=(d * 0.35, fc_v * 3.5),
                 fontsize=7.5, annotation_clip=True,
                 arrowprops=dict(arrowstyle="-",color="#AAAAAA",lw=0.6))

ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("Degree k  (log scale)")
ax2.set_ylabel("Frequency P(k)  (log scale)")
ax2.set_title("Metabolic Network Degree Distribution\n(Preliminary degree distribution, iBsu1103)")
ax2.legend(fontsize=8, loc="upper right")
plbl(ax2, "B")
save(fig, "fig6_fva_degree.png")

# ═══════════════════════════════════════════════════════════════
# FIG 7  ——  新增：产量缺口三产品对比（3panel）
# ═══════════════════════════════════════════════════════════════
log.info("Fig 7 — Yield gap ...")
yg = pd.read_csv(OUTPUT / "04_yield_gap.csv")

products     = yg["product"].tolist()
fc_vals      = yg["fc_observed"].tolist()
theo_max     = yg["theo_max_flux"].tolist()
prac_max     = yg["prac_max_flux"].tolist()
curr_flux    = yg["current_flux"].tolist()
gap_pct      = yg["yield_gap_pct"].fillna(float('nan')).tolist()
indust_val   = yg["industrial_value"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
pal3 = [R, OR, B]
cats = ["Theoretical\nMaximum", "Practical Max\n(80% growth)", "Current\ncFBA Flux"]
keys = [theo_max, prac_max, curr_flux]

for col_i, (ax, prod, fc, t, p, c, gp, iv) in enumerate(
        zip(axes, products, fc_vals, theo_max, prac_max, curr_flux, gap_pct, indust_val)):
    vals_bar = [abs(t), abs(p), abs(c)]
    if all(v == 0 for v in vals_bar):
        ax.bar(cats, vals_bar, color=["#CCCCCC", "#CCCCCC", "#CCCCCC"],
               width=0.5, edgecolor="white")
        ax.text(0.5, 0.5, "MODEL GAP\n(FVA max = 0)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=R, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F0", ec=R, lw=1.5))
    else:
        bar_cols = ["#CCCCCC", pal3[col_i], R if c < 0 else "#666666"]
        bars = ax.bar(cats, vals_bar, color=bar_cols, width=0.5,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals_bar):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+max(vals_bar)*0.02,
                        f"{val:.0f}" if val < 1000 else f"{val/1000:.1f}k",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        if not np.isnan(gp) and abs(p) > 0:
            ax.annotate(f"Yield gap\n{gp:.0f}%",
                        xy=(1, abs(p)), xytext=(1.55, abs(p)*0.65),
                        fontsize=8, color=R, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=R, lw=1.0),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E7", ec=OR, lw=0.9))
        if c < 0:
            ax.text(2, -max(vals_bar)*0.04, "(uptake direction)",
                    ha="center", va="top", fontsize=7, color="#555555", style="italic")

    ax.set_xticklabels(cats, fontsize=8, rotation=15, ha="right")
    prod_short = prod if len(prod)<=14 else prod[:13]+"."
    ax.set_title(f"{prod_short}\n(FC = {fc:.0f}x; {iv.split(',')[0]})",
                 fontsize=9)
    ax.set_ylabel("Flux (rel. units)" if col_i == 0 else "")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"{v:.0f}" if v<1000 else f"{v/1000:.1f}k"))
    plbl(ax, "ABC"[col_i])

save(fig, "fig7_yield_gap.png")

# ═══════════════════════════════════════════════════════════════
# FIG 8  ——  锚定代谢物网络枢纽分析（重绘：气泡图+分类条图）
#   Panel A: 锚定代谢物 degree × log2FC 气泡图（气泡大小=betweenness）
#   Panel B: 按化学类别分组的锚定代谢物数量条图（纳豆上调/大豆上调）
# ═══════════════════════════════════════════════════════════════
log.info("Fig 8 — Network hubs ...")
nd = pd.read_csv(OUTPUT / "05_network_nodes.csv")
anch = nd[nd["is_anchor"]==True].copy()

# 读取锚定代谢物的FC和类别信息
mapped = pd.read_csv(OUTPUT / "01_mapped_metabolites.csv")
mapped["logfc"] = np.log2(mapped["fc"].clip(lower=1e-3))

# 合并网络信息（先删除anch中的direction列避免冲突）
anch_m = anch.drop(columns=["direction"], errors="ignore").merge(
    mapped[["model_id","fc","logfc","cls","direction"]],
    left_on="node_id", right_on="model_id", how="left")
anch_m = anch_m.dropna(subset=["fc"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                                gridspec_kw={"width_ratios":[1.2, 1]})

# ── Panel A: degree × log2FC 气泡图 ──────────────────────────
col_map = {"natto_up": R, "soy_up": B}
for _, row in anch_m.iterrows():
    col = col_map.get(row.get("direction",""), OR)
    bsz = max(30, min(600, float(row["betweenness"]) * 8 + 30))
    ax1.scatter(row["degree"], row["logfc"],
                s=bsz, c=col, alpha=0.75,
                edgecolors="white", linewidths=0.5, zorder=4)

# 标注高betweenness或极端FC的节点
top_label = anch_m.nlargest(8, "betweenness")
for _, row in top_label.iterrows():
    nm = str(row["name"]).split("_")[0][:14]
    ax1.annotate(nm, xy=(row["degree"], row["logfc"]),
                 xytext=(row["degree"]+1.5, row["logfc"]+0.3),
                 fontsize=6, color="#222222", annotation_clip=True,
                 arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.4))

ax1.axhline(0, color=GY, lw=0.8, ls="--", zorder=1)
ax1.set_xlabel("Degree (number of connected reactions)")
ax1.set_ylabel("log$_2$(FC)  natto / soy")
ax1.set_title("Anchor Metabolite Network Position\n"
              "Bubble size ∝ betweenness centrality")
ax1.legend(handles=[
    mpatches.Patch(color=R, label="Natto-up"),
    mpatches.Patch(color=B, label="Soy-up"),
], fontsize=8, loc="upper right")
# 气泡大小图例
for sz, lbl in [(30,"low BC"),(200,"mid BC"),(500,"high BC")]:
    ax1.scatter([],[], s=sz, c=GY, alpha=0.6, label=lbl, edgecolors="white")
ax1.legend(fontsize=7.5, loc="upper left",
           handles=[mpatches.Patch(color=R,label="Natto-up"),
                    mpatches.Patch(color=B,label="Soy-up")] +
                   [plt.scatter([],[],s=s,c=GY,alpha=0.6,edgecolors="white")
                    for s in [40,200,500]],
           labels=["Natto-up","Soy-up","BC low","BC mid","BC high"])
plbl(ax1, "A")

# ── Panel B: 化学类别分组条图 ─────────────────────────────────
cls_counts = anch_m.groupby(["cls","direction"]).size().unstack(fill_value=0)
# 保留前8类（按总数降序）
cls_counts["total"] = cls_counts.sum(axis=1)
cls_counts = cls_counts.nlargest(8,"total").drop(columns="total")
if len(cls_counts.columns) > 0:
    cls_counts = cls_counts.sort_values(cls_counts.columns[0], ascending=True)

# 缩短类别名
def shorten_cls(s):
    m = {"Amino acid and derivatives":"Amino acids",
         "Carboxylic acids and derivatives":"Carboxylic acids",
         "Nucleotide and its derivates":"Nucleotides",
         "Benzene and substituted deriv.":"Benzene deriv.",
         "Organooxygen compounds":"Organooxygen",
         "Keto acids and derivatives":"Keto acids",
         "Carbohydrates":"Carbohydrates",
         "Fatty Acyls":"Fatty acyls",
         "Organic acids and derivatives":"Organic acids",
         "Imidazopyrimidines":"Imidazopyrimidines",
         "Phenols/Phenylpropanoic acids":"Phenols/Prop.",
         "Phytohormone":"Phytohormone",
         "Cholines":"Cholines",
         "Alkaloids; amino acids":"Alkaloids/AA",
         }
    return m.get(s, s[:22] if len(s)>22 else s)

cls_counts.index = [shorten_cls(i) for i in cls_counts.index]
y = np.arange(len(cls_counts))
w = 0.38
cols_dir = {"natto_up": R, "soy_up": B}
for i, col in enumerate(cls_counts.columns):
    vals = cls_counts[col].values
    offset = (i - (len(cls_counts.columns)-1)/2) * w
    ax2.barh(y + offset, vals, height=w,
             color=cols_dir.get(col, GY), alpha=0.85,
             edgecolor="white", linewidth=0.4,
             label="Natto-up" if col=="natto_up" else "Soy-up")
ax2.set_yticks(y)
ax2.set_yticklabels(cls_counts.index.tolist(), fontsize=8.5)
ax2.set_xlabel("Number of anchor metabolites")
ax2.set_title("Anchor Metabolites by Chemical Class\n(Top 8 classes)")
ax2.legend(fontsize=8)
ax2.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
ax2.set_axisbelow(True)
plbl(ax2, "B")

plt.tight_layout(pad=0.6, w_pad=1.2)
save(fig, "fig8_network_hubs.png")


# ═══════════════════════════════════════════════════════════════
# 复制到 paper/figures/
# ═══════════════════════════════════════════════════════════════
PAPER_FIG = OUTPUT.parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
for old in PAPER_FIG.glob("fig*.png"):
    old.unlink()
cnt = 0
for fn in sorted(OUTPUT.glob("fig[1-8]*.png")):
    shutil.copy(fn, PAPER_FIG / fn.name); cnt += 1
log.info(f"Done — {cnt} figures → paper/figures/")
print(f"\n完成：{cnt} 张图（Fig1-8）全部已复制至 paper/figures/")
