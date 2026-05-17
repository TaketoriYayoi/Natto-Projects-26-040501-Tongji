"""
06_regenerate_figures.py  —  重新生成所有图表（最终修正版）
配色：Paul Tol bright  ( R=#BB5566  B=#004488  OR=#DDAA33  GR=#117733  PU=#5B2C6F )
图中标签全部用英文（matplotlib 无 CJK 字体），中文描述放 LaTeX 图注
尺寸：单栏 5.5×4.0 in  /  双栏 9.5×4.0 in，300 dpi
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
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import utils
from utils import OUTPUT, DATA

log = utils.get_logger("06_figures")

# ═══════════════════════════════════════════════════════
#  全局样式
# ═══════════════════════════════════════════════════════
def setup():
    import matplotlib as mpl
    mpl.rcdefaults()
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.family":           "Times New Roman",
        "mathtext.fontset":      "stix",
        "axes.unicode_minus":    False,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.04,
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "axes.edgecolor":        "#444444",
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             False,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "xtick.major.width":     0.75,
        "ytick.major.width":     0.75,
        "xtick.major.size":      3.0,
        "ytick.major.size":      3.0,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "axes.labelsize":        9,
        "axes.titlesize":        9.5,
        "axes.titleweight":      "bold",
        "axes.labelpad":         4,
        "legend.fontsize":       7.5,
        "legend.frameon":        True,
        "legend.framealpha":     0.92,
        "legend.edgecolor":      "#CCCCCC",
        "lines.linewidth":       1.5,
        "patch.linewidth":       0.5,
        "figure.constrained_layout.use": True,
    })

setup()

R  = "#BB5566"   # Natto-up  — rose red
B  = "#004488"   # Soy-up    — deep blue
OR = "#DDAA33"   # Anchor    — amber
GR = "#117733"   # Positive  — forest green
PU = "#5B2C6F"   # Negative  — deep purple
GY = "#AAAAAA"   # Neutral   — grey
LB = "#4477AA"   # Soft blue

def lbl(ax, letter, x=-0.14, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

def save(fig, name):
    p = OUTPUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  {name} saved")

# ═══════════════════════════════════════════════════════
# FIG 01 — Venn diagram
# ═══════════════════════════════════════════════════════
log.info("Fig 01 ...")
from matplotlib_venn import venn3
from utils import load_s1, load_s2, load_model_metadata

s1      = load_s1(sig_only=True)
s2      = load_s2()
sp_df,_ = load_model_metadata()

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

fig, ax = plt.subplots(figsize=(5.5, 4.0))
v = venn3(subsets=sub,
          set_labels=("Table S1\n(161 sig. metabolites)",
                      "Table S2\n(52 KEGG pathways)",
                      "iBsu1103\n(1,381 species)"),
          set_colors=(R, B, GR), alpha=0.38, ax=ax)
for p in v.patches:
    if p: p.set_edgecolor("white"); p.set_linewidth(1.5)
for t in v.set_labels:
    if t: t.set_fontsize(9)
for t in v.subset_labels:
    if t: t.set_fontsize(9); t.set_fontweight("bold")
ax.annotate(f"Anchor set: S1 ∩ iBsu1103\n= {sub[4]+sub[6]} metabolites",
            xy=(0.97,0.04), xycoords="axes fraction", fontsize=8.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.8))
ax.set_title("Integration of Three Data Sources")
save(fig, "fig01_venn.png")

# ═══════════════════════════════════════════════════════
# FIG 02 — Volcano plot
# ═══════════════════════════════════════════════════════
log.info("Fig 02 ...")
s1a    = load_s1(sig_only=False)
mapped = pd.read_csv(OUTPUT / "01_mapped_metabolites.csv")
m_low  = set(mapped["name"].str.lower().str.split(";").str[0].str.strip())
is_map = s1a["name"].str.lower().str.split(";").str[0].str.strip().isin(m_low)
yv     = -np.log10(s1a["pval"].clip(lower=1e-10))

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ns = s1a["pval"] >= 0.05
ax.scatter(s1a.loc[ns,"logfc"], yv[ns], c=GY, s=9, alpha=0.4,
           linewidths=0, label=f"NS (n={ns.sum()})", rasterized=True)
ms = (s1a["pval"]<0.05)&(s1a["fc"]<1)&~is_map
ax.scatter(s1a.loc[ms,"logfc"], yv[ms], c=B, s=14, alpha=0.75,
           linewidths=0, label=f"Soy-up (n={ms.sum()})")
mn2 = (s1a["pval"]<0.05)&(s1a["fc"]>1)&~is_map
ax.scatter(s1a.loc[mn2,"logfc"], yv[mn2], c=R, s=14, alpha=0.75,
           linewidths=0, label=f"Natto-up (n={mn2.sum()})")
ax.scatter(s1a.loc[is_map,"logfc"], yv[is_map], c=OR, s=48, alpha=0.95,
           edgecolors="white", linewidths=0.5, zorder=6,
           label=f"Anchored (n={is_map.sum()})")

top = pd.concat([mapped.nlargest(5,"fc"), mapped.nsmallest(4,"fc")]).drop_duplicates("name")
for _, r in top.iterrows():
    nm = r["name"].split(";")[0].strip()
    if len(nm)>18: nm = nm[:17]+"."
    xp = r["logfc"]; yp = -np.log10(max(r["pval"],1e-10))
    dx = 0.45 if xp>0 else -0.45
    ax.annotate(nm, xy=(xp,yp), xytext=(xp+dx, yp+0.22),
                fontsize=6, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

ax.axhline(-np.log10(0.05), color="#999999", lw=0.8, ls="--")
ax.axvline(0, color="#CCCCCC", lw=0.7, ls=":")
xlim = ax.get_xlim()
ax.text(xlim[1]*0.97, -np.log10(0.05)+0.1, "p=0.05",
        fontsize=7, color="#888888", ha="right")
ax.set_xlabel(r"$\log_2$ Fold Change  [Natto/Soybean]")
ax.set_ylabel(r"$-\log_{10}$ p-value")
ax.set_title("Differential Metabolite Volcano Plot")
ax.legend(loc="upper left", markerscale=1.3)
save(fig, "fig02_volcano.png")

# ═══════════════════════════════════════════════════════
# FIG 03 — Anchor metabolite lollipop
# ═══════════════════════════════════════════════════════
log.info("Fig 03 ...")
mapped["fc_abs"] = mapped["fc"].apply(lambda x: x if x>=1 else 1/x)
top20 = mapped.nlargest(20,"fc_abs").copy().sort_values("fc_abs")
top20["label"] = top20["name"].str.split(";").str[0].str.strip().apply(
    lambda s: s[:24]+"." if len(s)>25 else s)
top20["lfc2"] = np.log2(top20["fc_abs"]) * top20["fc"].apply(lambda x: 1 if x>=1 else -1)
top20["col"]  = top20["fc"].apply(lambda x: R if x>=1 else B)

fig, ax = plt.subplots(figsize=(6.0, 5.5))
for i,(_, r) in enumerate(top20.iterrows()):
    ax.hlines(i, 0, r["lfc2"], color=r["col"], lw=1.5, alpha=0.82)
    ax.scatter(r["lfc2"], i, color=r["col"], s=44, zorder=5,
               edgecolors="white", linewidths=0.5)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["label"].tolist(), fontsize=7.5)
ax.axvline(0, color="#444444", lw=0.8)
ax.set_xlabel(r"$\log_2$ Fold Change  [Natto/Soybean]")
ax.set_title("Top 20 Anchor Metabolites by Fold Change")
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
p1 = mpatches.Patch(color=R, label="Natto-up")
p2 = mpatches.Patch(color=B, label="Soy-up")
ax.legend(handles=[p1,p2])
save(fig, "fig03_mapped_barplot.png")

# ═══════════════════════════════════════════════════════
# FIG 04 — Flux change heatmap  (wide)
# ═══════════════════════════════════════════════════════
log.info("Fig 04 ...")
cfba = pd.read_csv(OUTPUT / "02_cfba_fluxes.csv")
cfba = cfba.set_index("reaction")
# take top 30 by abs delta, filter out huge exchange values
cfba_filt = cfba[cfba["abs_delta"] < 8000]
top30 = cfba_filt["abs_delta"].nlargest(30).index.tolist()
hm = cfba.loc[top30, ["baseline_flux","cfba_flux"]].copy()
hm.columns = ["Baseline FBA", "Natto cFBA"]
hm = hm.clip(-800, 800)

row_lbl = [r[:30]+"." if len(r)>31 else r for r in hm.index]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
vmax = hm.abs().values.max()
sns.heatmap(hm, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.3, linecolor="#EEEEEE",
            cbar_kws={"label":"Flux (relative units)", "shrink":0.65, "pad":0.02},
            yticklabels=row_lbl, xticklabels=["Baseline FBA","Natto cFBA"])
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
ax.set_title("Top 30 Reactions by Flux Change: Baseline vs. Natto-constrained FBA")
ax.set_ylabel("")
save(fig, "fig04_flux_heatmap.png")

# ═══════════════════════════════════════════════════════
# FIG 05 — Key pathway fluxes  (single)
# ═══════════════════════════════════════════════════════
log.info("Fig 05 ...")
st = pd.read_csv(OUTPUT / "02_flux_stories.csv")
st_filt = st[st["delta"].abs() < 5000].copy()
st_filt = st_filt[~st_filt["reaction"].str.startswith("EX_")]
show5 = st_filt.reindex(st_filt["delta"].abs().sort_values(ascending=False).index).head(12)
show5 = show5.sort_values("delta")
show5["label"] = show5["reaction"].apply(lambda x: x[:22])
show5["col"]   = show5["delta"].apply(lambda x: R if x>0 else B)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for i,(_, r) in enumerate(show5.iterrows()):
    ax.hlines(i, 0, r["delta"], color=r["col"], lw=1.5, alpha=0.85)
    ax.scatter(r["delta"], i, color=r["col"], s=42, zorder=5,
               edgecolors="white", linewidths=0.5)
ax.set_yticks(range(len(show5)))
ax.set_yticklabels(show5["label"].tolist(), fontsize=7.5)
ax.axvline(0, color="#444444", lw=0.8)
ax.set_xlabel("Flux Change Δv  [Natto − Baseline]")
ax.set_title("Top Flux Changes: Key Metabolic Reactions")
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
p1 = mpatches.Patch(color=R, label="Natto-up"); p2 = mpatches.Patch(color=B, label="Natto-down")
ax.legend(handles=[p1,p2])
save(fig, "fig05_tca_fluxes.png")

# ═══════════════════════════════════════════════════════
# FIG 06 — AA / GABA fluxes  (single)
# ═══════════════════════════════════════════════════════
log.info("Fig 06 ...")
aa_kw = ["amino","glutam","aspart","lysine","arginin","histidin",
         "gaba","tyrosin","phenylalan","tryptoph","alanine","valine","serine"]
mask_aa = st["reaction"].str.lower().apply(
    lambda x: any(k in x for k in aa_kw))
aa = st[mask_aa & (st["delta"].abs()<5000)].copy()
if len(aa) < 3:
    aa = st_filt.copy()
aa = aa.sort_values("delta").head(12)
aa["label"] = aa["reaction"].apply(lambda x: x[:22])
aa["col"]   = aa["delta"].apply(lambda x: R if x>0 else B)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for i,(_, r) in enumerate(aa.iterrows()):
    ax.hlines(i, 0, r["delta"], color=r["col"], lw=1.5, alpha=0.85)
    ax.scatter(r["delta"], i, color=r["col"], s=42, zorder=5,
               edgecolors="white", linewidths=0.5)
ax.set_yticks(range(len(aa)))
ax.set_yticklabels(aa["label"].tolist(), fontsize=7.5)
ax.axvline(0, color="#444444", lw=0.8)
ax.set_xlabel("Flux Change Δv  [Natto − Baseline]")
ax.set_title("Amino Acid & GABA Pathway Flux Changes")
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[p1,p2])
save(fig, "fig06_aa_gaba_fluxes.png")

# ═══════════════════════════════════════════════════════
# FIG 07 — PEA response  (single)
# ═══════════════════════════════════════════════════════
log.info("Fig 07 ...")
pea = pd.read_csv(OUTPUT / "03_pea_scan.csv")
# cols: pea_ub, growth, pea_flux, ri_pea
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(pea["pea_ub"], pea["pea_flux"],
        color=R, lw=2.2, marker="o", ms=5.5,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=R)
ax.fill_between(pea["pea_ub"], pea["pea_flux"], alpha=0.10, color=R)
ax.axhline(0, color="#444444", lw=0.8)
ymax = pea["pea_flux"].max()
ax.annotate("Model Gap\n(FVA max = 0)",
            xy=(pea["pea_ub"].median(), 0),
            xytext=(pea["pea_ub"].median()*0.55,
                    max(ymax*0.5, pea["pea_flux"].std()*2+0.01)),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF9E7", ec=OR, lw=0.9))
ax.set_xlabel("PEA exchange upper bound (allowed)")
ax.set_ylabel("Actual PEA secretion flux")
ax.set_title("Phenethylamine Secretion Response Curve")
save(fig, "fig07_pea_response.png")

# ═══════════════════════════════════════════════════════
# FIG 08 — Risk overview  (wide, 2 panels A+B)
# ═══════════════════════════════════════════════════════
log.info("Fig 08 ...")
rf = pd.read_csv(OUTPUT / "03_risk_fluxes.csv")
ri = pd.read_csv(OUTPUT / "03_risk_index.csv")
# ri cols: scenario, Phenethylamine, Phenylacetyl-Gln(PAGln), L-Arg (ADMA proxy), RI_total

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Panel A — grouped bar for risk metabolite max fluxes
ax = axes[0]
mets = rf["metabolite"].tolist()[:5]
x = np.arange(len(mets)); w = 0.33
ax.bar(x-w/2, rf["baseline_flux"].head(5).abs(), w,
       color=GY, label="Baseline FBA", edgecolor="white")
ax.bar(x+w/2, rf["cfba_flux"].head(5).abs(), w,
       color=R, alpha=0.85, label="Natto cFBA", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([m[:12] for m in mets], rotation=25, ha="right", fontsize=7.5)
ax.set_ylabel("Max secretion flux (rel. units)")
ax.set_title("Risk Metabolite Flux Comparison")
ax.legend(fontsize=7.5)
lbl(ax, "A")

# Panel B — RI heatmap per scenario
ax2 = axes[1]
ri_data = ri.set_index("scenario")
ri_num  = ri_data.select_dtypes(include=np.number)
sns.heatmap(ri_num.T, ax=ax2, cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size":7.5},
            cbar_kws={"label":"Risk Index (RI)", "shrink":0.85})
ax2.set_xlabel(""); ax2.set_ylabel("")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=7.5, rotation=0)
ax2.set_title("Risk Index Heatmap by Scenario")
lbl(ax2, "B")
save(fig, "fig08_risk_overview.png")

# ═══════════════════════════════════════════════════════
# FIG 09 — RI summary bar  (single)
# ═══════════════════════════════════════════════════════
log.info("Fig 09 ...")
fig, ax = plt.subplots(figsize=(5.2, 3.5))
ri_tot = ri[["scenario","RI_total"]].copy()
cols_ri = [R if v>=1 else OR if v>=0.5 else GR for v in ri_tot["RI_total"]]
bars = ax.barh(ri_tot["scenario"], ri_tot["RI_total"],
               color=cols_ri, edgecolor="white", height=0.55)
ax.axvline(1, color=R, lw=1.2, ls="--", label="RI = 1 (threshold)")
ax.axvline(0, color="#444444", lw=0.8)
for bar, val in zip(bars, ri_tot["RI_total"]):
    ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=7.5, fontweight="bold")
ax.set_xlabel("Composite Risk Index (RI total)")
ax.set_title("Food Safety Risk Index by Fermentation Scenario")
ax.legend(fontsize=7.5)
ax.set_xlim(0, ri_tot["RI_total"].max()*1.18)
save(fig, "fig09_risk_index_heatmap.png")

# ═══════════════════════════════════════════════════════
# FIG 10 — Yield gap  (wide, 3 panels)
# ═══════════════════════════════════════════════════════
log.info("Fig 10 ...")
yg = pd.read_csv(OUTPUT / "04_yield_gap.csv")
# cols: product, exchange_rxn, fc_observed, industrial_value,
#       theo_max_flux, prac_max_flux, current_flux, yield_gap_pct

fig, axes = plt.subplots(1, min(len(yg),3), figsize=(9.5, 3.6))
if not hasattr(axes,"__len__"): axes=[axes]
pal = [R, OR, B]
cats = ["Theoretical\nMax", "Practical Max\n(80% growth)", "Current\ncFBA"]
keys = ["theo_max_flux","prac_max_flux","current_flux"]

for i,(ax,(_,row)) in enumerate(zip(axes, yg.head(3).iterrows())):
    vals = [abs(float(row.get(k,0))) for k in keys]
    vals = [min(v,9999) for v in vals]
    clrs = ["#CCCCCC", pal[i], R]
    bars = ax.bar(cats, vals, color=clrs, width=0.5,
                  edgecolor="white", linewidth=0.6)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f"{val:.0f}" if val<1000 else f"{val/1000:.1f}k",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    gap = row.get("yield_gap_pct", 0)
    if gap and gap != gap: gap = 0  # NaN check
    if float(str(gap).replace("%","")) > 0:
        ax.annotate(f"Gap\n{gap}",
                    xy=(1, vals[1]), xytext=(1.6, vals[1]*0.65),
                    fontsize=7, color=R, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=R, lw=0.9))
    ax.set_xticklabels(cats, fontsize=7.5)
    prod = str(row.get("product",""))
    ax.set_title(prod[:18] + (f"\n(FC={row.get('fc_observed',0):.0f}x)" if "fc_observed" in row.index else ""))
    ax.set_ylabel("Flux (rel. units)" if i==0 else "")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"{v:.0f}" if v<1000 else f"{v/1000:.1f}k"))
    lbl(ax, "ABC"[i])
save(fig, "fig10_yield_gap.png")

# ═══════════════════════════════════════════════════════
# FIG 11 — FSEOF targets  (single)
# ═══════════════════════════════════════════════════════
log.info("Fig 11 ...")
fs = pd.read_csv(OUTPUT / "04_fseof_L_Histidine.csv")
# cols: reaction, rxn_name, slope, intercept, gpr, n_genes, target_type
fs = fs[fs["slope"].abs() > 0.01].copy()
top_p = fs.nlargest(8,"slope")
top_n = fs.nsmallest(8,"slope")
show = pd.concat([top_p,top_n]).drop_duplicates("reaction").sort_values("slope")
show["label"] = show["reaction"].apply(lambda x: x[:22])
show["col"]   = show["slope"].apply(lambda x: GR if x>0 else PU)

fig, ax = plt.subplots(figsize=(6.0, 5.0))
for i,(_, r) in enumerate(show.iterrows()):
    ax.hlines(i, 0, r["slope"], color=r["col"], lw=1.8, alpha=0.88)
    ax.scatter(r["slope"], i, color=r["col"], s=50, zorder=5,
               edgecolors="white", linewidths=0.6)
ax.set_yticks(range(len(show)))
ax.set_yticklabels(show["label"].tolist(), fontsize=7.5)
ax.axvline(0, color="#444444", lw=0.9)
ax.set_xlabel("Flux-Production Regression Slope")
ax.set_title("L-Histidine FSEOF Engineering Targets")
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
p1 = mpatches.Patch(color=GR, label="Overexpression (positive slope)")
p2 = mpatches.Patch(color=PU, label="Knockout (negative slope)")
ax.legend(handles=[p1,p2], fontsize=7.5)
save(fig, "fig11_engineering_targets.png")

# ═══════════════════════════════════════════════════════
# FIG 13 — Essential gene map  (wide, 2 panels)
# ═══════════════════════════════════════════════════════
log.info("Fig 13 ...")
ko_all  = pd.read_csv(OUTPUT / "05_essential_natto.csv")
ko_std  = pd.read_csv(OUTPUT / "05_essential_standard.csv")
ko_spec = pd.read_csv(OUTPUT / "05_natto_specific_essential.csv")
n_natto = len(ko_all)
n_std   = len(ko_std)
n_spec  = len(ko_spec)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Panel A — summary bar
ax = axes[0]
cats_e = ["Standard\ncondition", "Natto\ncondition", "Natto-specific\n(unique)"]
vals_e = [n_std, n_natto, n_spec]
clrs_e = [B, R, OR]
bars_e = ax.bar(cats_e, vals_e, color=clrs_e, width=0.5,
                edgecolor="white", linewidth=0.6)
for bar,val in zip(bars_e,vals_e):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.5, str(val),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of essential genes")
ax.set_title("Essential Genes: Standard vs. Natto Conditions")
ax.set_ylim(0, max(vals_e)*1.18)
lbl(ax, "A")

# Panel B — growth rate distribution
ax2 = axes[1]
gr_all  = ko_all["growth_natto"].clip(0,1.5)
gr_spec = ko_spec["growth_natto"].clip(0,1.5)
ax2.hist(gr_all, bins=35, color=GY, alpha=0.65, density=True,
         label=f"All essential (n={n_natto})", edgecolor="white")
ax2.hist(gr_spec, bins=max(4,len(gr_spec)//2+1),
         color=OR, alpha=0.88, density=True,
         label=f"Natto-specific (n={n_spec})", edgecolor="white")
ax2.axvline(0.01, color=R, lw=1.6, ls="--", label="Lethality cutoff (1%)")
ax2.set_xlabel("Normalized growth rate (KO / WT)")
ax2.set_ylabel("Density")
ax2.set_title("Growth Rate Distribution after Gene Knockout")
ax2.legend(fontsize=7.5)
lbl(ax2, "B")
save(fig, "fig13_essential_gene_map.png")

# ═══════════════════════════════════════════════════════
# FIG 14 — Network topology  (wide)
# ═══════════════════════════════════════════════════════
log.info("Fig 14 ...")
try:
    import networkx as nx
    nd = pd.read_csv(OUTPUT / "05_network_nodes.csv")
    # cols: node_id, name, degree, betweenness, compartment, is_anchor, direction

    top_n = nd.nlargest(220, "degree")
    G = nx.Graph()
    G.add_nodes_from(top_n["node_id"].tolist())

    import cobra
    model = cobra.io.read_sbml_model(str(DATA/"MODEL1507180015_url.xml"))
    for rxn in model.reactions:
        if rxn.id.startswith("EX_") or "bio" in rxn.id.lower(): continue
        mids = [m.id for m in list(rxn.reactants)+list(rxn.products) if m.id in G.nodes]
        for a in range(len(mids)):
            for b in range(a+1,len(mids)):
                G.add_edge(mids[a],mids[b])
    G = G.subgraph([n for n in G.nodes if G.degree(n)>0]).copy()

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    pos = nx.spring_layout(G, k=0.24, iterations=45, seed=42)

    anchor_info = {r["node_id"]:(r["direction"], r["degree"])
                   for _,r in top_n[top_n["is_anchor"]].iterrows()}

    nc, ns, na = [], [], []
    for n in G.nodes:
        deg = G.degree(n)
        if n in anchor_info:
            d = anchor_info[n][0]
            nc.append(R if d=="natto_up" else (B if d=="soy_up" else OR))
            ns.append(min(deg*22+45,380))
            na.append(0.92)
        else:
            nc.append(GY); ns.append(min(deg*5+6,70)); na.append(0.35)

    nx.draw_networkx_edges(G, pos, alpha=0.055, edge_color="#999999",
                           width=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=ns, ax=ax)

    # label top 12 hubs
    top12 = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:12]
    met_dict = {m.id: m.name for m in model.metabolites}
    for n in top12:
        nm = met_dict.get(n, n)[:14]
        x,y = pos[n]
        ax.text(x, y+0.022, nm, fontsize=5.5, ha="center", va="bottom",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    handles = [mpatches.Patch(color=R, label="Natto-up anchors"),
               mpatches.Patch(color=B, label="Soy-up anchors"),
               mpatches.Patch(color=OR, label="Other anchors"),
               mpatches.Patch(color=GY, label="Background metabolites")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              framealpha=0.92)
    ax.set_title("iBsu1103 Metabolic Network Topology (top 220 metabolites by degree)")
    ax.axis("off")
except Exception as e:
    log.warning(f"Fig14 fallback: {e}")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.text(0.5, 0.5, f"Network plot\n(see 05_network_nodes.csv)\n{e}",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("iBsu1103 Network Topology"); ax.axis("off")
save(fig, "fig14_network_topology.png")

# ═══════════════════════════════════════════════════════
# FIG 15 — Degree distribution  (wide, 2 panels)
# ═══════════════════════════════════════════════════════
log.info("Fig 15 ...")
nd = pd.read_csv(OUTPUT / "05_network_nodes.csv")
degrees = nd["degree"].values

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Panel A — linear histogram
ax = axes[0]
ax.hist(degrees, bins=35, color=LB, alpha=0.78, edgecolor="white")
ax.set_xlabel("Degree (no. of reactions)"); ax.set_ylabel("Count")
ax.set_title("Metabolite Degree Distribution (linear)")
lbl(ax, "A")

# Panel B — log-log + power-law fit
ax2 = axes[1]
dc = pd.Series(degrees).value_counts().sort_index()
xv = dc.index.values.astype(float); yv_c = dc.values.astype(float)
mask = (xv>0)&(yv_c>0)
ax2.scatter(xv[mask], yv_c[mask], c=LB, s=24, alpha=0.78,
            edgecolors="white", linewidths=0.4, label="Observed", zorder=5)
if mask.sum()>3:
    xl = np.log10(xv[mask]); yl = np.log10(yv_c[mask])
    slope, intercept, r, *_ = stats.linregress(xl, yl)
    xfit = np.logspace(xl.min(), xl.max(), 80)
    ax2.plot(xfit, 10**intercept * xfit**slope, color=R, lw=2.2,
             label=f"Power-law fit: γ={abs(slope):.2f},  R²={r**2:.3f}")
anch_deg = nd[nd["is_anchor"]==True]["degree"].values if "is_anchor" in nd.columns else []
if len(anch_deg):
    for d in anch_deg:
        fc_v = dc.get(d, 1)
        ax2.scatter(d, fc_v, c=OR, s=55, zorder=7,
                    edgecolors="white", linewidths=0.5)
    ax2.scatter([],[],c=OR,s=55,label=f"Anchor metabolites (n={len(anch_deg)})")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("Degree k (log)"); ax2.set_ylabel("Frequency P(k) (log)")
ax2.set_title("Log-log Degree Distribution (scale-free test)")
ax2.legend(fontsize=7.5)
lbl(ax2, "B")
save(fig, "fig15_degree_distribution.png")

# ═══════════════════════════════════════════════════════
# 复制到 paper/figures/
# ═══════════════════════════════════════════════════════
PAPER_FIG = OUTPUT.parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
cnt = 0
for fn in sorted(OUTPUT.glob("fig*.png")):
    shutil.copy(fn, PAPER_FIG / fn.name); cnt += 1
log.info(f"Done — {cnt} figures copied to paper/figures/")
print(f"\n生成完成：{cnt} 张图，配色 Paul Tol bright，全英文标签")
