"""
05_essential_genes.py — 模块五：必需基因预测与网络脆弱性分析
──────────────────────────────────────────────────────────────
输出（code/output/）：
  05_essential_standard.csv      标准条件必需基因
  05_essential_natto.csv         纳豆发酵条件必需基因
  05_natto_specific_essential.csv 纳豆特异性必需基因
  05_fva_bottlenecks.csv         FVA 瓶颈反应（range<0.01）
  05_network_nodes.csv           代谢物节点拓扑属性
  05_network_edges.csv           代谢物-反应边表（二部图）
  fig13_essential_gene_map.png   必需基因通量模块分布
  fig14_network_topology.png     网络拓扑图（差异代谢物着色）
  fig15_degree_distribution.png  代谢物度分布（幂律验证）
"""

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
from cobra.flux_analysis import single_gene_deletion, flux_variability_analysis
import networkx as nx
from scipy import stats

import utils
from utils import set_style, save_fig, log_progress, OUTPUT, DATA

log = utils.get_logger("05_essential")
set_style()

MODEL_XML = DATA / "MODEL1507180015_url.xml"
BIOMASS   = "bio00006"
GROWTH_CUTOFF = 0.01   # 生长率低于此值视为致死

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
log.info(f"  Reactions:{len(model_ref.reactions)}  Metabolites:{len(model_ref.metabolites)}  Genes:{len(model_ref.genes)}")

# ══════════════════════════════════════════════════════════
# 2. 标准条件全基因组单基因敲除
# ══════════════════════════════════════════════════════════
log.info("Running standard-condition single gene deletion (all 1109 genes) ...")
with model_ref as m:
    m.objective = BIOMASS
    base_growth_std = m.slim_optimize()
    log.info(f"  Baseline growth (standard): {base_growth_std:.4f}")

    ko_std = single_gene_deletion(m, m.genes, processes=1)

ko_std.columns = ["gene_ids", "growth_std", "status_std"]
ko_std["gene_id"]    = ko_std["gene_ids"].apply(lambda s: next(iter(s)))
ko_std["essential_std"] = ko_std["growth_std"].apply(
    lambda g: True if (pd.isna(g) or g < GROWTH_CUTOFF * base_growth_std) else False
)
n_ess_std = ko_std["essential_std"].sum()
log.info(f"  Standard essential genes: {n_ess_std}")

# ══════════════════════════════════════════════════════════
# 3. 纳豆发酵条件单基因敲除
# ══════════════════════════════════════════════════════════
log.info("Running natto-condition single gene deletion ...")
with model_ref as m:
    apply_natto_constraints(m)
    m.objective = BIOMASS
    base_growth_natto = m.slim_optimize()
    log.info(f"  Baseline growth (natto): {base_growth_natto:.4f}")

    ko_natto = single_gene_deletion(m, m.genes, processes=1)

ko_natto.columns = ["gene_ids", "growth_natto", "status_natto"]
ko_natto["gene_id"]      = ko_natto["gene_ids"].apply(lambda s: next(iter(s)))
ko_natto["essential_natto"] = ko_natto["growth_natto"].apply(
    lambda g: True if (pd.isna(g) or g < GROWTH_CUTOFF * base_growth_natto) else False
)
n_ess_natto = ko_natto["essential_natto"].sum()
log.info(f"  Natto essential genes: {n_ess_natto}")

# ══════════════════════════════════════════════════════════
# 4. 合并，找纳豆特异性必需基因
# ══════════════════════════════════════════════════════════
ko_merged = ko_std[["gene_id","growth_std","essential_std"]].merge(
    ko_natto[["gene_id","growth_natto","essential_natto"]],
    on="gene_id", how="outer"
)
ko_merged["growth_ratio"] = ko_merged["growth_natto"] / base_growth_natto

# 添加基因名（iBsu1103 gene_id = peg.XXXX）
def gene_info(gene_id, model):
    try:
        g = model.genes.get_by_id(gene_id)
        rxns = [r.id for r in g.reactions]
        return pd.Series({"gene_name": g.name if g.name else gene_id,
                          "n_reactions": len(rxns),
                          "reactions": ";".join(rxns[:8])})
    except Exception:
        return pd.Series({"gene_name": gene_id, "n_reactions": 0, "reactions": ""})

log.info("Annotating gene info ...")
gene_meta = ko_merged["gene_id"].apply(lambda gid: gene_info(gid, model_ref))
ko_merged = pd.concat([ko_merged, gene_meta], axis=1)

# 标准条件必需
std_essential = ko_merged[ko_merged["essential_std"] == True].copy()
std_essential.to_csv(OUTPUT / "05_essential_standard.csv", index=False, encoding="utf-8-sig")
log.info(f"  05_essential_standard.csv saved ({len(std_essential)} genes)")

# 纳豆条件必需
natto_essential = ko_merged[ko_merged["essential_natto"] == True].copy()
natto_essential.to_csv(OUTPUT / "05_essential_natto.csv", index=False, encoding="utf-8-sig")
log.info(f"  05_essential_natto.csv saved ({len(natto_essential)} genes)")

# 纳豆特异性必需（纳豆必需 & 标准不必需）
natto_specific = ko_merged[
    (ko_merged["essential_natto"] == True) &
    (ko_merged["essential_std"] == False)
].copy()
natto_specific.to_csv(OUTPUT / "05_natto_specific_essential.csv", index=False, encoding="utf-8-sig")
log.info(f"  05_natto_specific_essential.csv saved ({len(natto_specific)} genes)")

# ══════════════════════════════════════════════════════════
# 5. FVA 瓶颈反应（从模块二读取已有结果）
# ══════════════════════════════════════════════════════════
log.info("Loading FVA bottleneck data ...")
fva_df = pd.read_csv(OUTPUT / "02_fva_ranges.csv", index_col=0)

# 添加反应名称
def rxn_name(rid, model):
    try:
        return model.reactions.get_by_id(rid).name[:60]
    except Exception:
        return rid

fva_df["rxn_name"] = [rxn_name(rid, model_ref) for rid in fva_df.index]
bottlenecks = fva_df[fva_df["range"] < 0.01].sort_values("range")
flexible    = fva_df[fva_df["range"] > 100].sort_values("range", ascending=False)

bottlenecks.to_csv(OUTPUT / "05_fva_bottlenecks.csv", encoding="utf-8-sig")
log.info(f"  Bottleneck reactions (range<0.01): {len(bottlenecks)}")
log.info(f"  Highly flexible (range>100):       {len(flexible)}")

# ══════════════════════════════════════════════════════════
# 6. 网络拓扑分析（代谢物-反应二部图）
# ══════════════════════════════════════════════════════════
log.info("Building metabolite-reaction bipartite network ...")

G = nx.DiGraph()

# 添加代谢物节点
for m in model_ref.metabolites:
    G.add_node(m.id, node_type="metabolite",
               compartment=m.compartment,
               name=m.name[:40])

# 添加反应节点和边
for rxn in model_ref.reactions:
    if rxn.id.startswith("EX_") or rxn.id.startswith("bio"):
        continue
    G.add_node(rxn.id, node_type="reaction", name=rxn.name[:40])
    for reactant in rxn.reactants:
        G.add_edge(reactant.id, rxn.id)
    for product in rxn.products:
        G.add_edge(rxn.id, product.id)

log.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 只保留代谢物节点做度分布
met_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "metabolite"]
met_degrees = {n: G.degree(n) for n in met_nodes}

# 差异代谢物（锚定节点）
mapped_df = pd.read_csv(OUTPUT / "01_mapped_metabolites.csv")
anchor_met_ids = set(
    mapped_df["model_id"]
    .str.replace("^M_", "", regex=True)
    .tolist()
)

# 计算 betweenness（仅对小子图采样，全图耗时）
log.info("Computing betweenness centrality (sampled) ...")
try:
    # 取最大连通分量
    undirected = G.to_undirected()
    largest_cc = max(nx.connected_components(undirected), key=len)
    G_sub = undirected.subgraph(largest_cc)
    # 采样 200 nodes
    sample_nodes = list(G_sub.nodes)[:200]
    btw = nx.betweenness_centrality_subset(G_sub, sample_nodes, sample_nodes)
    log.info(f"  Betweenness computed for {len(btw)} nodes (sampled)")
except Exception as e:
    log.warning(f"  Betweenness failed: {e}")
    btw = {n: 0.0 for n in G.nodes}

# 输出节点属性表
node_rows = []
for n in met_nodes:
    node_rows.append({
        "node_id":       n,
        "name":          G.nodes[n].get("name", n),
        "degree":        G.degree(n),
        "betweenness":   round(btw.get(n, 0.0), 8),
        "compartment":   G.nodes[n].get("compartment", ""),
        "is_anchor":     n in anchor_met_ids,
        "direction":     mapped_df[mapped_df["model_id"] == "M_"+n]["direction"].values[0]
                         if n in anchor_met_ids else "",
    })
node_df = pd.DataFrame(node_rows).sort_values("degree", ascending=False)
node_df.to_csv(OUTPUT / "05_network_nodes.csv", index=False, encoding="utf-8-sig")
log.info(f"  05_network_nodes.csv saved ({len(node_df)} metabolite nodes)")

# ══════════════════════════════════════════════════════════
# 7. 图13：必需基因分布（标准 vs 纳豆条件）
# ══════════════════════════════════════════════════════════
log.info("Drawing fig13_essential_gene_map.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：必需基因数量对比（Venn-style 柱状图）
ax = axes[0]
categories = ["Standard only", "Both conditions", "Natto only (natto-specific)"]
std_only    = ((ko_merged["essential_std"] == True) & (ko_merged["essential_natto"] == False)).sum()
both_ess    = ((ko_merged["essential_std"] == True) & (ko_merged["essential_natto"] == True)).sum()
natto_only  = len(natto_specific)
values = [std_only, both_ess, natto_only]
colors = [utils.PALETTE["soy_up"], utils.PALETTE["neutral"], utils.PALETTE["natto_up"]]
bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Number of genes", fontsize=10)
ax.set_title("Essential Gene Distribution\nStandard vs. Natto Conditions",
             fontsize=10, fontweight="bold")
ax.set_xticklabels(categories, fontsize=8.5, rotation=10, ha="right")

# 右：natto-specific 必需基因的 growth ratio 分布
ax2 = axes[1]
all_growth_ratio = ko_merged["growth_ratio"].dropna()
spec_growth_ratio = natto_specific["growth_ratio"].dropna()

ax2.hist(all_growth_ratio.clip(0, 1.5), bins=40, color=utils.PALETTE["neutral"],
         alpha=0.6, label=f"All genes (n={len(all_growth_ratio)})", density=True)
if len(spec_growth_ratio) > 0:
    ax2.hist(spec_growth_ratio.clip(0, 1.5), bins=max(5, len(spec_growth_ratio)//2),
             color=utils.PALETTE["natto_up"], alpha=0.8,
             label=f"Natto-specific essential (n={len(spec_growth_ratio)})", density=True)

ax2.axvline(GROWTH_CUTOFF, color="#e74c3c", lw=1.5, ls="--", label=f"Lethality cutoff ({GROWTH_CUTOFF})")
ax2.set_xlabel("Normalized growth rate after KO (natto/baseline)", fontsize=9)
ax2.set_ylabel("Density", fontsize=9)
ax2.set_title("Growth Rate Distribution After Gene KO\n(Natto Conditions)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)

plt.tight_layout()
save_fig(fig, "fig13_essential_gene_map.png")
log.info("  fig13_essential_gene_map.png saved")

# ══════════════════════════════════════════════════════════
# 8. 图14：网络拓扑图（差异代谢物高亮）
# ══════════════════════════════════════════════════════════
log.info("Drawing fig14_network_topology.png ...")

# 取前 300 个代谢物节点（按 degree 降序）做可视化
top_met_nodes = node_df.head(300)["node_id"].tolist()
G_vis = undirected.subgraph(top_met_nodes).copy()

fig, ax = plt.subplots(figsize=(12, 10))

# 布局
try:
    pos = nx.spring_layout(G_vis, k=0.15, iterations=30, seed=42)
except Exception:
    pos = nx.random_layout(G_vis, seed=42)

# 颜色与大小
node_colors, node_sizes = [], []
for n in G_vis.nodes:
    deg = G_vis.degree(n)
    if n in anchor_met_ids:
        direction = node_df[node_df["node_id"] == n]["direction"].values
        if len(direction) > 0 and direction[0] == "natto_up":
            node_colors.append(utils.PALETTE["natto_up"])
        elif len(direction) > 0 and direction[0] == "soy_up":
            node_colors.append(utils.PALETTE["soy_up"])
        else:
            node_colors.append(utils.PALETTE["hit"])
        node_sizes.append(min(deg * 20, 400))
    else:
        node_colors.append(utils.PALETTE["neutral"])
        node_sizes.append(min(deg * 5, 100))

nx.draw_networkx_edges(G_vis, pos, alpha=0.08, edge_color="#aaa", width=0.4, ax=ax)
nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)

# 标注 top 20 高度节点
top20_nodes = sorted(G_vis.nodes, key=lambda n: G_vis.degree(n), reverse=True)[:20]
labels_dict = {n: G_vis.nodes[n].get("name", n)[:15] for n in top20_nodes}
nx.draw_networkx_labels(G_vis, pos, labels=labels_dict, font_size=6, ax=ax)

patches = [
    mpatches.Patch(color=utils.PALETTE["natto_up"],  label="Natto-up anchors"),
    mpatches.Patch(color=utils.PALETTE["soy_up"],    label="Soy-up anchors"),
    mpatches.Patch(color=utils.PALETTE["hit"],       label="Other anchors"),
    mpatches.Patch(color=utils.PALETTE["neutral"],   label="Non-anchor metabolites"),
]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.set_title("iBsu1103 Metabolic Network Topology\n(Top 300 metabolites by degree; anchor nodes colored)",
             fontsize=11, fontweight="bold", pad=12)
ax.axis("off")

save_fig(fig, "fig14_network_topology.png")
log.info("  fig14_network_topology.png saved")

# ══════════════════════════════════════════════════════════
# 9. 图15：代谢物度分布（幂律拟合）
# ══════════════════════════════════════════════════════════
log.info("Drawing fig15_degree_distribution.png ...")

degrees = node_df["degree"].values
deg_counts = pd.Series(degrees).value_counts().sort_index()
deg_vals = deg_counts.index.values
deg_freq = deg_counts.values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：线性坐标直方图
ax = axes[0]
ax.hist(degrees, bins=40, color=utils.PALETTE["neutral"], alpha=0.8, edgecolor="white")
ax.set_xlabel("Degree (number of reactions)", fontsize=10)
ax.set_ylabel("Number of metabolites", fontsize=10)
ax.set_title("Metabolite Degree Distribution\n(Linear scale)", fontsize=10, fontweight="bold")

# 右：log-log 坐标 + 幂律拟合
ax2 = axes[1]
# 过滤零值
mask = (deg_vals > 0) & (deg_freq > 0)
x_log = np.log10(deg_vals[mask])
y_log = np.log10(deg_freq[mask])

ax2.scatter(deg_vals[mask], deg_freq[mask],
            color=utils.PALETTE["neutral"], alpha=0.7, s=30, label="Observed")

# 幂律拟合（线性回归在 log-log 空间）
if len(x_log) > 3:
    slope_pow, intercept_pow, r_val, p_val, _ = stats.linregress(x_log, y_log)
    x_fit = np.logspace(x_log.min(), x_log.max(), 100)
    y_fit = 10**intercept_pow * x_fit**slope_pow
    ax2.plot(x_fit, y_fit, color=utils.PALETTE["natto_up"], lw=2.5,
             label=f"Power-law fit: γ={abs(slope_pow):.2f}\nR²={r_val**2:.3f}, p={p_val:.2e}")
    log.info(f"  Power-law fit: gamma={abs(slope_pow):.3f}, R2={r_val**2:.4f}")

# 高亮锚定代谢物
anchor_degs = node_df[node_df["is_anchor"] == True]["degree"].values
for d in anchor_degs:
    freq_v = deg_counts.get(d, 1)
    ax2.scatter(d, freq_v, color=utils.PALETTE["hit"], s=60, zorder=5, alpha=0.9)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Degree k (log)", fontsize=10)
ax2.set_ylabel("Count P(k) (log)", fontsize=10)
ax2.set_title("Degree Distribution (Log-Log)\nPower-law verification",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=9)

# 锚定代谢物标注
ax2.scatter([], [], color=utils.PALETTE["hit"], s=60, label=f"Anchor metabolites (n={len(anchor_degs)})")
ax2.legend(fontsize=8)

plt.tight_layout()
save_fig(fig, "fig15_degree_distribution.png")
log.info("  fig15_degree_distribution.png saved")

# ══════════════════════════════════════════════════════════
# 10. 控制台摘要
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Module 5 Complete: Essential Genes & Network Vulnerability")
print("=" * 65)
print()
print(f"  Total genes analyzed:           {len(ko_merged)}")
print(f"  Standard essential:             {n_ess_std}")
print(f"  Natto essential:                {n_ess_natto}")
print(f"  Natto-specific essential:       {len(natto_specific)}")
print()
if len(natto_specific) > 0:
    print("  Top 10 natto-specific essential genes:")
    top10 = natto_specific.sort_values("growth_ratio").head(10)
    for _, r in top10.iterrows():
        gr = r["growth_natto"] if not pd.isna(r["growth_natto"]) else 0.0
        print(f"    {r['gene_id']:20}  growth={gr:.4f}  rxns={r['reactions'][:50]}")
print()
print(f"  FVA bottleneck reactions (range<0.01): {len(bottlenecks)}")
print(f"  Highly flexible reactions (range>100): {len(flexible)}")
print()
print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
top5_met = node_df.head(5)
print("  Top 5 hub metabolites (by degree):")
for _, r in top5_met.iterrows():
    anchor_str = "[ANCHOR]" if r["is_anchor"] else ""
    print(f"    {r['node_id']:20}  degree={r['degree']:4}  {r['name'][:35]} {anchor_str}")
print()
print("  Output files:")
for f_name in ["05_essential_standard.csv", "05_essential_natto.csv",
               "05_natto_specific_essential.csv", "05_fva_bottlenecks.csv",
               "05_network_nodes.csv",
               "fig13_essential_gene_map.png", "fig14_network_topology.png",
               "fig15_degree_distribution.png"]:
    status = "[OK]" if (OUTPUT / f_name).exists() else "[--]"
    print(f"    {status}  code/output/{f_name}")
print("=" * 65)

log_progress("Module 5",
             f"Essential genes: std={n_ess_std}, natto={n_ess_natto}, "
             f"natto-specific={len(natto_specific)}; network nodes={G.number_of_nodes()}")
log.info("Module 5 complete. ALL MODULES DONE.")
