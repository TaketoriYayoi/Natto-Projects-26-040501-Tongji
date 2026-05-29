import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3          # pip install matplotlib-venn
import seaborn as sns
from collections import defaultdict

import utils
from utils import (
    load_s1, load_s2, load_model_metadata,
    fuzzy_match_to_model, fc_to_exchange_bounds,
    set_style, save_fig, log_progress, OUTPUT,
)

log = utils.get_logger("01_integration")
set_style()

# ══════════════════════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════════════════════
log.info("加载 Table S1 ...")
s1 = load_s1(sig_only=True, p_thresh=0.05)
log.info(f"  显著差异代谢物: {len(s1)} 个  (纳豆上调={( s1.fc>1).sum()}, 大豆上调={(s1.fc<1).sum()})")

log.info("加载 Table S2 ...")
s2 = load_s2()
log.info(f"  KEGG 富集路径: {len(s2)} 条")

log.info("解析 iBsu1103 XML ...")
species_df, rxn_df = load_model_metadata()
log.info(f"  代谢物: {len(species_df)}  反应: {len(rxn_df)}")

# ══════════════════════════════════════════════════════════
# 2. S1 代谢物 → iBsu1103 节点映射
# ══════════════════════════════════════════════════════════
log.info("开始节点映射 ...")

mapped_rows   = []
unmapped_rows = []

for _, row in s1.iterrows():
    hit = fuzzy_match_to_model(row["name"], species_df)
    if hit is not None:
        lb, ub = fc_to_exchange_bounds(row["fc"])
        mapped_rows.append({
            # S1 信息
            "name":        row["name"],
            "kegg":        row.get("kegg", ""),
            "cls":         row.get("cls", ""),
            "fc":          round(row["fc"], 4),
            "logfc":       round(row.get("logfc", np.log2(row["fc"])), 4),
            "pval":        round(row["pval"], 6),
            "mean_natto":  round(row.get("mean_natto", 0), 6),
            "mean_soy":    round(row.get("mean_soy", 0), 6),
            "direction":   row["direction"],
            # 模型节点信息
            "model_id":    hit["id"],
            "model_name":  hit["name"],
            "compartment": hit["compartment"],
            # exchange 约束
            "ex_lb":       lb,
            "ex_ub":       ub,
        })
    else:
        unmapped_rows.append(row.to_dict())

mapped   = pd.DataFrame(mapped_rows)
unmapped = pd.DataFrame(unmapped_rows)

log.info(f"  命中模型节点: {len(mapped)} 个")
log.info(f"  未命中:        {len(unmapped)} 个")

# ══════════════════════════════════════════════════════════
# 3. 补充 Table S2 通路注释
# ══════════════════════════════════════════════════════════
log.info("补充 KEGG 通路注释 ...")

# 建立 compound名 → pathway 的反向索引
cpd_to_path = defaultdict(list)
for _, row in s2.iterrows():
    pname = str(row.get("pathway_name", row["pathway"]))
    for cpd in row["compound_list"]:
        # 去掉 KEGG cpd: 前缀和ID，只保留名称
        if " " in cpd:
            cpd_name = cpd.split(" ", 1)[1].strip()
        else:
            cpd_name = cpd.strip()
        cpd_to_path[cpd_name.lower()].append(pname)

def find_pathways(name: str) -> str:
    nm = name.lower().split(";")[0].strip()
    hits = []
    for cpd_key, paths in cpd_to_path.items():
        if nm in cpd_key or cpd_key in nm:
            hits.extend(paths)
    return " | ".join(sorted(set(hits))) if hits else ""

mapped["pathways_s2"] = mapped["name"].apply(find_pathways)

# ══════════════════════════════════════════════════════════
# 4. 生成约束表（用于模块二）
# ══════════════════════════════════════════════════════════
# 约束表只保留有 exchange 方向的代谢物（lb>0 或 ub<0）
constraints = mapped[
    (mapped["ex_lb"] > -999) | (mapped["ex_ub"] < 999)
].copy()
constraints["constraint_type"] = constraints.apply(
    lambda r: "secretion" if r["ex_lb"] > 0 else "uptake", axis=1
)
log.info(f"  有效 exchange 约束: {len(constraints)} 个  "
         f"(分泌={( constraints.constraint_type=='secretion').sum()}, "
         f"摄取={(constraints.constraint_type=='uptake').sum()})")

# ══════════════════════════════════════════════════════════
# 5. 保存 CSV
# ══════════════════════════════════════════════════════════
mapped.to_csv(OUTPUT / "01_mapped_metabolites.csv",     index=False, encoding="utf-8-sig")
constraints.to_csv(OUTPUT / "01_constraints_table.csv", index=False, encoding="utf-8-sig")
unmapped.to_csv(OUTPUT / "01_unmapped_metabolites.csv", index=False, encoding="utf-8-sig")
log.info("CSV 文件已保存至 code/output/")

# ══════════════════════════════════════════════════════════
# 6. 图1：三数据源韦恩图
# ══════════════════════════════════════════════════════════
log.info("绘制 fig01_venn.png ...")

# 集合定义
s1_names   = set(s1["name"].str.lower().str.split(";").str[0].str.strip())
s2_cpds    = set()
for _, row in s2.iterrows():
    for cpd in row["compound_list"]:
        nm = (cpd.split(" ", 1)[1].strip() if " " in cpd else cpd.strip()).lower()
        s2_cpds.add(nm)
model_names = set(species_df["name_short"].dropna())

# 计算交集
only_s1       = s1_names - s2_cpds - model_names
only_s2       = s2_cpds  - s1_names - model_names
only_model    = model_names - s1_names - s2_cpds
s1_s2         = (s1_names & s2_cpds) - model_names
s1_model      = (s1_names & model_names) - s2_cpds
s2_model      = (s2_cpds  & model_names) - s1_names
all_three     = s1_names & s2_cpds & model_names

fig, ax = plt.subplots(figsize=(7, 6))
try:
    venn3(
        subsets=(
            len(only_s1), len(only_s2), len(s1_s2),
            len(only_model), len(s1_model), len(s2_model),
            len(all_three),
        ),
        set_labels=("Table S1\n(161 sig.)", "Table S2\n(52 pathways)", "iBsu1103\n(1381 metabolites)"),
        set_colors=("#E74C3C", "#3498DB", "#27AE60"),
        alpha=0.55,
        ax=ax,
    )
except Exception:
    # matplotlib_venn 未安装时降级为文字说明
    ax.text(0.5, 0.5,
            f"Three-source overlap\n\n"
            f"S1 ∩ Model = {len(s1_model) + len(all_three)}\n"
            f"S1 ∩ S2   = {len(s1_s2)   + len(all_three)}\n"
            f"All three  = {len(all_three)}",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.axis("off")

ax.set_title("Data Source Integration Overview", fontsize=14, fontweight="bold", pad=12)

# 注释框
info_text = (
    f"Key intersection: S1 ∩ iBsu1103\n"
    f"  = {len(s1_model) + len(all_three)} metabolites\n"
    f"  (anchor nodes for cFBA)"
)
ax.annotate(info_text, xy=(0.98, 0.02), xycoords="axes fraction",
            fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#ccc"),
            ha="right", va="bottom")

save_fig(fig, "fig01_venn.png")
log.info("  fig01_venn.png 已保存")

# ══════════════════════════════════════════════════════════
# 7. 图2：Volcano Plot（标注命中节点）
# ══════════════════════════════════════════════════════════
log.info("绘制 fig02_volcano.png ...")

s1_all = load_s1(sig_only=False)   # 全量（不过滤显著性）

fig, ax = plt.subplots(figsize=(9, 6))

# 颜色分层
def point_color(row):
    if row["pval"] >= 0.05:
        return utils.PALETTE["neutral"]
    return utils.PALETTE["natto_up"] if row["fc"] > 1 else utils.PALETTE["soy_up"]

colors = s1_all.apply(point_color, axis=1)

# 底层：全部点
ax.scatter(
    s1_all["logfc"], -np.log10(s1_all["pval"].clip(lower=1e-10)),
    c=colors, s=18, alpha=0.6, linewidths=0,
)

# 高亮：命中模型节点的点
mapped_names_lower = set(mapped["name"].str.lower().str.split(";").str[0].str.strip())
is_mapped = s1_all["name"].str.lower().str.split(";").str[0].str.strip().isin(mapped_names_lower)

ax.scatter(
    s1_all.loc[is_mapped, "logfc"],
    -np.log10(s1_all.loc[is_mapped, "pval"].clip(lower=1e-10)),
    c=utils.PALETTE["hit"], s=55, alpha=0.9,
    edgecolors="white", linewidths=0.5,
    zorder=5, label=f"Mapped to iBsu1103 (n={is_mapped.sum()})",
)

# 标注 Top 15 高倍差异代谢物（FC 最极端 + 命中模型）
top_label = (
    mapped.nlargest(8, "fc")
    .append(mapped.nsmallest(7, "fc"))
    if hasattr(mapped, "append")
    else pd.concat([mapped.nlargest(8, "fc"), mapped.nsmallest(7, "fc")])
)
for _, r in top_label.iterrows():
    short = r["name"].split(";")[0][:22]
    xv = r["logfc"]
    yv = -np.log10(max(r["pval"], 1e-10))
    ax.annotate(
        short, xy=(xv, yv),
        xytext=(xv + (0.3 if xv > 0 else -0.3), yv + 0.15),
        fontsize=6.5, color="#333",
        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
    )

# 参考线
ax.axhline(-np.log10(0.05), color="#999", lw=0.8, ls="--")
ax.axvline(0, color="#bbb", lw=0.6, ls=":")

# 图例
patches = [
    mpatches.Patch(color=utils.PALETTE["natto_up"],  label="Natto↑ (p<0.05)"),
    mpatches.Patch(color=utils.PALETTE["soy_up"],    label="Soybean↑ (p<0.05)"),
    mpatches.Patch(color=utils.PALETTE["neutral"],   label="Not significant"),
    mpatches.Patch(color=utils.PALETTE["hit"],       label=f"Mapped to iBsu1103 (n={is_mapped.sum()})"),
]
ax.legend(handles=patches, fontsize=8, loc="upper left", framealpha=0.9)

ax.set_xlabel("log₂(Fold Change)  [Natto / Soybean]", fontsize=11)
ax.set_ylabel("−log₁₀(p-value)", fontsize=11)
ax.set_title("Natto vs Soybean: Differential Metabolites\n(Orange = anchored to iBsu1103 nodes)",
             fontsize=12, fontweight="bold")

save_fig(fig, "fig02_volcano.png")
log.info("  fig02_volcano.png 已保存")

# ══════════════════════════════════════════════════════════
# 8. 图3：命中代谢物按类别 & 方向分布
# ══════════════════════════════════════════════════════════
log.info("绘制 fig03_mapped_barplot.png ...")

# 清理类别名（合并近似类别）
def clean_cls(c: str) -> str:
    c = str(c).strip()
    if "amino acid" in c.lower():   return "Amino acids & derivatives"
    if "flavon" in c.lower():       return "Flavonoids"
    if "alkaloid" in c.lower():     return "Alkaloids"
    if "fatty" in c.lower():        return "Fatty acyls"
    if "nucleotide" in c.lower():   return "Nucleotides"
    if "phenol" in c.lower():       return "Phenols"
    if "carboxylic" in c.lower():   return "Carboxylic acids"
    if "organooxygen" in c.lower(): return "Organooxygen cpds"
    if "terpenoid" in c.lower() or "terpene" in c.lower(): return "Terpenoids"
    if "steroid" in c.lower():      return "Steroids"
    if "keto" in c.lower():         return "Keto acids"
    if "carbohydrate" in c.lower(): return "Carbohydrates"
    if "miscellaneous" in c.lower() or c.lower() in ("", "unknown", "none"):
        return "Others"
    return c

mapped["cls_clean"] = mapped["cls"].apply(clean_cls)

# 统计
grp = (mapped.groupby(["cls_clean", "direction"])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=["natto_up", "soy_up"], fill_value=0))
grp = grp[grp.sum(axis=1) >= 1].sort_values("natto_up", ascending=True)

fig, ax = plt.subplots(figsize=(8, max(5, len(grp) * 0.42)))

y = np.arange(len(grp))
h = 0.38
ax.barh(y + h/2, grp["natto_up"], height=h,
        color=utils.PALETTE["natto_up"], label="Natto↑", alpha=0.85)
ax.barh(y - h/2, grp["soy_up"],   height=h,
        color=utils.PALETTE["soy_up"],   label="Soybean↑", alpha=0.85)

ax.set_yticks(y)
ax.set_yticklabels(grp.index, fontsize=9)
ax.set_xlabel("Number of metabolites", fontsize=10)
ax.set_title(f"Distribution of {len(mapped)} Anchor Metabolites\nby Chemical Class and Direction",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.axvline(0, color="black", lw=0.5)

# 数字标注
for i, (nu, su) in enumerate(zip(grp["natto_up"], grp["soy_up"])):
    if nu > 0:
        ax.text(nu + 0.05, i + h/2, str(nu), va="center", fontsize=8)
    if su > 0:
        ax.text(su + 0.05, i - h/2, str(su), va="center", fontsize=8)

save_fig(fig, "fig03_mapped_barplot.png")
log.info("  fig03_mapped_barplot.png 已保存")

# ══════════════════════════════════════════════════════════
# 9. 控制台摘要
# ══════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  模块一完成：数据整合与节点映射")
print("═" * 60)
print(f"  Table S1 显著差异代谢物:   {len(s1):>4} 个")
print(f"  → 命中 iBsu1103 节点:       {len(mapped):>4} 个")
print(f"  → 未命中:                   {len(unmapped):>4} 个")
print(f"  有效 exchange 约束:         {len(constraints):>4} 个")
print(f"    ├─ 分泌方向（纳豆上调）:  {(constraints.constraint_type=='secretion').sum():>4} 个")
print(f"    └─ 摄取方向（大豆上调）:  {(constraints.constraint_type=='uptake').sum():>4} 个")
print()
print("  输出文件：")
for f in ["01_mapped_metabolites.csv", "01_constraints_table.csv",
          "01_unmapped_metabolites.csv",
          "fig01_venn.png", "fig02_volcano.png", "fig03_mapped_barplot.png"]:
    print(f"    code/output/{f}")
print("═" * 60)

# ══════════════════════════════════════════════════════════
# 10. 打印关键代谢物摘要（方便核对）
# ══════════════════════════════════════════════════════════
print("\n【Top 15 纳豆上调 & 命中模型节点】")
top_natto = mapped[mapped["direction"] == "natto_up"].nlargest(15, "fc")
print(top_natto[["name", "fc", "pval", "model_id", "pathways_s2"]].to_string(index=False))

print("\n【大豆更高 & 命中模型节点（全部）】")
soy_hits = mapped[mapped["direction"] == "soy_up"].sort_values("fc")
print(soy_hits[["name", "fc", "pval", "model_id"]].to_string(index=False))

# 进度记录
log_progress("模块一", f"节点映射完成：{len(mapped)}命中/{len(s1)}显著差异代谢物，输出3个CSV+3张图")
log.info("模块一全部完成。")
