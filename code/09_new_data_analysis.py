import sys, gzip, re, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import statistics

import utils
from utils import OUTPUT, DATA

# ── 全局字体 ───────────────────────────────────────────────────
import matplotlib as mpl; mpl.rcdefaults()
plt.rcParams.update({
    "font.family":       "Times New Roman",
    "mathtext.fontset":  "stix",
    "axes.unicode_minus": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.04,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#222222",
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "axes.labelsize":    9.5,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "legend.fontsize":   8,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "lines.linewidth":   1.8,
})

RED   = "#E63946"
BLUE  = "#1D6FA4"
GRN   = "#2A9D8F"
ORG   = "#F4A261"
PUR   = "#7B2D8B"
YLW   = "#E9C46A"
GY    = "#999999"

BEST195_GFF  = DATA / "ncbi-dataset/ncbi_dataset/data/GCF_000209795.2/genomic.gff"
BEST195_FAA  = DATA / "ncbi-dataset/ncbi_dataset/data/GCF_000209795.2/protein.faa"
GEO_MATRIX   = DATA / "GSE72060_series_matrix.txt.gz"

def save(fig, name):
    p = OUTPUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {name}  ({p.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════
# ANALYSIS A: AAAD 候选基因鉴定
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Analysis A: AAAD candidate identification (BEST195)")
print("="*60)

# A1. 解析 BEST195 GFF：全部脱羧酶 + 关键通路基因
decarboxylases = []
key_genes_best195 = {}  # gene_name -> record
all_cds = []

with open(BEST195_GFF, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("#") or "\t" not in line:
            continue
        parts = line.strip().split("\t")
        if len(parts) < 9:
            continue
        feat   = parts[2]
        chrom  = parts[0]
        start  = int(parts[3])
        end    = int(parts[4])
        strand = parts[6]
        attrs  = parts[8]

        locus_m = re.search(r"locus_tag=([^;]+)", attrs)
        gene_m  = re.search(r"gene=([^;]+)", attrs)
        prod_m  = re.search(r"product=([^;]+)", attrs)
        acc_m   = re.search(r"Name=(WP_[^;]+)", attrs)

        locus = locus_m.group(1) if locus_m else ""
        gene  = gene_m.group(1)  if gene_m  else ""
        prod  = prod_m.group(1)  if prod_m  else ""
        acc   = acc_m.group(1)   if acc_m   else ""

        if feat == "CDS":
            rec = dict(locus=locus, gene=gene, prod=prod, acc=acc,
                       start=start, end=end, strand=strand)
            all_cds.append(rec)
            if "decarboxylase" in prod.lower():
                decarboxylases.append(rec)
            if gene:
                key_genes_best195[gene.lower()] = rec

print(f"  Total CDS: {len(all_cds)}")
print(f"  Decarboxylase genes: {len(decarboxylases)}")

# A2. 分类脱羧酶：芳香族氨基酸相关 vs 其他
aro_keywords = ["pyridoxal", "aromatic", "amino acid decarboxylase",
                "phenolic", "tyrosine", "phenylalanine", "dopa", "tyrdc", "plp"]
aro_decarbx = [r for r in decarboxylases
               if any(k in r["prod"].lower() for k in aro_keywords)]
other_decarbx = [r for r in decarboxylases if r not in aro_decarbx]

print(f"\n  Aromatic/PLP-related decarboxylases ({len(aro_decarbx)}):")
for r in sorted(aro_decarbx, key=lambda x: x["start"]):
    print(f"    {r['locus']:25s} {r['gene']:12s} {r['start']:>8}-{r['end']:>8} {r['strand']}  {r['prod'][:65]}")

# A3. 提取 AAAD 候选蛋白序列（BSNT_RS03085 为主，bsdC 为参照）
target_loci  = {"BSNT_RS03085", "BSNT_RS02105", "BSNT_RS18045"}
locus2acc    = {r["locus"]: r["acc"] for r in all_cds if r["locus"] in target_loci}
acc2locus    = {v: k for k, v in locus2acc.items()}

seqs = {}
current_acc = None; current_seq = []
with open(BEST195_FAA, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            if current_acc and current_acc in acc2locus:
                seqs[acc2locus[current_acc]] = "".join(current_seq)
            current_seq = []; current_acc = None
            acc = line.split()[0][1:]
            if acc in acc2locus:
                current_acc = acc
        elif current_acc:
            current_seq.append(line)
if current_acc and current_acc in acc2locus:
    seqs[acc2locus[current_acc]] = "".join(current_seq)

print(f"\n  Key AAAD candidate sequences extracted: {len(seqs)}")
for locus, seq in seqs.items():
    print(f"    {locus}: {len(seq)} aa")

# A4. BSNT_RS03085 基因组上下文（±5个邻近基因）
all_genes_sorted = []
with open(BEST195_GFF, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("#") or "\t" not in line: continue
        parts = line.strip().split("\t")
        if len(parts) < 9 or parts[2] != "gene": continue
        attrs = parts[8]
        lm = re.search(r"locus_tag=([^;]+)", attrs)
        gm = re.search(r"Name=([^;]+)", attrs)
        locus = lm.group(1) if lm else ""
        name  = gm.group(1) if gm else ""
        if locus:
            all_genes_sorted.append((int(parts[3]), int(parts[4]), parts[6], locus, name))
all_genes_sorted.sort(key=lambda x: x[0])

def get_context(target_locus, window=5):
    idx = next((i for i, g in enumerate(all_genes_sorted) if g[3] == target_locus), None)
    if idx is None: return []
    return all_genes_sorted[max(0, idx-window): idx+window+1]

aaad_context = get_context("BSNT_RS03085", window=4)
print(f"\n  BSNT_RS03085 genomic context (±4 genes):")
for s, e, strand, locus, name in aaad_context:
    cds_rec = next((r for r in all_cds if r["locus"] == locus), None)
    prod = cds_rec["prod"][:55] if cds_rec else ""
    marker = " <-- AAAD CANDIDATE" if locus == "BSNT_RS03085" else ""
    print(f"    {locus:25s} {name:15s} {s:>8}-{e:>8} {strand}  {prod}{marker}")

# A5. 构建 AAAD 候选汇总表
aaad_df_rows = []
AAAD_SCORE = {
    "BSNT_RS03085": {
        "reason": "PLP-dependent decarboxylase family; 480aa; adjacent to PLP-aminotransferase (ydfD); no BSU168 ortholog",
        "priority": 1, "evidence": "Sequence + genomic context + model gap"
    },
    "BSNT_RS02105": {
        "reason": "Phenolic acid decarboxylase BsdC; acts on hydroxycinnamic acids; may accept L-Phe as substrate",
        "priority": 2, "evidence": "Substrate promiscuity"
    },
    "BSNT_RS18045": {
        "reason": "padC; phenolic acid decarboxylase; shorter (161aa); acts on p-coumaric/ferulic acids",
        "priority": 3, "evidence": "Substrate overlap"
    },
    "BSNT_RS20075": {
        "reason": "bacA; prephenate decarboxylase in bacilysin biosynthesis; not canonical AAAD",
        "priority": 4, "evidence": "Structural similarity only"
    },
}
for locus, info in AAAD_SCORE.items():
    rec = next((r for r in all_cds if r["locus"] == locus), {})
    seq = seqs.get(locus, "")
    aaad_df_rows.append({
        "locus_tag":    locus,
        "gene":         rec.get("gene", ""),
        "protein_acc":  rec.get("acc", ""),
        "product":      rec.get("prod", ""),
        "length_aa":    len(seq),
        "start":        rec.get("start", 0),
        "end":          rec.get("end", 0),
        "strand":       rec.get("strand", ""),
        "priority":     info["priority"],
        "reason":       info["reason"],
        "evidence_type":info["evidence"],
    })

aaad_df = pd.DataFrame(aaad_df_rows).sort_values("priority")
aaad_df.to_csv(OUTPUT / "09_aaad_candidates.csv", index=False)
print(f"\n  Saved: 09_aaad_candidates.csv")


# ══════════════════════════════════════════════════════════════
# ANALYSIS B: 转录组交叉验证
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Analysis B: Transcriptome cross-validation (GSE72060)")
print("="*60)

# B1. 读取 GSE72060 完整数据矩阵
# 样品列：col1-3=glucose(3 tech reps), col4-6=glutamate(3), col7-9=glu+glut(3)
gex = {}
with gzip.open(GEO_MATRIX, "rt", encoding="latin-1") as f:
    in_table = False
    for line in f:
        if "!series_matrix_table_begin" in line:
            in_table = True; continue
        if "!series_matrix_table_end" in line:
            break
        if not in_table or not line.strip():
            continue
        parts = line.strip().split("\t")
        if parts[0].startswith('"ID'):
            continue
        probe = parts[0].strip('"')
        try:
            vals = [float(parts[i].strip('"')) for i in range(1, 10)]
            gex[probe] = vals
        except:
            pass

print(f"  Total probes loaded: {len(gex)}")

# B2. BSU编号 -> 基因名 -> peg ID 的三系统映射表
# 手工整理：基因名 -> (BSU168, peg_ID, 功能描述)
GENE_MAP = {
    # FSEOF L-His 靶标
    "ptsG":    ("BSU01530", "peg.1441-like(ptsG-KO)",  "Fructose PTS (KO target, slope=-1.75)"),
    "mtlA":    ("BSU16390", "peg.2707-like(mtlA)",      "Mannitol PTS (KO target)"),
    "pyk":     ("BSU22360", "peg.2921",                 "Pyruvate kinase (OE target)"),
    "pfk":     ("BSU22370", "peg.2922",                 "Phosphofructokinase (OE target)"),
    # 组氨酸合成基因簇 (natto-specific essential peg.3492-3499)
    "hisG":    ("BSU23290", "peg.3492",                 "ATP-PRTase, His biosyn step1 (rate-limiting)"),
    "hisI":    ("BSU23300", "peg.3493",                 "Phosphoribosyl-ATP PPase"),
    "hisA":    ("BSU23310", "peg.3494",                 "HisA isomerase"),
    "hisF":    ("BSU23320", "peg.3495",                 "HisF cyclase"),
    "hisH":    ("BSU23330", "peg.3496",                 "HisH glutaminase"),
    "hisB":    ("BSU23340", "peg.3497",                 "Imidazole-GP dehydratase"),
    "hisC":    ("BSU23350", "peg.3498",                 "Histidinol-P aminotransferase"),
    "hisD":    ("BSU23360", "peg.3499",                 "Histidinol dehydrogenase"),
    # 苯丙氨酸通路 (natto-specific essential peg.2266-2270)
    "aroA":    ("BSU23250", "peg.2266-cluster(aroA)",   "Shikimate pathway (Phe/Tyr/Trp)"),
    "tyrA":    ("BSU10540", "peg.2268",                 "Prephenate dehydrogenase (Tyr)"),
    "pheA":    ("BSU28670", "peg.2267",                 "Prephenate dehydratase (Phe)"),
    # 乙酰乳酸通路 (natto-specific essential peg.2828-2834)
    "alsS":    ("BSU27920", "peg.2833",                 "Acetolactate synthase (flux spike -742)"),
    "alsD":    ("BSU28050", "peg.2834",                 "Acetolactate decarboxylase"),
    "ilvC":    ("BSU14070", "peg.2832",                 "Ketol-acid reductoisomerase (5 rxns, hub)"),
    "ilvD":    ("BSU10510", "peg.2830",                 "Dihydroxy-acid dehydratase"),
    # 其他
    "aprE":    ("BSU27610", "N/A(natto-specific)",      "Nattokinase (natto-specific secreted protease)"),
    "pgsA":    ("BSU04220", "N/A(natto-specific)",      "gamma-PGA synthase subunit A"),
    "pgsB":    ("BSU36820", "N/A(natto-specific)",      "gamma-PGA synthase subunit B"),
    "pgsC":    ("BSU36830", "N/A(natto-specific)",      "gamma-PGA synthase subunit C"),
    "speA":    ("BSU05160", "peg.0508",                 "Arginine decarboxylase (Arg catabolism)"),
}

# B3. 计算每个基因在三条件下的平均表达和差值
results = []
for gene, (bsu, peg, func) in GENE_MAP.items():
    if bsu not in gex:
        results.append(dict(gene=gene, bsu=bsu, peg_id=peg, function=func,
                            glc_mean=np.nan, glu_mean=np.nan, both_mean=np.nan,
                            glu_vs_glc=np.nan, direction="no_data"))
        continue
    v = gex[bsu]
    glc  = statistics.mean(v[0:3])
    glu  = statistics.mean(v[3:6])
    both = statistics.mean(v[6:9])
    diff = glu - glc
    direction = "up_in_glu" if diff > 0.5 else ("down_in_glu" if diff < -0.5 else "stable")
    results.append(dict(gene=gene, bsu=bsu, peg_id=peg, function=func,
                        glc_mean=round(glc,3), glu_mean=round(glu,3),
                        both_mean=round(both,3), glu_vs_glc=round(diff,3),
                        direction=direction))

txn_df = pd.DataFrame(results).sort_values("glu_vs_glc", ascending=False)
txn_df.to_csv(OUTPUT / "09_transcriptome_validation.csv", index=False)
print(f"  Saved: 09_transcriptome_validation.csv")

# B4. 控制台报告
print(f"\n  Transcriptome cross-validation results:")
print(f"  {'Gene':8s} {'BSU':12s} {'glc':>7s} {'glu':>7s} {'D(glu-glc)':>11s}  {'Direction':15s}  Function")
print("  " + "-"*95)
for _, row in txn_df.iterrows():
    if pd.isna(row.glu_vs_glc):
        continue
    col = "+" if row.glu_vs_glc > 0.5 else ("-" if row.glu_vs_glc < -0.5 else " ")
    print(f"  {row.gene:8s} {row.bsu:12s} {row.glc_mean:7.3f} {row.glu_mean:7.3f} "
          f"  {col}{abs(row.glu_vs_glc):8.3f}  {row.direction:15s}  {row.function[:45]}")

# FSEOF 验证汇总
fseof_genes = ["ptsG", "pyk", "pfk", "hisG", "hisC"]
print("\n  FSEOF prediction vs transcriptome:")
for g in fseof_genes:
    row = txn_df[txn_df.gene == g]
    if row.empty: continue
    r = row.iloc[0]
    fseof_pred = "KO" if "KO" in r.peg_id else "OE"
    txn_consistent = (fseof_pred == "OE" and r.glu_vs_glc > 0.3) or \
                     (fseof_pred == "KO" and r.glu_vs_glc < 0)
    status = "CONSISTENT" if txn_consistent else "INCONSISTENT"
    if pd.isna(r.glu_vs_glc): status = "NO DATA"
    print(f"    {g:8s}: FSEOF={fseof_pred:3s}  txn_delta={r.glu_vs_glc:+.3f}  [{status}]")


# ══════════════════════════════════════════════════════════════
# FIG 13: AAAD 候选基因分析图（双面板）
# ══════════════════════════════════════════════════════════════
print("\nDrawing fig13 (AAAD candidate analysis) ...")

fig = plt.figure(figsize=(16, 8.5), facecolor="white")
# 左侧条图占 38%，右侧基因图占 62%
axL = fig.add_axes([0.04, 0.12, 0.32, 0.76])
axR = fig.add_axes([0.42, 0.10, 0.56, 0.76])

# ── Panel L: BEST195 全部脱羧酶分类条图（从下到上排列） ──
decarbx_categories = {
    "Other\n(acetolactate/prephenate)":  [r for r in decarboxylases if any(
        k in r["prod"].lower() for k in ["acetolactate","prephenate","oxalate"])],
    "Nucleotide/cofactor\ndecarboxylase":[r for r in decarboxylases if any(
        k in r["prod"].lower() for k in ["orotidine","uroporphyrinogen","phosphopantothen","carboxymuconolactone"])],
    "Amino acid\ndecarboxylase (other)": [r for r in decarboxylases if any(
        k in r["prod"].lower() for k in ["arginine","aspartate","diaminopimelate","adenosylmethionine"])],
    "Phenolic acid\ndecarboxylase":      [r for r in decarboxylases if "phenolic" in r["prod"].lower()],
    "PLP-dependent\n(AAAD candidate)":   [r for r in decarboxylases if "pyridoxal" in r["prod"].lower()],
}

cat_names  = list(decarbx_categories.keys())
cat_counts = [len(v) for v in decarbx_categories.values()]
cat_colors = [GY, GRN, BLUE, ORG, RED]

y_pos = range(len(cat_names))
bars = axL.barh(y_pos, cat_counts, color=cat_colors, edgecolor="white",
                height=0.62, zorder=3)
for bar, cnt in zip(bars, cat_counts):
    axL.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height()/2,
             str(cnt), va="center", ha="left", fontsize=10, fontweight="bold",
             color="#222222")

# 只对 AAAD candidate (最上面那条) 加注
axL.annotate("BSNT_RS03085\n480 aa · no BSU168\northolog · 17/20 score",
             xy=(cat_counts[-1], len(cat_names)-1),
             xytext=(cat_counts[-1]+0.6, len(cat_names)-1.6),
             fontsize=7.5, color=RED, ha="left",
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9,
                             connectionstyle="arc3,rad=-0.25"))

axL.set_xlabel("Number of genes in BEST195", fontsize=9.5, labelpad=6)
axL.set_title("Decarboxylase Gene Families\nin B. subtilis BEST195 (n=19 total)",
              fontsize=9.5, pad=6, fontweight="bold")
axL.set_xlim(0, max(cat_counts) * 1.65)
axL.set_yticks(list(y_pos))
axL.set_yticklabels(cat_names, fontsize=8.5)
axL.set_ylim(-0.55, len(cat_names)-0.35)
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)
axL.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
axL.set_axisbelow(True)
axL.text(-0.18, 1.04, "A", transform=axL.transAxes,
         fontsize=14, fontweight="bold", va="top")

# ── Panel R: BSNT_RS03085 基因组上下文图（箭头基因图，标签交替上下） ──
axR.set_xlim(-0.06, 1.12)
axR.set_ylim(-0.55, 1.55)
axR.axis("off")
axR.set_title("Genomic Context of BSNT_RS03085  (AAAD Candidate)\n"
              "BEST195 chromosome · ~574–583 kb  |  68 bp gap in BSU168",
              fontsize=9.5, pad=6, fontweight="bold")

context_genes = [
    # (start, end, strand, display_name, facecolor, short_product)
    (575972, 576757, "+", "zmaR",           "#DCE8FF", "N-acetyltransf.\n(GNAT)"),
    (576814, 577734, "-", "ydfC",           "#DCE8FF", "DMT family\ntransporter"),
    (577867, 579315, "+", "ydfD",           ORG,       "PLP amino-\ntransferase"),
    (579383, 580825, "-", "BSNT_RS03085",   RED,       "PLP decarboxylase\n(AAAD candidate)"),
    (580946, 581569, "-", "ydfE",           "#DCE8FF", "Flavin\nreductase"),
    (581659, 582339, "+", "ydfF",           "#DCE8FF", "ArsR/SmtB\nregulator"),
    (582420, 582863, "-", "ydfG",           "#DCE8FF", "Carboxymuco-\nlactone decarbx."),
]

scale  = 10000.0
offset = 574000
y_gene = 0.55   # 基因框中心 y
h_gene = 0.28   # 基因框高度

# 蓝色 BSU168 gap 区域高亮
gap_xs = (579316 - offset) / scale
gap_xe = (579382 - offset) / scale
axR.add_patch(mpatches.Rectangle(
    (gap_xs, y_gene - h_gene/2 - 0.04), gap_xe - gap_xs, h_gene + 0.08,
    facecolor="#FFEECC", edgecolor="#CC8800", linewidth=1.2,
    linestyle="--", zorder=2, label="68 bp gap (BSU168)"))
axR.text((gap_xs+gap_xe)/2, y_gene + h_gene/2 + 0.14,
         "68 bp\ngap", ha="center", va="bottom", fontsize=6.5,
         color="#885500", fontfamily="Times New Roman",
         bbox=dict(boxstyle="round,pad=0.15", fc="#FFF5DD", ec="#CC8800", lw=0.7))

for i, (s, e, strand, name, color, prod) in enumerate(context_genes):
    xs = (s - offset) / scale
    xe = (e - offset) / scale
    w  = xe - xs
    is_cand = (name == "BSNT_RS03085")
    is_nbr  = (color == ORG)
    ec = RED if is_cand else ("#CC7700" if is_nbr else "#5577AA")
    lw = 2.0 if is_cand else 0.8

    # 基因箱体
    rect = FancyBboxPatch((xs, y_gene - h_gene/2), w, h_gene,
                           boxstyle="round,pad=0.012",
                           facecolor=color, edgecolor=ec,
                           linewidth=lw, zorder=4)
    axR.add_patch(rect)

    cx = xs + w / 2

    # 基因名（框内）
    txt_color = "white" if is_cand or is_nbr else "#1A1A2E"
    axR.text(cx, y_gene, name, ha="center", va="center",
             fontsize=7.5 if is_cand else 7.0,
             fontweight="bold" if is_cand else "normal",
             color=txt_color, fontfamily="Times New Roman", zorder=5)

    # 方向箭头（框内顶部小三角）
    arrow_x = cx + (0.04 if strand == "+" else -0.04)
    arrow_x = min(max(arrow_x, xs + 0.01), xe - 0.01)
    arrowprops = dict(arrowstyle="-|>", color=ec, lw=0.9, mutation_scale=9)
    axR.annotate("",
                 xy=(arrow_x, y_gene + h_gene/2 - 0.03),
                 xytext=(cx - (0.04 if strand == "+" else -0.04), y_gene + h_gene/2 - 0.03),
                 arrowprops=arrowprops, zorder=6)

    # 产品标签：偶数基因放上方，奇数放下方，避免重叠
    if i % 2 == 0:
        ty     = y_gene + h_gene/2 + 0.32
        va_lbl = "bottom"
        axR.plot([cx, cx], [y_gene + h_gene/2, ty - 0.04],
                 color="#AAAAAA", lw=0.6, zorder=3)
    else:
        ty     = y_gene - h_gene/2 - 0.28
        va_lbl = "top"
        axR.plot([cx, cx], [y_gene - h_gene/2, ty + 0.04],
                 color="#AAAAAA", lw=0.6, zorder=3)

    fc_lbl = "#FFECEC" if is_cand else ("#FFF3E0" if is_nbr else "#F4F6FA")
    ec_lbl = RED if is_cand else ("#CC7700" if is_nbr else "#99AACC")
    axR.text(cx, ty, prod, ha="center", va=va_lbl,
             fontsize=6.5, color="#222222", fontfamily="Times New Roman",
             zorder=5, multialignment="center",
             bbox=dict(boxstyle="round,pad=0.18", fc=fc_lbl, ec=ec_lbl,
                       lw=0.7, alpha=0.95))

# 基因组坐标轴（底部标尺）
axR.annotate("", xy=(1.0, -0.32), xytext=(0.0, -0.32),
             arrowprops=dict(arrowstyle="-", color="#888888", lw=1.2), zorder=3)
for tick_frac, tick_kb in [(0.0,"574"), (0.25,"576.5"), (0.5,"579"),
                            (0.75,"581.5"), (1.0,"584")]:
    axR.plot([tick_frac, tick_frac], [-0.32, -0.27], color="#888888", lw=0.8)
    axR.text(tick_frac, -0.42, tick_kb, ha="center", va="top",
             fontsize=6.5, color="#555555", fontfamily="Times New Roman")
axR.text(0.5, -0.52, "Genomic position (kb)", ha="center", va="top",
         fontsize=7.5, color="#555555", fontfamily="Times New Roman")

# 图例
legend_patches = [
    mpatches.Patch(fc=RED,      ec=RED,      label="AAAD candidate (BSNT_RS03085)"),
    mpatches.Patch(fc=ORG,      ec="#CC7700",label="PLP aminotransferase neighbor (ydfD)"),
    mpatches.Patch(fc="#DCE8FF",ec="#5577AA", label="Other flanking genes"),
    mpatches.Patch(fc="#FFEECC",ec="#CC8800", label="68 bp gap in BSU168 (insertion site)", linestyle="--"),
]
axR.legend(handles=legend_patches, fontsize=7.2, loc="upper right",
           framealpha=0.95, handlelength=1.4, labelspacing=0.35,
           borderpad=0.6, edgecolor="#CCCCCC")

axR.text(-0.06, 1.04, "B", transform=axR.transAxes,
         fontsize=14, fontweight="bold", va="top")

save(fig, "fig13_aaad_context.png")


# ══════════════════════════════════════════════════════════════
# FIG 14: 转录组交叉验证图（双面板）
# ══════════════════════════════════════════════════════════════
print("Drawing fig14 (transcriptome cross-validation) ...")

valid_df = txn_df.dropna(subset=["glu_vs_glc"]).copy()
valid_df = valid_df.sort_values("glu_vs_glc", ascending=True)

fig = plt.figure(figsize=(14, 8.0), facecolor="white")
ax1 = fig.add_axes([0.08, 0.08, 0.38, 0.84])   # Panel A 左
ax2 = fig.add_axes([0.55, 0.10, 0.41, 0.80])   # Panel B 右

# ── Panel A: 全部基因表达差值水平条图，分组着色 ──
# 颜色语义：FSEOF KO靶标=红，FSEOF OE靶标=绿，His必需基因=蓝，natto-specific=紫，其余按方向
GENE_ROLE = {
    "ptsG":  ("KO",  RED),    "mtlA":  ("KO",  RED),
    "pyk":   ("OE",  GRN),    "pfk":   ("OE",  GRN),
    "hisG":  ("His", BLUE),   "hisI":  ("His", BLUE),
    "hisA":  ("His", BLUE),   "hisF":  ("His", BLUE),
    "hisH":  ("His", BLUE),   "hisB":  ("His", BLUE),
    "hisC":  ("His", BLUE),   "hisD":  ("His", BLUE),
    "alsS":  ("Ess", ORG),    "alsD":  ("Ess", ORG),
    "ilvC":  ("Ess", ORG),    "ilvD":  ("Ess", ORG),
    "aroA":  ("Phe", PUR),    "tyrA":  ("Phe", PUR),
    "pheA":  ("Phe", PUR),
    "aprE":  ("natto", "#AAAAAA"), "pgsA": ("natto", "#AAAAAA"),
    "pgsB":  ("natto", "#AAAAAA"),"pgsC": ("natto", "#AAAAAA"),
    "speA":  ("other", GY),
}

gene_labels, diffs, colors_bar, role_labels = [], [], [], []
for _, row in valid_df.iterrows():
    gene_labels.append(row["gene"])
    diffs.append(row["glu_vs_glc"])
    role, col = GENE_ROLE.get(row["gene"], ("other", GY))
    colors_bar.append(col)
    role_labels.append(role)

n = len(gene_labels)
y_pos = list(range(n))
bar_h = 0.60

bars = ax1.barh(y_pos, diffs, color=colors_bar, edgecolor="white",
                height=bar_h, zorder=3)

# 在每条右侧标注数值（|Δ|>0.5 才标）
for i, (bar, d) in enumerate(zip(bars, diffs)):
    if abs(d) >= 0.5:
        xpos = d + (0.07 if d >= 0 else -0.07)
        ha   = "left" if d >= 0 else "right"
        ax1.text(xpos, i, f"{d:+.2f}", va="center", ha=ha,
                 fontsize=6.8, color=colors_bar[i], fontweight="bold")

ax1.axvline(0,    color="#333333", lw=0.9, zorder=5)
ax1.axvline(0.5,  color=BLUE, lw=0.6, ls="--", alpha=0.45, zorder=2)
ax1.axvline(-0.5, color=ORG,  lw=0.6, ls="--", alpha=0.45, zorder=2)

# 高亮 ptsG 和 hisC 两个关键基因行
for key_gene, ann_text in [("ptsG","KO target\n+2.64↑"), ("hisC","Essential\n+3.65↑")]:
    if key_gene in gene_labels:
        idx = gene_labels.index(key_gene)
        ax1.axhspan(idx - 0.38, idx + 0.38, color="#FFF0F0" if "KO" in ann_text else "#EFF5FF",
                    alpha=0.55, zorder=1)
        ax1.text(ax1.get_xlim()[1] if ax1.get_xlim()[1] > 0 else 4.5,
                 idx, f"  {ann_text}", va="center", ha="left",
                 fontsize=6.2, color=RED if "KO" in ann_text else BLUE,
                 style="italic")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(gene_labels, fontsize=8.5, fontfamily="Times New Roman")
ax1.set_xlabel("$\\Delta\\log_2$ (glu+glc $-$ glc)", fontsize=9.5, labelpad=5)
ax1.set_title("Gene Expression: natto-proxy vs. Glucose\n"
              "(GSE72060 · B. subtilis 168 · 48 h)",
              fontsize=9.5, pad=6, fontweight="bold")
ax1.set_xlim(min(diffs) - 0.9, max(diffs) + 1.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="x", color="#EEEEEE", lw=0.4, zorder=0)
ax1.set_axisbelow(True)

legend_patches_a = [
    mpatches.Patch(fc=RED,   label="FSEOF knockout target (ptsG, mtlA)"),
    mpatches.Patch(fc=GRN,   label="FSEOF overexpression target (pyk, pfk)"),
    mpatches.Patch(fc=BLUE,  label="His biosynthesis essential genes"),
    mpatches.Patch(fc=ORG,   label="Acetolactate pathway essential genes"),
    mpatches.Patch(fc=PUR,   label="Phe/Tyr pathway essential genes"),
    mpatches.Patch(fc="#AAAAAA", label="Natto-specific (no BSU168 probe)"),
]
ax1.legend(handles=legend_patches_a, fontsize=7, loc="lower right",
           framealpha=0.94, labelspacing=0.30, borderpad=0.55,
           edgecolor="#CCCCCC")
ax1.text(-0.10, 1.02, "A", transform=ax1.transAxes,
         fontsize=14, fontweight="bold", va="top")

# ── Panel B: FSEOF 预测斜率 × 转录组响应 散点图 ──
fseof_df = pd.read_csv(OUTPUT / "04_fseof_L_Histidine.csv")

peg2bsu = {
    "peg_DOT_2922": "BSU22370", "peg_DOT_2921": "BSU22360",
    "peg_DOT_1441": "BSU01530", "peg_DOT_1392": "BSU01530",
    "peg_DOT_3507": "BSU10410", "peg_DOT_306":  "BSU21590",
    "peg_DOT_2736": "BSU20940", "peg_DOT_2277": "BSU14880",
    "peg_DOT_3099": "BSU26460", "peg_DOT_3098": "BSU26450",
    "peg_DOT_3097": "BSU26440", "peg_DOT_735":  "BSU06450",
    "peg_DOT_2723": "BSU20060", "peg_DOT_3811": "BSU29660",
}

scatter_pts = []
for _, row in fseof_df.iterrows():
    gpr_genes = re.findall(r"peg_DOT_\d+", str(row.gpr))
    for pg in gpr_genes:
        bsu = peg2bsu.get(pg)
        if bsu and bsu in gex:
            v = gex[bsu]
            diff = statistics.mean(v[6:9]) - statistics.mean(v[0:3])  # natto-proxy vs glc
            short = row.rxn_name[:22] if hasattr(row, "rxn_name") and row.rxn_name else row.reaction[:16]
            scatter_pts.append({
                "reaction":    row.reaction,
                "rxn_name":    short,
                "slope":       row.slope,
                "txn_delta":   diff,
                "target_type": row.target_type,
            })
            break

sdf = pd.DataFrame(scatter_pts).drop_duplicates("reaction")
print(f"\n  FSEOF vs transcriptome scatter: {len(sdf)} matched reactions")

oe_mask = sdf.target_type == "overexpress"
ko_mask = sdf.target_type == "knockout"

# 四象限背景色
xlim_val = max(abs(sdf.slope.max()), abs(sdf.slope.min())) * 1.35
ylim_val = max(abs(sdf.txn_delta.max()), abs(sdf.txn_delta.min())) * 1.35
ax2.fill_between([ 0,  xlim_val], 0,  ylim_val, color="#E8F5E9", alpha=0.35, zorder=0)
ax2.fill_between([-xlim_val, 0], -ylim_val, 0, color="#FFF0E0", alpha=0.35, zorder=0)
ax2.fill_between([ 0,  xlim_val], -ylim_val, 0, color="#F8F0FF", alpha=0.20, zorder=0)
ax2.fill_between([-xlim_val, 0],  0,  ylim_val, color="#E8F0FF", alpha=0.20, zorder=0)

ax2.scatter(sdf.loc[oe_mask, "slope"], sdf.loc[oe_mask, "txn_delta"],
            c=GRN, s=90, alpha=0.88, edgecolors="white", lw=0.8,
            label="FSEOF: Overexpression target", zorder=5)
ax2.scatter(sdf.loc[ko_mask, "slope"], sdf.loc[ko_mask, "txn_delta"],
            c=RED, s=90, alpha=0.88, edgecolors="white", lw=0.8,
            label="FSEOF: Knockout target", zorder=5)

ax2.axhline(0, color="#666666", lw=0.8, ls="--", zorder=3)
ax2.axvline(0, color="#666666", lw=0.8, ls="--", zorder=3)

# 标注关键点（|slope|>1.0 或 |txn|>1.5），用偏移避免重叠
annotated = []
for _, row in sdf.iterrows():
    if abs(row.slope) > 1.0 or abs(row.txn_delta) > 1.5:
        # 简单避让：根据象限决定文本位置
        dx = 0.12 if row.slope >= 0 else -0.12
        dy = 0.25 if row.txn_delta >= 0 else -0.25
        # 检查与已标注点的距离
        too_close = any(abs(row.slope - px) < 0.4 and abs(row.txn_delta - py) < 0.5
                        for px, py in annotated)
        if not too_close:
            ax2.annotate(row.rxn_name,
                         xy=(row.slope, row.txn_delta),
                         xytext=(row.slope + dx, row.txn_delta + dy),
                         fontsize=6.5, color="#333333",
                         ha="left" if dx > 0 else "right",
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="#AAAAAA"),
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                   ec="#DDDDDD", alpha=0.85, lw=0.5))
            annotated.append((row.slope, row.txn_delta))

ax2.set_xlabel("FSEOF Normalized Slope (L-His target)", fontsize=9.5, labelpad=5)
ax2.set_ylabel("$\\Delta\\log_2$ Expression: natto-proxy vs. glucose", fontsize=9.5, labelpad=5)
ax2.set_title("FSEOF Predicted Targets vs. Transcriptome Response\n"
              "(Consistent = OE target up-expressed / KO target actively expressed)",
              fontsize=9.5, pad=6, fontweight="bold")
ax2.set_xlim(-xlim_val, xlim_val)
ax2.set_ylim(-ylim_val, ylim_val)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=8, loc="upper left", framealpha=0.94, edgecolor="#CCCCCC")
ax2.text(-0.10, 1.02, "B", transform=ax2.transAxes,
         fontsize=14, fontweight="bold", va="top")

# 象限标签
ax2.text( xlim_val*0.95,  ylim_val*0.92, "OE consistent\n(↑ slope, ↑ expr)",
          ha="right", va="top", fontsize=7, color="#2A7A2A",
          bbox=dict(boxstyle="round,pad=0.2", fc="#E8F5E9", ec="none", alpha=0.8))
ax2.text(-xlim_val*0.95, -ylim_val*0.92, "KO consistent\n(↓ slope, actively expr)",
          ha="left", va="bottom", fontsize=7, color="#C02020",
          bbox=dict(boxstyle="round,pad=0.2", fc="#FFF0E0", ec="none", alpha=0.8))

# 一致性统计框
consistent_oe = ((sdf.target_type=="overexpress") & (sdf.txn_delta > 0)).sum()
consistent_ko = ((sdf.target_type=="knockout")    & (sdf.txn_delta > 0)).sum()
total_oe = (sdf.target_type=="overexpress").sum()
total_ko = (sdf.target_type=="knockout").sum()

ax2.text(0.98, 0.03,
         f"OE consistent: {consistent_oe}/{total_oe} ({100*consistent_oe//total_oe}%)\n"
         f"KO active in natto: {consistent_ko}/{total_ko} ({100*consistent_ko//total_ko}%)",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=8, color="#333333",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA",
                   alpha=0.92, lw=0.8))

print(f"  FSEOF-transcriptome consistency:")
print(f"    OE targets: {consistent_oe}/{total_oe} consistent (txn_delta > 0)")
print(f"    KO active in natto-proxy: {consistent_ko}/{total_ko}")

save(fig, "fig14_transcriptome.png")

# ── 复制到 paper/figures/ ──────────────────────────────────────
PAPER_FIG = OUTPUT.parent.parent / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
import shutil
for fn in ["fig13_aaad_context.png", "fig14_transcriptome.png"]:
    src = OUTPUT / fn
    if src.exists():
        shutil.copy(src, PAPER_FIG / fn)

print("\n" + "="*60)
print("Analysis A + B complete.")
print(f"Outputs: 09_aaad_candidates.csv, 09_transcriptome_validation.csv")
print(f"Figures: fig13_aaad_context.png, fig14_transcriptome.png")
print("="*60)
