"""
utils.py — 公共工具函数
供所有模块复用：路径管理、数据加载、绘图风格、日志记录
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无头模式，Windows 不弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────
#  路径常量（所有脚本 import utils 后直接用）
# ─────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent          # natto-last/
DATA   = ROOT / "data"
CODE   = ROOT / "code"
OUTPUT = CODE  / "output"
LOG    = ROOT  / "log"

MODEL_XML = DATA / "MODEL1507180015_url.xml"
TABLE_S1  = DATA / "Table S1.xlsx"
TABLE_S2  = DATA / "Table S2.xlsx"
PROGRESS  = LOG  / "progress.md"

OUTPUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  日志配置
# ─────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def log_progress(module: str, message: str):
    """在 log/progress.md 中追加一条进度记录"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"| {ts} | {module} | {message} |\n"
    if not PROGRESS.exists():
        PROGRESS.write_text(
            "# 进度日志\n\n"
            "| 时间 | 模块 | 内容 |\n"
            "|------|------|------|\n"
        )
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line)


# ─────────────────────────────────────────────
#  数据加载
# ─────────────────────────────────────────────
def load_s1(sig_only: bool = True,
            p_thresh: float = 0.05) -> pd.DataFrame:
    """
    读取 Table S1.xlsx，返回整洁 DataFrame。
    列名统一为小写英文，方便后续引用。
    """
    df = pd.read_excel(TABLE_S1)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_")
                  for c in df.columns]
    # 统一关键列名
    rename_map = {
        "compound_name":   "name",
        "kegg_id":         "kegg",
        "class_en":        "cls",
        "mean_natto":      "mean_natto",
        "mean_soybean":    "mean_soy",
        "p_value":         "pval",
        "q_value":         "qval",
        "fold_change":     "fc",
        "log_foldchange":  "logfc",
        "vip":             "vip",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 数值型保证
    for col in ["pval", "qval", "fc", "logfc", "vip", "mean_natto", "mean_soy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["name", "pval", "fc"])

    if sig_only:
        df = df[df["pval"] < p_thresh].copy()

    # 标注方向
    df["direction"] = df["fc"].apply(
        lambda x: "natto_up" if x > 1 else "soy_up"
    )
    df["fc_abs"] = df["fc"].apply(lambda x: x if x >= 1 else 1 / x)
    df = df.sort_values("fc_abs", ascending=False).reset_index(drop=True)
    return df


def load_s2() -> pd.DataFrame:
    """读取 Table S2 Sheet1，返回 pathway → compound_list 的 DataFrame"""
    df = pd.read_excel(TABLE_S2, sheet_name="Sheet1")
    df.columns = ["pathway", "compounds"]
    df = df.dropna(subset=["pathway", "compounds"])
    # 拆分化合物列表
    df["compound_list"] = df["compounds"].apply(
        lambda x: [c.strip() for c in str(x).split(";")]
    )
    # 提取路径名（去括号内数字）
    df["pathway_name"] = df["pathway"].str.extract(r"(?:bsu|gmx)\d+\s+(.+?)\s*(?:\(\d+\))?$")
    return df


def load_model_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    解析 iBsu1103 XML，返回：
      species_df : 代谢物表（id, name, compartment）
      rxn_df     : 反应表（id, name, reversible, gene_count）
    不加载 cobra，速度快，用于数据整合阶段。
    """
    NS  = "http://www.sbml.org/sbml/level3/version1/core"
    FBC = "http://www.sbml.org/sbml/level3/version1/fbc/version2"

    tree  = ET.parse(MODEL_XML)
    root  = tree.getroot()
    model = root.find(f"{{{NS}}}model")

    # 代谢物
    species_rows = []
    for sp in model.find(f"{{{NS}}}listOfSpecies"):
        species_rows.append({
            "id":          sp.get("id", ""),
            "name":        sp.get("name", ""),
            "compartment": sp.get("compartment", ""),
            "name_short":  sp.get("name", "").split("_")[0].lower(),
        })
    species_df = pd.DataFrame(species_rows)

    # 反应
    rxn_rows = []
    for rxn in model.find(f"{{{NS}}}listOfReactions"):
        gpa = rxn.find(f"{{{FBC}}}geneProductAssociation")
        gene_count = 0
        if gpa is not None:
            gene_count = ET.tostring(gpa, encoding="unicode").count("geneProductRef")
        rxn_rows.append({
            "id":         rxn.get("id", ""),
            "name":       rxn.get("name", ""),
            "reversible": rxn.get("reversible", "true") == "true",
            "gene_count": gene_count,
        })
    rxn_df = pd.DataFrame(rxn_rows)

    return species_df, rxn_df


# ─────────────────────────────────────────────
#  节点映射核心函数
# ─────────────────────────────────────────────
def fuzzy_match_to_model(compound_name: str,
                         species_df: pd.DataFrame,
                         min_len: int = 4) -> pd.Series | None:
    """
    将 S1 化合物名称模糊匹配到 iBsu1103 代谢物。
    策略（按优先级）：
      1. 精确匹配 name_short
      2. 子串包含（compound 含 model 或 model 含 compound）
      3. 取分号前第一个名称重试
    返回命中的 species 行，未命中返回 None。
    """
    candidates = compound_name.lower().split(";")
    for cand in candidates:
        cand = cand.strip().split(",")[0].strip()
        if len(cand) < min_len:
            continue
        # 精确
        exact = species_df[species_df["name_short"] == cand]
        if not exact.empty:
            return exact.iloc[0]
        # 子串
        sub = species_df[
            species_df["name_short"].apply(
                lambda m: (cand in m or m in cand) and len(m) >= min_len
            )
        ]
        if not sub.empty:
            # 选最长匹配（最具体）
            sub = sub.copy()
            sub["match_len"] = sub["name_short"].str.len()
            return sub.sort_values("match_len", ascending=False).iloc[0]
    return None


# ─────────────────────────────────────────────
#  绘图工具
# ─────────────────────────────────────────────
PALETTE = {
    "natto_up": "#C0392B",   # deep red: natto-elevated
    "soy_up":   "#2471A3",   # deep blue: soy-elevated
    "neutral":  "#AAB7B8",   # light grey: non-significant
    "hit":      "#D4AC0D",   # amber: anchored to model
    "tca":      "#1E8449",   # forest green: TCA-related
    "aa":       "#6C3483",   # deep purple: amino acid
    "bg":       "#F8F9FA",   # figure background
    "grid":     "#E5E8E8",   # gridline colour
}

# ── Publication-quality colour sequences ──────────────────
CMAP_DIV  = "RdBu_r"          # diverging (flux heatmaps)
CMAP_SEQ  = "YlOrRd"          # sequential (risk heatmaps)

def set_style():
    """
    Global figure style: Times New Roman (serif) + SimSun (CJK fallback).
    All text rendered in English to avoid glyph issues.
    300 dpi output, white background, minimal grid.
    """
    # Reset any previous seaborn state first
    import matplotlib as mpl
    mpl.rcdefaults()

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    plt.rcParams.update({
        # ── fonts ──────────────────────────────────────────
        "font.family":          ["Times New Roman", "SimSun", "DejaVu Sans"],
        "font.serif":           ["Times New Roman", "SimSun"],
        "mathtext.fontset":     "stix",          # matches Times New Roman math
        "axes.unicode_minus":   False,

        # ── resolution ─────────────────────────────────────
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,

        # ── figure background ───────────────────────────────
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#333333",
        "axes.linewidth":       0.8,

        # ── grid ────────────────────────────────────────────
        "axes.grid":            True,
        "grid.color":           "#E5E8E8",
        "grid.linewidth":       0.5,
        "grid.alpha":           0.8,

        # ── ticks ────────────────────────────────────────────
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,

        # ── axes labels / title ─────────────────────────────
        "axes.labelsize":       10,
        "axes.titlesize":       11,
        "axes.titleweight":     "bold",
        "axes.labelweight":     "normal",
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # ── legend ───────────────────────────────────────────
        "legend.fontsize":      8.5,
        "legend.frameon":       True,
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "#CCCCCC",

        # ── lines / markers ──────────────────────────────────
        "lines.linewidth":      1.6,
        "lines.markersize":     5,

        # ── layout ───────────────────────────────────────────
        "figure.constrained_layout.use": True,
    })


def save_fig(fig: plt.Figure, filename: str):
    """Save figure to the output directory (300 dpi, tight bbox)."""
    path = OUTPUT / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  exchange 约束生成
# ─────────────────────────────────────────────
def fc_to_exchange_bounds(fc: float,
                          baseline: float = 1.0,
                          scale: float = 0.5) -> tuple[float, float]:
    """
    将 Fold Change 转化为 exchange 反应的 (lb, ub)。

    逻辑：
      - fc > 1（纳豆更高）→ 菌体净分泌 → lb = +scale * log2(fc),  ub = 1000
      - fc < 1（大豆更高）→ 菌体净摄取 → lb = -1000,  ub = -scale * log2(1/fc)
      - fc ≈ 1              → 不约束   → (-1000, 1000)
    scale 控制约束的松紧程度，默认 0.5（保守）
    """
    if fc > 1.5:
        lb = scale * np.log2(fc)
        return (round(lb, 4), 1000.0)
    elif fc < 0.67:
        ub = -scale * np.log2(1.0 / fc)
        return (-1000.0, round(ub, 4))
    else:
        return (-1000.0, 1000.0)
