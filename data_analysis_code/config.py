"""项目路径与分析常量。"""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "original_data_from_the_competition_organizer"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "data_analysis_results"

DATASET_DIRS = {
    "dataset01": DATA_ROOT / "dataset01",
    "dataset02": DATA_ROOT / "dataset02",
    "dataset03": DATA_ROOT / "dataset03",
    "dataset04": DATA_ROOT / "dataset04",
}

# 全零时间步判定。赛方特征里的填充和缺失都是精确 0，阈值只吸收浮点噪声。
ZERO_EPS = 1e-8
# float32 保存后再与 float64 原特征比较时允许的绝对误差。
MATCH_ATOL = 1e-3

# 赛题约定：回归标签 [-3, 0) 负向，0 中性，(0, 3] 正向。
POLARITY_ZH = {
    "negative": "负向",
    "neutral": "中性",
    "positive": "正向",
}
CLASS_ID_TO_POLARITY = {0: "negative", 1: "neutral", 2: "positive"}
SPLIT_ZH = {"train": "训练集", "valid": "验证集", "test": "测试集"}

# 对齐特征里，音频/视觉第 0 步在全量训练集上恒为 0，对应文本 [CLS] 槽位。
ALIGNED_STRUCTURAL_LEADING_ZERO = ("audio", "vision")
