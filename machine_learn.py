"""
@filename:machine_learn.py
@author:Jason
@time:2025-12-18
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 算法导入
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor


def load_local_data(file_path):
    """
    从本地读取数据集并进行初步清洗
    :param file_path: 本地 CSV 文件的路径
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}，请检查路径是否正确。")

    print(f"正在从本地读取数据: {file_path}...")
    df = pd.read_csv(file_path)

    # 自动识别目标变量列名（常见为 SalePrice）
    target_col = 'SalePrice'
    if target_col not in df.columns:
        # 如果列名大小写不同，尝试匹配
        cols = {c.lower(): c for c in df.columns}
        if 'saleprice' in cols:
            target_col = cols['saleprice']
        else:
            raise KeyError(f"数据集中未找到目标列 '{target_col}'")

    # 剔除面积过大但价格过低的极端异常值（数据清洗常规操作）
    if 'GrLivArea' in df.columns:
        df = df.drop(df[(df['GrLivArea'] > 4000) & (df[target_col] < 300000)].index)

    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # 对目标变量取对数，使其符合正态分布，提升模型精度
    y_log = np.log1p(y)

    return X, y_log


def build_pipeline(model):
    """
    构建处理流水线：自动处理数值型和类别型特征
    """
    # 识别特征类型
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # 数值预处理：填充中位数 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别预处理：填充缺失值 + 独热编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])


# --- 主程序 ---

# 1. 设置你的本地文件路径（请根据实际情况修改）
# 例如: "C:/Users/Desktop/AmesHousing.csv" 或 "./data/train.csv"
DATA_PATH = "AmesHousing.csv"

try:
    X, y = load_local_data(DATA_PATH)

    # 2. 划分数据集 (3:1 -> 训练集 75%, 测试集 25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 3. 定义模型（包含之前优化的 MLP 参数）
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR (Optimized)": SVR(kernel='rbf', C=20, epsilon=0.01),
        "MLP (Optimized)": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=5000,  # 修改点：从 2000 增加到 5000 以解决 ConvergenceWarning
            solver='lbfgs',
            alpha=0.05,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    print("\n开始模型训练与评估...")
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))  # 还原对数显示真实金额误差

        results[name] = {"R2": r2, "MAE": mae, "Pred": y_pred}
        print(f"{name:18} -> R2 Score: {r2:.4f}, MAE: ${mae:,.2f}")

    # 4. 可视化结果
    plt.figure(figsize=(14, 6))

    # 子图1：各模型 R2 得分对比
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in model_names]
    colors = sns.color_palette("viridis", len(model_names))

    # 修改点：修复 Seaborn FutureWarning，添加 hue=model_names 和 legend=False
    sns.barplot(x=model_names, y=r2_scores, hue=model_names, legend=False, palette=colors)

    plt.title("Comparison of Model R2 Scores")
    plt.ylabel("R2 Score")
    plt.ylim(0.7, 1.0)  # 聚焦高分区域
    plt.xticks(rotation=15)

    # 子图2：预测值 vs 实际值 (以最佳模型为例)
    plt.subplot(1, 2, 2)
    best_model = max(results, key=lambda k: results[k]["R2"])
    plt.scatter(y_test, results[best_model]["Pred"], alpha=0.5, edgecolors='w')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.xlabel("Actual Price (Log)")
    plt.ylabel("Predicted Price (Log)")
    plt.title(f"Best Model: {best_model} Residual Plot")

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"运行出错: {e}")