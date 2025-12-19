"""
@filename:climate.py
@author:Jason
@time:2025-12-18
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# 设置随机种子以保证结果可复现
tf.random.set_seed(42)
np.random.seed(42)
# 配置 Matplotlib 以支持中文显示 (针对 Windows 系统)
# SimHei 是黑体，Windows 自带，能很好地显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号 (解决使用了中文导致负号显示为方块的问题)
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 获取并加载数据
# ==========================================
def load_data():
    """加载本地 Jena Climate 数据集"""
    csv_path = "jena_climate_2009_2016.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"无法在当前目录下找到 {csv_path}。请确保文件已下载并放置在正确位置。")

    df = pd.read_csv(csv_path)
    return df


print("正在加载数据...")
df = load_data()

# 数据预处理：每10分钟记录一次。
# 我们可以选择使用所有特征，但为了简化演示并减少噪音，我们通常会移除 'Date Time'。
# 这里我们保留所有数值型特征用于输入。
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(f"数据总行数: {len(df)}")

# ==========================================
# 4. 数据划分 (训练集 70%, 验证集 20%, 测试集 10%)
# ==========================================
n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")

# ==========================================
# 2. 标准化 (StandardScaler)
# ==========================================
# 关键点：只能使用训练集的均值和标准差来转换验证集和测试集，防止数据泄露
scaler = StandardScaler()
scaler.fit(train_df)

train_df = scaler.transform(train_df)
val_df = scaler.transform(val_df)
test_df = scaler.transform(test_df)

# ==========================================
# 3. 构建滑动窗口数据集
# ==========================================
# 任务要求：
# - 过去 48 小时 = 48 * 6 = 288 个时间步 (input_width)
# - 预测 24 小时后 = 24 * 6 = 144 个时间步 (offset)
# - 目标变量：气温 (T (degC))，它在数据集的第 1 列 (索引为 1，假设第0列是p (mbar)等)
#   我们需要确认 'T (degC)' 的列索引。

# 查找 T (degC) 的列索引
target_col_index = list(df.columns).index("T (degC)")
print(f"目标变量 'T (degC)' 的列索引是: {target_col_index}")


def make_dataset(data, delay, input_width=288, batch_size=256):
    """
    使用 keras.utils.timeseries_dataset_from_array 构建滑动窗口

    参数:
    data: 输入的标准化后的数据矩阵
    delay: 预测未来多少个时间步 (144)
    input_width: 输入的历史时间步长度 (288)
    """
    # targets 是我们要预测的值。它是 data 中的某一列（气温），偏移了 delay 个时间步
    # 比如：如果输入是 t=0 到 t=287，我们想预测 t=287+144 那个时刻的气温
    # targets 的起始位置应该是 input_width + delay - 1

    # 截取目标列（气温）
    targets = data[:, target_col_index]

    # 构建数据集
    # sampling_rate=1 表示连续取样（每10分钟）
    # sequence_length=input_width 是输入序列长度
    # shuffle=True 用于训练集，False 用于验证/测试
    dataset = keras.utils.timeseries_dataset_from_array(
        data=data[:-delay],  # 输入数据（去掉最后无法预测的部分）
        targets=targets[delay:],  # 标签数据（从 delay 开始对应）
        sequence_length=input_width,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset


# 定义参数
INPUT_WIDTH = 288  # 48小时
OFFSET = 144  # 24小时后
BATCH_SIZE = 256

# 构建 TF Dataset
print("正在构建数据集（可能需要几秒钟）...")
# 训练集需要打乱
train_ds = keras.utils.timeseries_dataset_from_array(
    data=train_df[:-OFFSET],
    targets=train_df[OFFSET:, target_col_index],
    sequence_length=INPUT_WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 验证集和测试集不打乱
val_ds = keras.utils.timeseries_dataset_from_array(
    data=val_df[:-OFFSET],
    targets=val_df[OFFSET:, target_col_index],
    sequence_length=INPUT_WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.utils.timeseries_dataset_from_array(
    data=test_df[:-OFFSET],
    targets=test_df[OFFSET:, target_col_index],
    sequence_length=INPUT_WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# 模型构建 (使用 LSTM)
# ==========================================
# 输入形状: (Batch_Size, Time_Steps, Features)
num_features = df.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(INPUT_WIDTH, num_features)),
    # 使用 LSTM 层处理时间序列
    keras.layers.LSTM(32, return_sequences=False),
    keras.layers.Dense(1)  # 输出一个标量：预测的气温
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.MeanAbsoluteError()])

model.summary()

# ==========================================
# 模型训练
# ==========================================
# 使用 EarlyStopping 防止过拟合
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_ds,
                    epochs=10,  # 演示用10轮，实际可以更多
                    validation_data=val_ds,
                    callbacks=[early_stopping])

# ==========================================
# 5. 评估 MSE 和 MAE
# ==========================================
print("\n正在评估测试集...")
results = model.evaluate(test_ds)
mse_score = results[0]
mae_score_scaled = results[1]  # 这是标准化后的 MAE

# 为了得到真实的 MAE (单位：°C)，我们需要反标准化
# 获取训练集上 'T (degC)' 的标准差
temp_std = np.sqrt(scaler.var_[target_col_index])
temp_mean = scaler.mean_[target_col_index]

# 真实 MAE ≈ 标准化 MAE * 标准差
real_mae = mae_score_scaled * temp_std

print(f"\n======== 评估结果 ========")
print(f"测试集 MSE (标准化后): {mse_score:.4f}")
print(f"测试集 MAE (标准化后): {mae_score_scaled:.4f}")
print(f"测试集 MAE (真实单位 °C): {real_mae:.4f} °C")

# ==========================================
# 6. 绘制 真实气温 vs 预测气温
# ==========================================
# 从测试集中取一批数据进行预测和绘图
# 为了展示清晰，只取前 300 个时间步
for x_batch, y_batch in test_ds.take(1):
    predictions = model.predict(x_batch)

    # 反标准化
    y_true = y_batch.numpy() * temp_std + temp_mean
    y_pred = predictions.flatten() * temp_std + temp_mean

    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:300], label='真实气温 (Actual)', color='blue')
    plt.plot(y_pred[:300], label='预测气温 (Predicted)', color='red', linestyle='--')
    plt.title('24小时后气温预测：真实值 vs 预测值 (测试集前300个样本)')
    plt.xlabel('样本索引 (Sample Index)')
    plt.ylabel('气温 (Temperature °C)')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plt.savefig('temperature_prediction.png')
    plt.show()
    break