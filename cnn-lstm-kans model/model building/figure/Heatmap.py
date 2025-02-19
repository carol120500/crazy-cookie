import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_csv("3month_dataset.csv")

# 2. 去除不相关列
data_cleaned = data.drop(columns=["Date", "number", "datetime"], errors="ignore")

# 3. 处理缺失值（如果有）
data_cleaned = data_cleaned.dropna()  # 或者 data_cleaned.fillna(method="ffill")

# 4. 计算相关性矩阵
corr_matrix = data_cleaned.corr()

# 5. 删除 "waterlevel1hourago" 和 "temperature air" 这一行和这一列
corr_matrix = corr_matrix.drop(["waterlevel1hourago"], axis=0)
corr_matrix = corr_matrix.drop(["waterlevel1hourago"], axis=1)

# 6. 绘制热力图
plt.figure(figsize=(14, 12))  # 增加图形大小，确保标签不会重叠
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 10},  # 增加字体大小
    cbar_kws={"shrink": 0.8}  # 调整颜色条大小
)

# 7. 旋转 x 轴标签，防止重叠
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# 8. 添加标题
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")

# 9. 显示图像
plt.tight_layout()  # 使图形布局更加紧凑，防止标签被截断
plt.show()
