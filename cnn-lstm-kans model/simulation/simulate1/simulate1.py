import pandas as pd
import numpy as np
from scipy.stats import genextreme
from sklearn.linear_model import LinearRegression

# Step 1: 读取数据
data = pd.read_csv("filtered_dataset.csv")

# Step 2: 提取变量
X = data[['tide', 'wind', 'atmosphericpressure', 'temperaturaair',
          'temperaturaacqua', 'rainfall', 'radiation', 'humidity', 'waterlevel1hourago']]
y = data['waterlevel']

# 确保数据没有缺失值
data = data.dropna()


# Step 3: 定义函数：拟合 GEV 参数
def fit_gev_with_conditions(X, y):
    """
    使用回归模型拟合 GEV 的参数 μ, σ, ξ
    """
    # 回归模型：拟合 μ (位置参数)
    model_mu = LinearRegression()
    model_mu.fit(X, y)
    mu = model_mu.predict(X)

    # 回归模型：拟合 log(σ) (尺度参数，保证正值)
    model_sigma = LinearRegression()
    log_sigma = np.log(np.abs(y - mu) + 1e-6)  # 避免 log(0)
    model_sigma.fit(X, log_sigma)
    sigma = np.exp(model_sigma.predict(X))

    # 回归模型：拟合 ξ (形状参数，直接回归)
    model_xi = LinearRegression()
    residuals = (y - mu) / sigma
    xi = np.sign(residuals) * (np.abs(residuals) ** (1 / 3))  # 粗略近似
    model_xi.fit(X, xi)
    xi = model_xi.predict(X)

    return mu, sigma, xi, model_mu, model_sigma, model_xi


# 拟合 GEV 参数
mu, sigma, xi, model_mu, model_sigma, model_xi = fit_gev_with_conditions(X, y)


# Step 4: 生成新的样本数据
def generate_gev_samples(mu, sigma, xi, n_samples=1000):
    """
    基于拟合的 GEV 参数生成样本数据
    """
    samples = []
    for i in range(len(mu)):
        sample = genextreme.rvs(c=-xi[i], loc=mu[i], scale=sigma[i], size=n_samples)
        samples.append(sample)
    return np.array(samples).flatten()


# 生成新数据
new_waterlevel = generate_gev_samples(mu, sigma, xi, n_samples=1)

# Step 5: 构造新的数据集
new_data = data.copy()
new_data['generated_waterlevel'] = new_waterlevel

# Step 6: 保存新的数据集
new_data.to_csv("generated_waterlevel_data.csv", index=False)
print("新的 GEV 模拟数据已保存到 'generated_waterlevel_data.csv'")

# 提取前 1440 行
new_data_1440 = new_data.iloc[:2160]

# 保存为新数据集
new_data_1440.to_csv('generated_waterlevel_data1.csv', index=False)
print("提取的前 1440 行数据已保存到 'generated_waterlevel_data_1440.csv'")

# 提取前 1440 行
new_data_1440 = new_data.iloc[:8760]

# 保存为新数据集
new_data_1440.to_csv('generated_waterlevel_data2.csv', index=False)
print("提取的前 1440 行数据已保存到 'generated_waterlevel_data_1440.csv'")

# 计算平均 GEV 分布参数
average_mu = np.mean(mu)
average_sigma = np.mean(sigma)
average_xi = np.mean(xi)

# 输出结果到控制台
print(f"平均 GEV 分布参数：")
print(f"Location (μ): {average_mu:.4f}")
print(f"Scale (σ): {average_sigma:.4f}")
print(f"Shape (ξ): {average_xi:.4f}")