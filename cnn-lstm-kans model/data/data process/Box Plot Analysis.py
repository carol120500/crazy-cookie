import pandas as pd

# Load data from the CSV file
filepath = '2years.csv'
data = pd.read_csv(filepath)

# 查看所有数值型列的范围
min_values = data.min()
max_values = data.max()

# 将最小值和最大值以表格形式输出
range_df = pd.DataFrame({
    'Min Value': min_values,
    'Max Value': max_values
})

# 打印表格
print(range_df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
filepath = '2years.csv'
data = pd.read_csv(filepath)

# Extract 'Temperature Air' and 'Temperature Acqua' columns
temperature_air = data['temperaturaair']
temperature_acqua = data['temperaturaacqua']

# Create a combined DataFrame
combined_temp = pd.DataFrame({
    'Temperature Air': temperature_air,
    'Temperature Acqua': temperature_acqua
})

# Plot boxplot for both variables
plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_temp, palette="Set2")

# Add title and labels
plt.title("Boxplot of Temperature Air and Temperature Acqua")
plt.xlabel("Temperature Type")
plt.ylabel("Temperature Values")

# Display the plot
plt.show()

