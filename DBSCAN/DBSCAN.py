import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取Excel数据集
data = pd.read_csv('smiley_face_dataset.csv')

# 提取特征列
X = data[['X', 'Y']]  # 请替换为你的实际特征列名称

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用DBSCAN聚类算法
dbscan = DBSCAN(eps=0.15, min_samples=5)  # 请根据需要调整eps和min_samples参数
dbscan.fit(X_scaled)

# 将聚类结果添加到数据集
data['Cluster'] = dbscan.labels_

# 绘制聚类结果的散点图
plt.figure(figsize=(8, 6))
plt.scatter(data['X'], data['Y'], c=data['Cluster'], cmap='viridis', s=50)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('DBSCAN Clustering')
plt.colorbar()
plt.show()

# 打印每个簇的数据点数量
cluster_counts = data['Cluster'].value_counts()
print(cluster_counts)
