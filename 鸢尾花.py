import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score

# --- 1. 加载并准备数据 ---

# 加载 scikit-learn 内置的鸢尾花数据集
iris = load_iris()

# 为了方便后续处理，我们将其转换为 Pandas DataFrame
# iris.data 是特征数据，iris.feature_names 是特征名称
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris.target 是目标类别（0, 1, 2），iris.target_names 是类别名称
df['species'] = iris.target
df['species_name'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

# **根据要求，我们只选择两个自变量：花瓣长度和花瓣宽度**
features_to_use = ['petal length (cm)', 'petal width (cm)']
X = df[features_to_use]
y = df['species']

print("--- 数据预览 (前5行) ---")
print(df.head())
print(f"\n选择的自变量: {features_to_use}")


# --- 2. 训练分类树模型 ---

# 初始化决策树分类器
# random_state=42 确保每次运行代码时，树的构建方式都一样，便于复现结果
tree_classifier = DecisionTreeClassifier(random_state=42)

# 使用选择的两个特征和目标类别来训练模型
tree_classifier.fit(X, y)

print("\n--- 模型训练完成 ---")


# --- 3. 分析与评估模型 ---

# a. 评估模型准确率
y_pred = tree_classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"模型在训练集上的准确率: {accuracy:.4f}")

# b. 查看树的文本结构
# 这可以清晰地展示出模型的决策规则
tree_rules = export_text(tree_classifier, feature_names=features_to_use)
print("\n--- 决策树规则 ---")
print(tree_rules)


# --- 4. 可视化分析 ---

# a. 可视化决策树的结构图
plt.figure(figsize=(20, 10))
plot_tree(tree_classifier,
          feature_names=features_to_use,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree Structure for Iris Classification", fontsize=16)
plt.show()


# b. 可视化决策边界
# 这是一个非常直观的图，展示了模型是如何在二维平面上划分不同类别的
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 7))

    # 设置绘图风格
    sns.set_style("whitegrid")

    # 绘制原始数据散点图
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='viridis', s=60, edgecolor='k')

    # 创建一个网格来覆盖整个特征空间
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 使用模型预测网格中每个点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界（背景填充色）
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # 设置图表标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(X.columns[0], fontsize=12)
    plt.ylabel(X.columns[1], fontsize=12)
    plt.legend(title='Species', labels=iris.target_names)
    plt.show()

# 调用函数绘制决策边界图
plot_decision_boundary(X, df['species_name'], tree_classifier, "Decision Boundary of Classification Tree on Iris Data")