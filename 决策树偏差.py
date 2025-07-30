import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text


def generate_hierarchical_data(n_samples=1000, noise_std=0.5):
    """
    生成具有层级结构的数据。
    - 特征 x1 是主要的分组依据。
    - 特征 x2 是次要的，只在 x1 > 0.5 的组内起作用。
    - 特征 x3, x4 是纯噪声，不应影响真实结果。

    Args:
        n_samples (int): 要生成的样本数量。
        noise_std (float): 加入的高斯噪声的标准差。

    Returns:
        pd.DataFrame: 包含特征、真实值和带噪声的目标值的DataFrame。
    """
    # 1. 生成四个特征，值均匀分布在 [0, 1] 区间
    X = np.random.rand(n_samples, 4)

    # 2. 初始化一个用于存储真实目标值（Ground Truth）的数组
    y_true = np.zeros(n_samples)

    # 3. 根据层级规则定义真实的目标值 y_true
    for i in range(n_samples):
        x1, x2, _, _ = X[i]

        # 第一层规则：由 x1 决定
        if x1 <= 0.5:
            y_true[i] = 10
        else:
            # 第二层规则：在 x1 > 0.5 的条件下，由 x2 决定
            if x2 <= 0.4:
                y_true[i] = 20
            else:
                y_true[i] = 30

    # 4. 在真实值的基础上加入高斯噪声，模拟现实世界的数据
    y_noisy = y_true + np.random.normal(0, noise_std, n_samples)

    # 5. 将所有数据整合到一个 Pandas DataFrame 中
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3_noise', 'x4_noise'])
    df['y_true'] = y_true  # 真实的、无噪声的值
    df['y'] = y_noisy  # 带噪声的、用于训练的值

    return df


def analyze_leaf_bias(df):
    """
    训练一个决策树模型，并分析不同深度叶子节点的预测偏差。

    Args:
        df (pd.DataFrame): 包含训练数据的DataFrame。

    Returns:
        pd.DataFrame: 附加了预测结果和分析列的DataFrame。
    """
    features = ['x1', 'x2', 'x3_noise', 'x4_noise']
    target = 'y'

    X = df[features]
    y = df[target]

    # 1. 训练一个决策树回归模型。不设置 max_depth，让树完全生长。
    #    设置 random_state 以保证结果可复现。
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X, y)

    # 2. (可选) 打印学到的决策树的文本表示，方便理解模型结构
    print("--- 学到的决策树结构 ---")
    tree_rules = export_text(tree, feature_names=features)
    print(tree_rules)

    # 3. 获取模型对每个样本的预测值
    df['y_pred'] = tree.predict(X)

    # 4. 获取每个样本数据点最终落入的叶子节点的ID
    leaf_ids = tree.apply(X)
    df['leaf_id'] = leaf_ids

    # 5. 计算树中每个节点的深度
    #    tree.tree_ 包含了树的底层结构
    node_depth = np.zeros(shape=tree.tree_.node_count, dtype=np.int64)
    is_leaves = np.zeros(shape=tree.tree_.node_count, dtype=bool)
    # 使用栈进行深度优先遍历来计算每个节点的深度
    stack = [(0, 0)]  # 初始节点 (node_id, depth)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # 检查当前节点是否为分裂节点
        is_split_node = tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]
        if is_split_node:
            # 如果是分裂节点，将其子节点压入栈中
            stack.append((tree.tree_.children_left[node_id], depth + 1))
            stack.append((tree.tree_.children_right[node_id], depth + 1))
        else:
            # 如果是叶子节点，进行标记
            is_leaves[node_id] = True

    # 6. 将每个叶子节点的深度映射回每个数据样本
    df['leaf_depth'] = node_depth[leaf_ids]

    # 7. 计算每个样本的预测误差（残差）
    #    偏差的估计可以理解为 E[y_pred] - y_true
    #    我们用 (y_pred - y_true) 作为每个数据点误差的近似
    df['error'] = df['y_pred'] - df['y_true']
    df['abs_error'] = np.abs(df['error'])

    # 8. 按叶子节点深度分组，计算误差的均值和标准差
    #    均值可作为偏差(Bias)的代理指标
    #    标准差可作为方差(Variance)的代理指标
    analysis = df.groupby('leaf_depth')['error'].agg(['mean', 'std', 'count']).rename(columns={
        'mean': 'Mean_Error (Bias Proxy)',
        'std': 'Std_Dev_Error (Variance Proxy)'
    })

    print("\n--- 按叶子节点深度分析误差 ---")
    print(analysis)

    return df


def visualize_results(df):
    """
    使用箱线图可视化不同深度叶子节点的误差分布。

    Args:
        df (pd.DataFrame): 包含分析结果的DataFrame。
    """
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- 子图1: 绝对误差 vs 叶子深度 ---
    # 箱线图可以很好地展示数据分布的五个关键点（最小值、第一四分位数、中位数、第三四分位数、最大
    sns.boxplot(x='leaf_depth', y='abs_error', data=df, ax=axes[0])
    axes[0].set_title('Absolute Prediction Error vs. Leaf Depth', fontsize=14)
    axes[0].set_xlabel('Leaf Depth', fontsize=12)
    axes[0].set_ylabel('Absolute Error (|y_pred - y_true|)', fontsize=12)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 子图2: 原始误差 vs 叶子深度 ---
    # 原始误差可以帮助我们观察偏差（误差均值是否偏离0）
    sns.boxplot(x='leaf_depth', y='error', data=df, ax=axes[1])
    axes[1].set_title('Raw Prediction Error (Bias) vs. Leaf Depth', fontsize=14)
    axes[1].set_xlabel('Leaf Depth', fontsize=12)
    axes[1].set_ylabel('Error (y_pred - y_true)', fontsize=12)
    axes[1].axhline(0, color='r', linestyle='--', label='Zero Error (No Bias)')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 生成包含2000个样本的模拟数据，噪声标准差为1.0
    simulated_data = generate_hierarchical_data(n_samples=2000, noise_std=1.0)

    # 2. 训练模型并进行分析
    results_df = analyze_leaf_bias(simulated_data)

    # 3. 将分析结果进行可视化
    visualize_results(results_df)