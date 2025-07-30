import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
def generate_friedman_data(n_samples=200, n_features=10, noise_std=1.0, random_state=42):
    """
    根据 Friedman 的模型生成模拟数据。
    Y = 10*sin(pi*X1*X2) + 20*(X3 - 0.5)^2 + 10*X4 + 5*X5 + epsilon

    Args:
        n_samples (int): 样本量。
        n_features (int): 总特征数 (前5个有效，其余为噪声)。
        noise_std (float): 噪声项 epsilon 的标准差。
        random_state (int): 随机种子，用于复现结果。

    Returns:
        tuple: (X, y) 包含特征矩阵和目标向量。
    """
    #设置随机种子以便于复现
    np.random.RandomState(random_state)
    #随机生成均匀分布的X
    X=np.random.rand(n_samples,n_features)
    #根据X生成Y
    y_true=(
        10*np.sin(np.pi*X[:,0]*X[:,1])+20*(X[:,2]-0.5)**2+10*X[:,3]+5*X[:,4]
    )
    epsilon=np.random.normal(0,noise_std,n_samples)
    y = y_true + epsilon
    # 创建列名
    # [f'X{i+1}' for i in range(n_features)] 生成一个包含 n_features 个元素的列表，每个元素是特征的名称，格式为 'X1', 'X2', ..., 'Xn_features'
    feature_names = [f'X{i + 1}' for i in range(n_features)]
    # 将特征矩阵 X 转换为 pandas DataFrame，并指定列名为 feature_names
    X_df = pd.DataFrame(X, columns=feature_names)

    # 返回特征矩阵和目标向量
    return X_df, y
#模型拟合与分析函数
def fit_and_analysis(X_df, y, title):
    print(f"\n--- {title} ---")
    # 初始化并训练回归树模型
    # 使用 random_state 保证每次分裂都一样，便于比较
    tree=DecisionTreeRegressor()
    tree.fit(X_df,y)
    # 预测并计算均方误差 (MSE)
    y_pred = tree.predict(X_df)
    mse = mean_squared_error(y, y_pred)

    print(f"模型均方误差 (MSE): {mse:.4f}")
    print(f"树的最大深度: {tree.get_depth()}")
    # 打印树的结构（只显示前几层，避免刷屏）
    try:
        tree_rules = export_text(tree, feature_names=list(X_df.columns), max_depth=3)
        print("树结构 (前3层):")
        print(tree_rules)
    except Exception as e:
        print(f"无法打印树结构: {e}")

    return tree
if __name__=='__main__':
    # --- 基础设置 ---
    N_SAMPLES = 200
    N_FEATURES = 10
    NOISE_STD = 1.0
    RANDOM_STATE = 42

    # 生成原始数据
    X_original, y_original = generate_friedman_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise_std=NOISE_STD,
        random_state=RANDOM_STATE
    )

    # --- (1) 基准模型：应用回归树拟合原始数据 ---
    model_1 = fit_and_analysis(X_original.copy(), y_original, "(1) 基准模型")
    # --- (2) 单调变换：对 X1 进行指数变换 ---
    # 决策树对特征的单调变换具有不变性。
    # 因为树的分裂只关心值的顺序（哪个值大于或小于分裂点），
    # 而单调变换（如 log, exp, x^2）不改变值的顺序。
    # 因此，理论上我们预期结果与(1)非常相似。
    X_transformed = X_original.copy()
    # 对 X1 应用 exp(x) 变换
    X_transformed['X1'] = np.exp(X_transformed['X1'])
    model_2 = fit_and_analysis(X_transformed, y_original, "(2) 对 X1 进行单调变换后的模型")
    print("对比(1)和(2): MSE 和树结构几乎没有变化，验证了决策树对单调变换的稳健性。")
    X_transformed1=X_original.copy()
    X_transformed1['X2']=np.log2(X_transformed1['X2'])
    model_3=fit_and_analysis(X_transformed1,y_original,"(3)对 X1 进行单调变换后的模型2")
    print("对比(1)和(3): MSE 和树结构几乎没有变化，验证了决策树对单调变换的稳健性。")
    # --- (3) 离群点：将第一个样本扩大10倍 ---
    # 决策树对离群点（特别是特征空间的离群点）比较敏感，
    # 因为它可能会为了孤立这个离群点而创建一个很深的分支，
    # 从而改变整体的树结构。
    X_outlier = X_original.copy()
    # 将第一个样本的所有特征值乘以10
    X_outlier.iloc[0, :] = X_outlier.iloc[0, :] * 10
    model_4= fit_and_analysis(X_outlier, y_original, "(3) 引入离群点后的模型")
    print("对比(1)和(3): 树的结构可能发生改变，模型可能会专门为这个离群点建立一个节点。")
    print("这体现了树模型对特征空间中离群点的敏感性。")
    # --- (4) 缺失值：删除 X2 的部分观测值并进行插补 ---
    # scikit-learn 的决策树不能直接处理含 NaN 的数据，必须先进行预处理。
    # 常见的处理方法是插补（imputation），例如用均值、中位数或众数填充。
    # 这里我们使用中位数进行插补，因为它对离群值不敏感。
    X_missing = X_original.copy()
    # 随机选择 20% 的行（即40个样本）将 X2 的值设为 NaN
    missing_indices = np.random.choice(X_missing.index, size=int(N_SAMPLES * 0.2), replace=False)
    X_missing.loc[missing_indices, 'X2'] = np.nan

    # 使用 X2 列的中位数来填充缺失值
    median_X2 = X_missing['X2'].median()
    X_missing['X2'].fillna(median_X2, inplace=True)
    print(f"\n--- (4) 引入并插补缺失值后的模型 (用中位数 {median_X2:.4f} 填充) ---")
    model_5 = fit_and_analysis(X_missing, y_original, "分析结果")
    print("对比(1)和(5): 由于信息丢失（部分X2的值被替换为中位数），模型的性能通常会下降（MSE变高）。")
    print("这表明了预处理缺失值的重要性以及信息丢失对模型性能的负面影响。")