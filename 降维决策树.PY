import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa  3D 曲面需要
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 1. 生成数据
def generate_data(n_samples=200, n_features=10, noise_std=1.0, random_state=42):
    rng = np.random.RandomState(random_state)      # 注意要绑定 rng
    X = rng.rand(n_samples, n_features)
    y_true = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    epsilon = rng.normal(0, noise_std, n_samples)
    y = y_true + epsilon
    feature_names = [f'X{i + 1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y

# 2. PCA 降维
def apply_pca(X_original, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cols = [f'PC{i+1}' for i in range(n_components)]
    return pd.DataFrame(X_pca, columns=cols), pca

# 3. 建模并分析
def fit_and_analysis(X_df, y, title, max_depth=None, draw_3d=False):
    print(f"\n--- {title} ---")
    # 划分训练 / 测试数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42)

    # 训练
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)

    # 评估
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"训练 MSE: {mse_train:.4f}")
    print(f"测试 MSE: {mse_test:.4f}")
    print(f"测试 R²:  {r2_test:.4f}")
    print(f"树深度: {tree.get_depth()}")

    # 打印树结构（前 3 层）
    try:
        tree_rules = export_text(tree, feature_names=list(X_df.columns), max_depth=3)
        print("树结构 (前3层):\n", tree_rules)
    except Exception as e:
        print("无法打印树结构:", e)

    # ------------ 可视化 ------------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_train, y_pred_train, alpha=0.7, label='Train')
    plt.scatter(y_test, y_pred_test, alpha=0.7, label='Test')
    lims = [min(plt.xlim()[0], plt.ylim()[0]),
            max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(f'{title}\nTrue vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 如果是 2D-PCA，额外画 3D 曲面
    if draw_3d and X_df.shape[1] == 2:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        # 网格
        x1_range = np.linspace(X_df.iloc[:, 0].min(), X_df.iloc[:, 0].max(), 50)
        x2_range = np.linspace(X_df.iloc[:, 1].min(), X_df.iloc[:, 1].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        z_grid = tree.predict(grid).reshape(x1_grid.shape)
        # 画曲面
        ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.7)
        # 画原始点
        ax.scatter(X_df.iloc[:, 0], X_df.iloc[:, 1], y,
                   c=y, cmap='viridis', edgecolor='k', s=15)
        ax.set_xlabel(X_df.columns[0])
        ax.set_ylabel(X_df.columns[1])
        ax.set_zlabel('y')
        ax.set_title(f'{title} - 预测曲面')
        plt.tight_layout()
        plt.show()

# 4. 主流程
if __name__ == '__main__':
    N_SAMPLES, N_FEATURES, NOISE_STD, RANDOM_STATE = 200, 10, 1.0, 42

    # 原始数据
    X_original, y_original = generate_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise_std=NOISE_STD,
        random_state=RANDOM_STATE
    )

    # 原始模型
    fit_and_analysis(X_original, y_original, "原始模型")

    # PCA 降维
    X_pca_df, _ = apply_pca(X_original, n_components=2)

    # 降维模型
    fit_and_analysis(X_pca_df, y_original, "PCA 降维后模型", draw_3d=True)