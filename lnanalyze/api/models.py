import lightgbm as lgb
import xgboost as xgb
import polars as pl
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def lgbm_train_fit(
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        x_col: List[str],
        y_col: str
) -> (pl.DataFrame, pl.DataFrame):
    """
    Trains a LightGBM model on train_df, fits both train_df and test_df, and returns the fitted DataFrames.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame (for fitting results only).
        x_col (List[str]): List of column names to use as features (X).
        y_col (str): Column name to use as target (y).

    Returns:
        (pl.DataFrame, pl.DataFrame): Fitted training DataFrame and testing DataFrame with predictions.
    """
    # 将 Polars DataFrame 转换为 Pandas 以便 LGBM 使用
    train_pd = train_df.to_pandas()
    test_pd = test_df.to_pandas()

    # 提取 X 和 y
    X_train = train_pd[x_col]
    y_train = train_pd[y_col]

    X_test = test_pd[x_col]  # 提取测试集 X 特征

    # LGBM 参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'boosting_type': 'gbdt',  # 传统的梯度提升
        'device': 'gpu',          # 使用 GPU
        'gpu_platform_id': 0,     # 默认 GPU 平台 ID
        'gpu_device_id': 0,       # 默认 GPU 设备 ID
        'verbose': -1
    }

    # 训练模型，使用 train_test_split 切割
    X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # 训练集与验证集
    train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 训练模型
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        # early_stopping_rounds=50,
        # verbose_eval=100
    )

    # 检查 best_iteration 是否为 0
    best_iteration = lgb_model.best_iteration if lgb_model.best_iteration > 0 else 100

    # 在整个训练数据集上重新训练
    final_train_data = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        lgb_params,
        final_train_data,
        num_boost_round=best_iteration
    )

    # 对训练集和测试集生成预测
    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    # 将预测结果加入到 Polars DataFrame 中
    fitted_train_df = train_df.with_columns([
        pl.Series('multi_factor_modeled', train_predictions)
    ])
    fitted_test_df = test_df.with_columns([
        pl.Series('multi_factor_modeled', test_predictions)
    ])

    fitted_train_df = fitted_train_df.select(['symbol', 'date', 'multi_factor_modeled'])
    fitted_test_df = fitted_test_df.select(['symbol', 'date', 'multi_factor_modeled'])

    return fitted_train_df, fitted_test_df

def xgboost_train_fit(
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        x_col: List[str],
        y_col: str
) -> (pl.DataFrame, pl.DataFrame):
    """
    Trains an XGBoost model on train_df using GPU, fits both train_df and test_df, and returns the fitted DataFrames.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame (for fitting results only).
        x_col (List[str]): List of column names to use as features (X).
        y_col (str): Column name to use as target (y).

    Returns:
        (pl.DataFrame, pl.DataFrame): Fitted training DataFrame and testing DataFrame with predictions.
    """
    # 将 Polars DataFrame 转换为 Pandas 以便 XGBoost 使用
    train_pd = train_df.to_pandas()
    test_pd = test_df.to_pandas()

    # 提取 X 和 y
    X_train = train_pd[x_col]
    y_train = train_pd[y_col]

    X_test = test_pd[x_col]  # 提取测试集 X 特征

    # 将数据转换为 DMatrix 格式（XGBoost 使用的格式）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # XGBoost 参数 (使用 GPU)
    xgb_params = {
        'objective': 'reg:squarederror',  # 回归任务
        'eval_metric': 'mae',            # 评价指标
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',        # 使用 GPU
        'gpu_id': 0,                      # 使用第一个 GPU
        # 'verbosity': 0
        # 'verbose': -1
    }

    # 使用 train_test_split 进行训练集和验证集的划分
    X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # 将分割后的数据转换为 DMatrix 格式
    dtrain_cv = xgb.DMatrix(X_train_cv, label=y_train_cv)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    evals = [(dtrain_cv, 'train'), (dval, 'valid')]
    xgb_model = xgb.train(
        xgb_params,
        dtrain_cv,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=0
    )

    # 检查最佳的 boosting 轮数
    best_iteration = xgb_model.best_iteration if xgb_model.best_iteration > 0 else 100

    # 在整个训练数据集上重新训练模型
    final_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=best_iteration
    )

    # 对训练集和测试集进行预测
    train_predictions = final_model.predict(dtrain)
    test_predictions = final_model.predict(dtest)

    # 将预测结果加入到 Polars DataFrame 中
    fitted_train_df = train_df.with_columns([
        pl.Series('multi_factor_modeled', train_predictions)
    ])
    fitted_test_df = test_df.with_columns([
        pl.Series('multi_factor_modeled', test_predictions)
    ])

    fitted_train_df = fitted_train_df.select(['symbol', 'date', 'multi_factor_modeled'])
    fitted_test_df = fitted_test_df.select(['symbol', 'date', 'multi_factor_modeled'])

    #print importance table
    importance = xgb_model.get_score(importance_type='gain')
    importance_df = pl.DataFrame({'factor': importance.keys(), 'importance': importance.values()})
    importance_df = importance_df.sort('importance', descending=True)
    print(importance_df)
    return fitted_train_df, fitted_test_df

def regression_train_fit_weight_control(
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        x_col: List[str],
        y_col: str,
        non_negative: List[str],  # 需要非负的特征列
        greater_weight: List[str],  # 需要增大权重的因子
        smaller_weight: List[str],  # 需要降低权重的因子
        greater_weight_factor: float = 1,  # 增大权重的因子
        smaller_weight_factor: float = 20000  # 减小权重的因子
) -> pl.DataFrame:
    
    # 准备数据
    X_train = train_df.select(x_col).to_numpy()
    y_train = train_df.select(y_col).to_numpy().ravel()
    X_test = test_df.select(x_col).to_numpy()

    # 增大某些因子的权重的目标函数
    def increase_weight_objective(coefs, X, y, indices, boost_factor=100):
        """
        目标函数：增大某些因子的权重
        """
        return np.sum((y - X @ coefs) ** 2) - boost_factor * np.sum(coefs[indices])

    # 减小某些因子的权重的目标函数
    def decrease_weight_objective(coefs, X, y, indices, reduce_factor=100):
        """
        目标函数：减小某些因子的权重
        """
        return np.sum((y - X @ coefs) ** 2) + reduce_factor * np.sum(coefs[indices]**2)

    # 初始系数 (设为1或其他初值)
    initial_coefs = np.ones(X_train.shape[1])

    # 获取 non_negative 因子索引
    non_negative_indices = [x_col.index(factor) for factor in non_negative]
    greater_weight_indices = [x_col.index(factor) for factor in greater_weight]
    smaller_weight_indices = [x_col.index(factor) for factor in smaller_weight]

    # 创建约束条件，确保 non_negative 列的系数非负
    constraints = [{'type': 'ineq', 'fun': lambda coefs, idx=idx: coefs[idx]} for idx in non_negative_indices]

    # 合并目标函数：增大和减小权重的部分
    def combined_objective_function(coefs, X, y):
        return (increase_weight_objective(coefs, X, y, greater_weight_indices, boost_factor=greater_weight_factor) +
                decrease_weight_objective(coefs, X, y, smaller_weight_indices, reduce_factor=smaller_weight_factor))

    # 执行优化
    result = minimize(combined_objective_function, initial_coefs, args=(X_train, y_train), constraints=constraints)

    # 使用优化后的系数进行预测
    optimized_coefs = result.x
    train_y_pred = X_train @ optimized_coefs
    test_y_pred = X_test @ optimized_coefs

    # 创建包含结果的 DataFrame
    train_result_df = train_df.select(['symbol', 'date'] + x_col + [y_col])
    train_result_df = train_result_df.with_columns(pl.Series(name='predicted', values=train_y_pred))
    test_result_df = test_df.select(['symbol', 'date'] + x_col + [y_col])
    test_result_df = test_result_df.with_columns(pl.Series(name='predicted', values=test_y_pred))

    # 打印系数表
    coef_table = pl.DataFrame({'factor': x_col, 'coef': optimized_coefs})
    print(coef_table)
    
    return train_result_df, test_result_df
