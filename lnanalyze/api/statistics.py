"""
Statistical functions for research assistance
@author: Neo
@date: 2024/6/20
"""

from math import isinf
import string
import numpy as np
import pandas as pd
import polars as pl
from polars import Boolean
from polars import Float64
import statsmodels.api as sm


from typing import Optional, Union, List

from qxgentools.utils.typing import ArrayLike, DataFrameLike


DEFAULT_TRADING_DAYS_PER_YEAR = 250 # Default trading days per year

def calculate_stats(series: np.ndarray):
    """Calculating statistics for a given data series: mean, standard deviation and median"""
    mean = np.mean(series, dtype=float)
    std = np.std(series, dtype=float)
    median = np.median(series)
    return mean, std, median

def extreme_mad(series: ArrayLike, n: float) -> np.ndarray:
    """
    Outlier handling using Median Absolute Deviation (MAD).
    """
    series = np.array(series)
    median, _, _ = calculate_stats(series)
    mad = np.median(np.abs(series - median))
    max_range = median + n * mad
    min_range = median - n * mad
    return np.clip(series, min_range, max_range)

def extreme_nsigma(series: ArrayLike, n: float = 3) -> np.ndarray:
    """
    Outlier handling using n standard deviations (sigma).
    """
    series = np.array(series)
    mean, std, _ = calculate_stats(series)
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)

def extreme_percentile(series: ArrayLike, min_percent: float = 0.025, max_percent: float = 0.975) -> np.ndarray:
    """
    Outlier handling based on percentiles.
    """
    series = np.array(series)
    low = np.percentile(series, min_percent * 100)
    high = np.percentile(series, max_percent * 100)
    return np.clip(series, low, high)

def zscore(series: ArrayLike) -> np.ndarray:
    """
    Standardization of a series using z-score.
    """
    series = np.array(series)
    mean, std, _ = calculate_stats(series)
    return (series - mean) / std


def _prepare_data_for_neutralization(df: DataFrameLike, factors: list, target: str) -> tuple:
    """Prepare data for regression, handling both pandas and Polars DataFrame."""
    if isinstance(df, pd.DataFrame):
        y = df[target].astype(float)
        x_list = [df[factor].astype(float) if factor != 'industry' else pd.get_dummies(df['industry']) 
                  for factor in factors if factor in df.columns]
    elif isinstance(df, pl.DataFrame):
        y = df.get_column(target).cast(pl.Float64)
        x_list = [df.select(factor).cast(pl.Float64) if factor != 'industry' else df.select('industry').to_dummies() 
                  for factor in factors if factor in df.columns]
    else:
        raise ValueError("Unsupported dataframe type")

    if not x_list:
        raise ValueError("No valid factors provided")
    
    x = x_list[0] if len(x_list) == 1 else (pd.concat(x_list, axis=1) if isinstance(df, pd.DataFrame) else pl.concat(x_list, how='horizontal'))

    return y, x

def neutralization_by_OLS(df: DataFrameLike, factors: Union[str, list] = ["mktcap", "industry"]) -> np.ndarray:
    """Perform OLS to neutralize factors and extract residuals."""
    if isinstance(factors, str):
        factors = [factors]

    y, x = _prepare_data_for_neutralization(df, factors, 'factor')
    x = sm.add_constant(x) if isinstance(x, pd.DataFrame) else x.with_columns(pl.lit(1).alias('const'))
    model = sm.OLS(y, x)
    results = model.fit()
    return results.resid

def neutralization_by_inv(df: DataFrameLike, factors: Union[str, list] = ["mktcap", "industry"]) -> np.ndarray:
    """Perform matrix inversion to neutralize factors and extract residuals."""
    if isinstance(factors, str):
        factors = [factors]

    y, x = _prepare_data_for_neutralization(df, factors, 'factor')
    x = np.column_stack([np.ones(len(x)), x]) if isinstance(x, pd.DataFrame) else np.hstack([np.ones((x.height, 1)), x.to_numpy()])
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    residuals = y - x.dot(beta)
    return residuals


def sharpe(returns: Union[DataFrameLike, ArrayLike],
    interval: int = DEFAULT_TRADING_DAYS_PER_YEAR
):
    """Compute sharpe ratio for given returns. """
    if isinstance(returns, List): returns = pd.Series(returns)

    mean_return = returns.mean()
    std_return = returns.std(ddof=1) + 1e-8  # Avoid division by zero
    return mean_return / std_return * np.sqrt(interval)


def ic(data: DataFrameLike, factor_column: str, return_column: str, method: str = 'spearman') -> float:
    """
    Calculate the Information Coefficient (IC).

    Args:
        data (DataFrameLike): DataFrame containing factor values and future returns.
        factor_column (str): Name of the column containing the factor values.
        return_column (str): Name of the column containing the future returns.
        method (str): Method for calculating the correlation coefficient, default is 'spearman', 'pearson' is also available.

    Returns:
        float: The calculated Information Coefficient (IC).
    """
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be either 'spearman' or 'pearson'")

    if isinstance(data, pd.DataFrame):
        valid_data = data[[factor_column, return_column]].dropna()
        ic_value = valid_data.corr(method=method).iloc[0, 1]
    elif isinstance(data, pl.DataFrame):
        valid_data = data.select([factor_column, return_column]).drop_nulls()
        if method == 'spearman':
            ic_value = valid_data.select(pl.corr(factor_column, return_column, method='spearman'))[0, 0]
        elif method == 'pearson':
            ic_value = valid_data.select(pl.corr(factor_column, return_column, method='pearson'))[0, 0]
    else:
        raise TypeError("Unsupported DataFrame type. Expected pandas or polars DataFrame.")

    return ic_value

def ic_batch(data: pl.DataFrame, factor_column: list[str], return_column: str, method: str = 'spearman') -> float:
    """
    Calculate the Information Coefficient (IC).

    Args:
        data (pl.Dataframe): DataFrame containing factor values and future returns.
        factor_column (str): Name of the column containing the factor values.
        return_column (str): Name of the column containing the future returns.
        method (str): Method for calculating the correlation coefficient, default is 'spearman', 'pearson' is also available.

    Returns:
        float: The calculated Information Coefficient (IC).
    """
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be either 'spearman' or 'pearson'")

    valid_data = data.select(factor_column + [return_column])
    valid_data = valid_data.to_pandas()
    factor_df = valid_data[factor_column]

    ic_value = factor_df.corrwith(valid_data[return_column], method=method).sort_values(ascending=False)
    return ic_value

def factor_stat(data: pl.DataFrame, factor_column:str, return_column:str, groups:Optional[int]=5) -> pl.DataFrame:
    """
    Calculate the factor statistics: mean, standard deviation, skewness, kurtosis, and IC.

    Args:
        data (DataFrame): DataFrame containing factor values and future returns.
        factor_column (str): Name of the column containing the factor values.
        return_column (str): Name of the column containing the future returns.
        groups (int, optional): ONLY use if factor columns are numbers. Number of quantile groups to divide the data into. Defaults to None. Default is 5.
    
    Returns:
        DataFrame: DataFrame containing the calculated statistics.
    """
    valid_data = data.select([factor_column, return_column]).drop_nulls()
    #Check if factor_column is in str or numbers
    if valid_data[factor_column].dtype in [pl.String, pl.Boolean, Boolean, string]:
        #group by factor_column and calculate agg the mean, std, skewness, kurtosis, and IC
        stat_df = valid_data.group_by(factor_column).agg([
            pl.col(return_column).mean().alias('mean'),
            pl.col(return_column).median().alias('median'),
            pl.col(return_column).std().alias('std'),
            pl.col(return_column).skew().alias('skewness'),
            pl.col(return_column).kurtosis().alias('kurtosis'),
            (pl.col(return_column).std() / pl.col(return_column).mean()).alias('CV'),
            pl.col(return_column).count().alias('count'),
        ])
    elif valid_data[factor_column].dtype in [pl.Int16, pl.Int32, pl.Int8, pl.Int64, pl.Float32, pl.Float32, int, float, Float64]:
        if isinstance(groups, int) == False:
            raise ValueError("Expected groups to be an integer.")
        valid_data = valid_data.with_columns(
            pl.col(factor_column).qcut(groups, labels=[str(i + 1) for i in range(groups)], allow_duplicates=True).alias("group")
        )
        stat_df = valid_data.group_by("group").agg([
            pl.col(factor_column).mean().alias('factor-mean'), 
            pl.col(return_column).mean().alias('mean'),
            pl.col(return_column).median().alias('median'),
            pl.col(return_column).std().alias('std'),
            pl.col(return_column).skew().alias('skewness'),
            pl.col(return_column).kurtosis().alias('kurtosis'),
            (pl.col(return_column).std() / pl.col(return_column).mean()).alias('CV'),
            pl.col(return_column).count().alias('count'),
            pl.corr(factor_column, return_column, method='spearman').alias('IC')
        ])
        #sort by group number
        stat_df = stat_df.sort("group")
    else:
        raise ValueError("Unsupported factor column type. Expected string or number. Your factor column type is: ", valid_data[factor_column].dtype)
    
    return stat_df
if __name__=='__main__':
    # ! test `extreme` & `zscore`
    # data = [10, 20, 30, 40, 50, 100]
    # print(extreme_mad(data, 3))
    # print(extreme_nsigma(data))
    # print(extreme_percentile(data))
    # print(zscore(data))

    # ! test `neutralization`
    # data_frame = pd.DataFrame({
    #     'factor': np.random.rand(100),
    #     'mktcap': np.random.rand(100) * 1000,
    #     'industry': np.random.choice(['Tech', 'Finance', 'Health'], 100)
    # })

    # print(neutralization_by_OLS(data_frame, ['mktcap', 'industry']))
    # print(neutralization_by_inv(data_frame, 'mktcap'))
    pass