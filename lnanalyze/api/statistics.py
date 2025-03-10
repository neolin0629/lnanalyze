"""
Statistical functions for research assistance
@author: Neo
@date: 2024/6/20
"""

from typing import Optional, Union, List

import numpy as np
import pandas as pd
import polars as pl

import statsmodels.api as sm

# ArrayLike = Union[list, np.ndarray, pd.Series, pl.Series]
from lntools.utils.typing import ArrayLike, DataFrameLike

# test git

DEFAULT_TRADING_DAYS_PER_YEAR = 250  # Default trading days per year


def _arraylike_to_array(series: ArrayLike) -> np.ndarray:
    # Convert input to numpy array for consistent handling
    if isinstance(series, (list, np.ndarray)):
        series = np.array(series, dtype=float)
    elif isinstance(series, pd.Series):
        series = series.to_numpy(dtype=float)
    elif isinstance(series, pl.Series):
        series = series.to_numpy()
    else:
        raise TypeError(
            f"Unsupported type: {type(series)}. Expected list, numpy array, pandas Series, or polars Series"
        )

    return series


def calculate_stats(series: ArrayLike) -> tuple:
    """Calculating statistics for a given data series: mean, standard deviation and median"""
    # Convert input to numpy array for consistent handling
    series = _arraylike_to_array(series)

    # Handle empty arrays
    if len(series) == 0:
        return float('nan'), float('nan'), float('nan')
    mean = np.mean(series, dtype=float)
    std = np.std(series, dtype=float)
    median = np.median(series)
    return mean, std, median


def extreme_mad(series: ArrayLike, n: float) -> np.ndarray:
    """
    Outlier handling using Median Absolute Deviation (MAD).
    """
    series = _arraylike_to_array(series)

    median, _, _ = calculate_stats(series)
    mad = np.median(np.abs(series - median))
    max_range = median + n * mad
    min_range = median - n * mad
    return np.clip(series, min_range, max_range)


def extreme_nsigma(series: ArrayLike, n: float = 3) -> np.ndarray:
    """
    Outlier handling using n standard deviations (sigma).
    """
    series = _arraylike_to_array(series)

    mean, std, _ = calculate_stats(series)
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)


def extreme_percentile(
    series: ArrayLike,
    min_percent: float = 0.025,
    max_percent: float = 0.975
) -> np.ndarray:
    """
    Outlier handling based on percentiles.
    """
    series = _arraylike_to_array(series)
    low = np.percentile(series, min_percent * 100)
    high = np.percentile(series, max_percent * 100)
    return np.clip(series, low, high)


def zscore(series: ArrayLike) -> np.ndarray:
    """
    Standardization of a series using z-score.
    """
    series = _arraylike_to_array(series)
    mean, std, _ = calculate_stats(series)
    return (series - mean) / std


def _prepare_data_for_neutralization(df: DataFrameLike, factors: list, target: str) -> tuple:
    """
    Prepares data for regression-based factor neutralization by extracting target variable and factor data.

    Parameters
    ----------
    df : DataFrameLike
        Input dataframe (pandas or Polars) containing target and factor columns
    factors : list
        List of column names to use as factors for neutralization
    target : str
        Column name of the target variable

    Returns
    -------
    tuple
        (y, x) where y is the target variable and x contains predictor variables
    """
    # Filter only valid factors present in the dataframe
    valid_factors = [f for f in factors if f in df.columns]

    if not valid_factors:
        raise ValueError("No valid factors provided")

    if isinstance(df, pd.DataFrame):
        y = df[target].astype(float)

        # Handle factors more efficiently
        x_frames = []
        for factor in valid_factors:
            if factor == 'industry':
                x_frames.append(pd.get_dummies(df['industry']))
            else:
                x_frames.append(df[[factor]].astype(float))

        x = pd.concat(x_frames, axis=1) if len(x_frames) > 1 else x_frames[0]

    elif isinstance(df, pl.DataFrame):
        y = df.get_column(target).cast(pl.Float64)

        # Handle factors more efficiently
        x_frames = []
        for factor in valid_factors:
            if factor == 'industry':
                x_frames.append(df.select('industry').to_dummies())
            else:
                x_frames.append(df.select(pl.col(factor).cast(pl.Float64)))

        x = pl.concat(x_frames, how='horizontal') if len(x_frames) > 1 else x_frames[0]

        y = y.to_pandas()  # Convert to pandas Series, Excessive data volume can affect performance
        x = x.to_pandas()  # Convert to pandas DataFrame

    else:
        raise TypeError(f"Unsupported dataframe type: {type(df)}. Expected pandas DataFrame or polars DataFrame.")

    return y, x


def neutralization_by_ols(
    df: DataFrameLike,
    factors: Optional[Union[str, list]] = None
) -> np.ndarray:
    """
    Perform OLS regression to neutralize factors and extract residuals.

    Parameters
    ----------
    df : DataFrameLike
        Input dataframe containing 'factor' column and factor columns
    factors : Union[str, list], default=["mktcap", "industry"]
        Factor or list of factors to neutralize against

    Returns
    -------
    np.ndarray
        Residuals after neutralization
    """
    if factors is None:
        factors = ["mktcap", "industry"]
    elif isinstance(factors, str):
        factors = [factors]

    # 'factor' is y, neutralize x
    y, x = _prepare_data_for_neutralization(df, factors, 'factor')

    # Add constant term based on dataframe type
    x = sm.add_constant(x)

    # Fit OLS model and return residuals
    model = sm.OLS(y, x)
    return model.fit().resid


def neutralization_by_inv(
    df: DataFrameLike,
    factors: Optional[Union[str, list]] = None
) -> np.ndarray:
    """
    Perform matrix inversion to neutralize factors and extract residuals.

    Parameters
    ----------
    df : DataFrameLike
        Input dataframe containing 'factor' column and factor columns
    factors : Union[str, list], default=["mktcap", "industry"]
        Factor or list of factors to neutralize against

    Returns
    -------
    np.ndarray
        Residuals after neutralization
    """
    if factors is None:
        factors = ["mktcap", "industry"]
    elif isinstance(factors, str):
        factors = [factors]

    y, x = _prepare_data_for_neutralization(df, factors, 'factor')

    # Convert to numpy and add constant column
    if isinstance(x, pd.DataFrame):
        x_array = np.column_stack([np.ones(len(x)), x.to_numpy()])
        y_array = y.to_numpy()
    else:  # polars DataFrame
        x_array = np.column_stack([np.ones(x.height), x.to_numpy()])
        y_array = y.to_numpy()

    # Compute beta using matrix inversion and return residuals
    # Using more numerically stable approach with np.linalg.lstsq
    beta, _, _, _ = np.linalg.lstsq(x_array, y_array, rcond=None)
    return y_array - x_array @ beta


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