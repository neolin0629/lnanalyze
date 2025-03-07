import polars as pl
from typing import Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# import qxdatac
# import qxanalyze

class DataPreProcess:
    """
    数据预处理类，用于对因子数据进行标准化、去极值和中性化处理。

    Attributes:
        data (pl.DataFrame): 输入的原始数据。
        date_col (str): 日期列的列名。
        symbol_col (str): 股票代码列的列名。
        factors_col (Optional[Union[str, List[str]]]): 因子列名或因子列名的列表。
        factors_prefix (Optional[str]): 因子列名的前缀，用于匹配多个因子列。
        neutralizing_factors (Optional[list]): 需要进行中性化处理的因子列表。
    """

    def __init__(
            self,
            data: pl.DataFrame,
            date_col: str = 'date',
            symbol_col: str = 'symbol',
            factors_col: Optional[Union[str, List[str]]] = None,
            factors_prefix: Optional[str] = None,
            neutralizing_factors: Optional[list] = None):
        """
        初始化DataPreProcess类。

        Args:
            data (pl.DataFrame): 输入的原始数据。
            date_col (str): 日期列的列名，默认为 'date'。
            symbol_col (str): 股票代码列的列名，默认为 'symbol'。
            factors_col (Optional[Union[str, List[str]]]): 因子列名或因子列名的列表。如果未提供factors_prefix，则必填。
            factors_prefix (Optional[str]): 因子列名的前缀，用于匹配多个因子列。如果未提供factors_col，则必填。
            neutralizing_factors (Optional[list]): 需要进行中性化处理的因子列表。
        
        Raises:
            ValueError: 如果factors_col和factors_prefix同时为空或同时不为空，将抛出此错误。
        """
        if factors_col is None and factors_prefix is None:
            raise ValueError("factors_col和factors_prefix不能同时为空")
        if factors_col is not None and factors_prefix is not None:
            raise ValueError("factors_col和factors_prefix不能同时非空")
        
        if isinstance(factors_col, str):
            factors_col = [factors_col]
        
        self.data = data
        if factors_prefix is not None:
            factors_col = [col for col in self.data.columns if col.startswith(factors_prefix)]
        
        self.scaler = None
        self.factor_col = factors_col

        if neutralizing_factors is not None:
            self.neutralizing_factors = neutralizing_factors
            self.data = self.data.select([date_col, symbol_col] + factors_col + neutralizing_factors)
        else:
            self.data = self.data.select([date_col, symbol_col] + factors_col)

        self.outlier = False
        self.z_score_trained = False
        self.neutralize_trained = False
        self.drop_set = False

    def outlier_MAD(self, series: pl.Series, threshold: float = 3.5) -> Tuple[pl.Series, float]:
        """
        使用MAD法去极值。

        Args:
            series (pl.Series): 需要去极值的序列。
            threshold (float): MAD法的阈值，默认为3.5。

        Returns:
            pl.Series: 去极值后的序列。
            float: 实际使用的阈值（MAD值）。
        """
        median = series.median()
        mad = (series - median).abs().median()
        adjusted_threshold = threshold * mad
        filtered_series = pl.when((series - median).abs() <= adjusted_threshold).then(series).otherwise(None)
        return filtered_series, adjusted_threshold

    def outlier_sigma(self, series: pl.Series, threshold: float = 3.0) -> Tuple[pl.Series, float]:
        """
        使用3σ法去极值。

        Args:
            series (pl.Series): 需要去极值的序列。
            threshold (float): 3σ法的阈值，默认为3.0。

        Returns:
            pl.Series: 去极值后的序列。
            float: 实际使用的标准差阈值。
        """
        mean = series.mean()
        std = series.std()
        adjusted_threshold = threshold * std
        filtered_series = pl.when((series - mean).abs() <= adjusted_threshold).then(series).otherwise(None)
        return filtered_series, adjusted_threshold

    def outlier_percentile(self, series: pl.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> Tuple[pl.Series, Tuple[float, float]]:
        """
        使用百分位法去极值。

        Args:
            series (pl.Series): 需要去极值的序列。
            lower_pct (float): 下限百分位，默认为1%。
            upper_pct (float): 上限百分位，默认为99%。

        Returns:
            pl.Series: 去极值后的序列。
            tuple: 实际使用的下限和上限百分位值。
        """
        lower_bound = series.quantile(lower_pct)
        upper_bound = series.quantile(upper_pct)
        filtered_series = pl.when((series >= lower_bound) & (series <= upper_bound)).then(series).otherwise(None)
        return filtered_series, (lower_bound, upper_bound)

    def outlier_process_train(self, mode: str = 'MAD', threshold: Optional[float] = None) -> None:
        """
        对训练数据进行去极值处理，并存储计算的阈值。

        Args:
            mode (str): 去极值方法，可以是 'MAD', 'sigma', 或 'percentile'。
            threshold (Optional[float]): 去极值的阈值。如果是百分位法，threshold应为(lower, upper)的元组。
        """
        self.outlier_thresholds = {}
        for col in self.factor_col:
            series = self.data[col]
            if mode == 'MAD':
                filtered_series, calculated_threshold = self.outlier_MAD(series, threshold if threshold is not None else 3.5)
                self.outlier_thresholds[col] = calculated_threshold
                self.data = self.data.with_columns([filtered_series.alias(col)])
            elif mode == 'sigma':
                filtered_series, calculated_threshold = self.outlier_sigma(series, threshold if threshold is not None else 3.0)
                self.outlier_thresholds[col] = calculated_threshold
                self.data = self.data.with_columns([filtered_series.alias(col)])
            elif mode == 'percentile':
                upper_percentile = threshold[1]
                lower_percentile = threshold[0]
                filtered_series, calculated_threshold = self.outlier_percentile(series, lower_percentile, upper_percentile)
                self.outlier_thresholds[col] = calculated_threshold
                self.data = self.data.with_columns([filtered_series.alias(col)])
            else:
                raise ValueError(f"未知的去极值模式: {mode}")
        self.outlier = True

    def outlier_process_fit(self, fit_data: pl.DataFrame) -> pl.DataFrame:
        """
        使用训练数据中的阈值对新数据进行去极值处理。

        Args:
            fit_data (pl.DataFrame): 需要去极值的新数据。

        Returns:
            pl.DataFrame: 去极值后的新数据。
        """
        if not self.outlier:
            raise ValueError("去极值方法尚未训练，请先调用outlier_process_train()。")

        for col in self.factor_col:
            series = fit_data[col]
            threshold = self.outlier_thresholds[col]
            if isinstance(threshold, tuple):  # 针对百分位法
                lower, upper = threshold
                fit_data = fit_data.with_columns([pl.when((series >= lower) & (series <= upper)).then(series).otherwise(None).alias(col)])
            else:  # 针对MAD或sigma法
                fit_data = fit_data.with_columns([pl.when((series - series.median()).abs() <= threshold).then(series).otherwise(None).alias(col)])

        return fit_data

    def z_score_train(self):
        """
        训练标准化器（StandardScaler），以对训练数据进行标准化处理。
        """
        self.scalers = {}
        for col in self.factor_col:
            scaler = StandardScaler()
            scaler.fit(self.data[col].to_numpy().reshape(-1, 1))
            self.scalers[col] = scaler

        self.data = self.z_score_fit(self.data)
        self.z_score_trained = True

    def z_score_fit(self, fit_data: pl.DataFrame) -> pl.DataFrame:
        """
        使用已训练的标准化器对数据进行标准化处理。

        Args:
            fit_data (pl.DataFrame): 需要进行标准化处理的数据。
        
        Returns:
            pl.DataFrame: 标准化处理后的数据。
        
        Raises:
            ValueError: 如果标准化器未被训练，调用此方法会抛出此错误。
        """
        if not self.scalers:
            raise ValueError("标准化器尚未训练，请先调用z_score_train()。")
        
        zscore_df = pl.DataFrame()

        for col in self.factor_col:
            scaled_col = self.scalers[col].transform(fit_data[col].to_numpy().reshape(-1, 1))
            zscore_df = zscore_df.with_columns(pl.Series(col, scaled_col.flatten()))

        other_cols = [pl.col(c) for c in fit_data.columns if c not in self.factor_col]
        zscore_df = zscore_df.hstack(fit_data.select(other_cols))

        return zscore_df
    
    def neutralize_train(self):
        """
        训练中性化模型，以对训练数据进行中性化处理。
        """
        self.neutralization_models = {}
        for target_col in self.factor_col:
            y = self.data[target_col].to_numpy()
            X = self.data.select(self.neutralizing_factors).to_numpy()
            
            model = LinearRegression()
            model.fit(X, y)
            self.neutralization_models[target_col] = model

        # Apply neutralization models to self.data
        neutralized_data = pl.DataFrame()

        for target_col in self.factor_col:
            y = self.data[target_col].to_numpy()
            X = self.data.select(self.neutralizing_factors).to_numpy()
            predictions = self.neutralization_models[target_col].predict(X)
            residuals = y - predictions
            neutralized_data = neutralized_data.with_columns(pl.Series(target_col + "_neutralized", residuals))

        # Combine neutralized factors with other columns
        other_cols = self.data.drop(self.factor_col)
        self.data = neutralized_data.hstack(self.data.select([pl.col(c) for c in other_cols.columns]))

        self.neutralize_trained = True

    def neutralize_fit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        使用已训练的中性化模型对测试数据进行中性化处理。

        Args:
            df (pl.DataFrame): 需要进行中性化处理的数据。
        
        Returns:
            pl.DataFrame: 中性化处理后的数据。
        
        Raises:
            ValueError: 如果中性化模型未被训练，调用此方法会抛出此错误。
        """
        if not self.neutralization_models:
            raise ValueError("中性化模型尚未训练，请先调用neutralize_train()。")
        
        neutralized_df = pl.DataFrame()

        for target_col in self.factor_col:
            if target_col not in self.neutralization_models:
                raise ValueError(f"未找到列 {target_col} 的中性化模型")
            
            y = df[target_col].to_numpy()
            X = df.select(self.neutralizing_factors).to_numpy()
            predictions = self.neutralization_models[target_col].predict(X)
            residuals = y - predictions
            neutralized_df = neutralized_df.with_columns(pl.Series(target_col + "_neutralized", residuals))

        other_cols = df.drop(self.factor_col)
        neutralized_df = neutralized_df.hstack(df.select([pl.col(c) for c in other_cols.columns]))

        return neutralized_df
    
    def null_process(self,
                     data: pl.DataFrame = None,
                     mode: str = None) -> pl.DataFrame:
        """
        处理缺失值，根据指定模式填充或删除缺失值。

        Args:
            data (pl.DataFrame, optional): 需要处理的数据。默认为None，如果不提供，将使用类中的数据。
            mode (str): 处理缺失值的模式，支持 'drop', 'median', 'mean', 'fill0', 'fill1', 'min', 'max'。
        
        Returns:
            pl.DataFrame: 处理后的数据。
        
        Raises:
            ValueError: 如果提供的mode不在支持的模式列表中，将抛出此错误。
        """
        train_flag = True
        if data is None:
            data = self.data
        else:
            train_flag = False
        
        if mode == 'drop':
            data = data.drop_nulls(subset=self.factor_col)
        
        elif mode == 'median':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(pl.col(col).median()))
        
        elif mode == 'mean':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(strategy="mean"))

        elif mode == 'fill0':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(strategy="zero"))
        
        elif mode == 'fill1':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(strategy="one"))

        elif mode == 'min':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(strategy="min"))
        
        elif mode == 'max':
            for col in self.factor_col:
                data = data.with_columns(pl.col(col).fill_null(strategy="max"))

        else:
            raise ValueError("mode必须是'drop', 'median', 'mean', 'fill0', 'fill1', 'min', 'max'之一")
        
        if train_flag:
            self.data = data
        else:
            return data