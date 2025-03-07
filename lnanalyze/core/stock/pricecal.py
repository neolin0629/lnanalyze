
import polars as pl
import qxdatac as qdc
from qxdatac.research.tcalendar import TCalendar

class PriceAnalyzer:
    """
    A class for analyzing stock prices and calculating various metrics.
    """
    
    def __init__(self, tdate: str = None, start_date: str = None, end_date: str = None, symbol: str = None):
        """
        Initializes the PriceAnalyzer class.

        Args:
            tdate (str, optional): The trade date. Defaults to None.
            start_date (str, optional): The start date for data retrieval. Defaults to None.
            end_date (str, optional): The end date for data retrieval. Defaults to None.
            symbol (str, optional): The symbol for which data is retrieved. Defaults to None.
        """
        self.price_df: pl.DataFrame = qdc.AShareDaily().get(start_date=start_date, end_date=end_date, symbols=symbol, df_lib='polars')
        self.adj_df: pl.DataFrame = qdc.AShareAdjFactor().get(start_date=start_date, end_date=end_date, symbols=symbol, df_lib='polars')

    # 获取价格
    def get_price(self, symbol_series: pl.Series, date_series: pl.Series, price_col: str, windows: int = None, temporal_choice: str = None, adj: bool = True):
        """
        Retrieves the price of a given symbol for specified dates, optionally adjusting for splits and dividends.

        Args:
            symbol_series (pl.Series): A series of symbols for which the price is to be retrieved.
            date_series (pl.Series): A series of dates corresponding to the symbols.
            price_col (str): The column name that contains the price information ('open', 'close', 'high', 'low').
            windows (int, optional): The window size for any rolling calculation. Defaults to None.
            temporal_choice (str, optional): A temporal choice parameter that specifies how to handle time-related data ('pre' or 'next'). Defaults to None.
            adj (bool, optional): Whether to adjust the price for splits, dividends, etc. Defaults to True.

        Returns:
            pl.Series: A series containing the retrieved prices.
        Raises:
            ValueError: If input parameters are incorrect or incomplete.
        """
        # 检查输入参数
        if windows is not None and temporal_choice is None:
            raise ValueError("temporal_choice must be either 'pre' or 'next' when windows is not None")
        if windows is None and temporal_choice is not None:
            raise ValueError("windows must be specified as INT when temporal_choice is not None")
        if price_col not in ['open', 'close', 'high', 'low']:
            raise ValueError("price_col must be either 'open', 'close', 'high', 'low'")
        check_if_tdate = TCalendar().check_tdate(date_series)
        if not check_if_tdate.all():
            raise ValueError("Some dates are not trade date. Please use 'get_closest_tdate' convert to trade date first")
        # 处理输入数据
        input_df = pl.DataFrame({
            'symbol': symbol_series,
            'tdate': date_series
        })
        if not input_df['tdate'].dtype == pl.Date:
            input_df = input_df.with_columns(
                pl.col("tdate").str.to_date().alias("tdate")
            )

        # 检查是否需要复权, 如果需要复权则将价格调整
        if adj:
            price_df = self.price_df.join(self.adj_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
            price_df = price_df.with_columns(
                pl.col(price_col) * pl.col('adj_factor').alias(price_col)
            )

        # 如果不需要时间窗口, 直接返回当前对应价格
        if windows is None and temporal_choice is None:
            input_df = input_df.join(price_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
            return input_df[price_col]
        
        # 如果需要时间窗口, 则返回对应时间窗口的价格
        # 如果是前n天, 则shift_days为正数, 如果是后n天, 则shift_days为负数
        if temporal_choice == 'pre':
            shift_days = windows
        elif temporal_choice == 'next':
            shift_days = windows * -1
        # 获取对应时间窗口的价格
        price_df = price_df.with_columns(
                pl.col(price_col).shift(shift_days).over("symbol").alias("shifted_price")
        )
        # 返回对应时间窗口的价格
        price_df = price_df.select(['symbol', 'tdate', 'shifted_price'])
        input_df = input_df.join(price_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
        return input_df['shifted_price']

    # 获取收益率
    def get_return(self, symbol_series:pl.Series, buy_date_series:pl.Series, buy_price_col:pl.Series, sell_price_col:pl.Series, sell_date_series:pl.Series=None, windows:int=None, temporal_choice:str=None):
        """
        Calculates the return on a given symbol for specified dates or windows.

        Args:
            symbol_series (pl.Series): A series of symbols for which the return is to be calculated.
            buy_date_series (pl.Series): A series of dates corresponding to the buy prices.
            buy_price_col (pl.Series): The column name that contains the buy prices ('open', 'close', 'high', 'low').
            sell_price_col (pl.Series): The column name that contains the sell prices ('open', 'close', 'high', 'low').
            sell_date_series (pl.Series, optional): A series of dates corresponding to the sell prices. Defaults to None.
            windows (int, optional): The window size for any rolling calculation. Defaults to None.
            temporal_choice (str, optional): A temporal choice parameter that specifies how to handle time-related data ('pre' or 'next'). Defaults to None.

        Returns:
            pl.Series: A series containing the calculated returns.
        Raises:
            ValueError: If input parameters are incorrect or incomplete.
        """

        # Check if the mode is in Date Mode or Window Mode
        if (windows is not None and temporal_choice is not None) and (sell_date_series is None):
            mode = 'window'
        elif (windows is None and temporal_choice is None) and (sell_date_series is not None):
            mode = 'date'
        else:
            raise ValueError("Please specify either window or date mode BY setting either 'windows and temporal_choice' or 'sell_date_series'")
        # 检查输入参数
        if (buy_price_col not in ['open', 'close', 'high', 'low']) or (sell_price_col is not None and sell_price_col not in ['open', 'close', 'high', 'low']):
            raise ValueError("price_col must be either 'open', 'close', 'high', 'low'")
        else:
            pass
        if sell_date_series is not None:
            to_check_series = pl.concat([buy_date_series, sell_date_series])
        check_if_tdate = TCalendar().check_tdate(to_check_series)
        if not check_if_tdate.all():
            raise ValueError("Some dates are not trade date. Please use 'get_closest_tdate' convert to trade date first")

        ### Date Mode
        if mode == 'date':
            input_df = pl.DataFrame({
                'symbol': symbol_series,
                'purchase_date': buy_date_series,
                'sell_date': sell_date_series
            })
            if not input_df['purchase_date'].dtype == pl.Date:
                input_df = input_df.with_columns(
                    pl.col("purchase_date").str.to_date().alias("purchase_date"),
                    pl.col("sell_date").str.to_date().alias("sell_date")
                )

            # 复权处理
            price_df = self.price_df.join(self.adj_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
            price_df = price_df.with_columns(
                (pl.col(buy_price_col) * pl.col('adj_factor')).alias('buy_price')
            )
            price_df = price_df.with_columns(
                (pl.col(sell_price_col) * pl.col('adj_factor')).alias('sell_price')
            )

            # 合并对应价格
            # 处理买入价格
            input_df = input_df.join(price_df, left_on=['symbol', 'purchase_date'], right_on=['symbol', 'tdate'], how='left')
            input_df = input_df.select(['symbol', 'purchase_date', 'buy_price', 'sell_date'])
            # 处理卖出价格
            input_df = input_df.join(price_df, left_on=['symbol', 'sell_date'], right_on=['symbol', 'tdate'], how='left')
            input_df = input_df.select(['symbol', 'purchase_date', 'buy_price', 'sell_date', 'sell_price'])

        ### Window Mode
        if mode == 'window':
            if temporal_choice not in ['pre', 'next']:
                raise ValueError("temporal_choice must be either 'pre' or 'next' when date mode is selected")
            
            input_df = pl.DataFrame({
                'symbol': symbol_series,
                'purchase_date': buy_date_series
            })
            if not input_df['purchase_date'].dtype == pl.Date:
                input_df = input_df.with_columns(
                    pl.col("purchase_date").str.to_date().alias("purchase_date")
                )

            ## 复权处理
            price_df = self.price_df.join(self.adj_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
            price_df = price_df.with_columns(
                (pl.col(buy_price_col) * pl.col('adj_factor')).alias('buy_price')
            )
            price_df = price_df.with_columns(
                (pl.col(sell_price_col) * pl.col('adj_factor')).alias('adj_sell_price')
            )

            # 处理卖出价格
            if temporal_choice == 'pre':
                shift_days = windows
            elif temporal_choice == 'next':
                shift_days = windows * -1
            # 获取对应时间窗口的价格
            print(price_df)
            price_df = price_df.with_columns(
                pl.col('adj_sell_price').shift(shift_days).over("symbol").alias("sell_price")
            )
            # 返回对应时间窗口的价格
            input_df = input_df.join(price_df, left_on=['symbol', 'purchase_date'], right_on=['symbol', 'tdate'], how='left')
            input_df = input_df.select(['symbol', 'purchase_date', 'buy_price', 'sell_price'])

        # 计算收益率
        input_df = input_df.with_columns(
            ((pl.col('sell_price') - pl.col('buy_price')) / pl.col('buy_price')).alias('return')
        )
        return input_df['return']

    # 获取统计价格 如最大值, 最小值, 均值, 标准差, 中位数, 偏度
    def get_summary_price(self, symbol_series:pl.Series, date_series:pl.Series, price_col:str, summary_select:str, windows:int, temporal_choice:str, adj:bool=True):
        """
        Retrieves summary statistics (e.g., min, max, mean) of prices for specified symbols and dates.

        Args:
            symbol_series (pl.Series): A series of symbols for which the summary statistics are to be calculated.
            date_series (pl.Series): A series of dates corresponding to the symbols.
            price_col (str): The column name that contains the price information ('open', 'close', 'high', 'low').
            summary_select (str): The summary statistic to calculate ('min', 'max', 'mean', 'std', 'sum', 'median', 'skew').
            windows (int): The window size for any rolling calculation.
            temporal_choice (str): A temporal choice parameter that specifies how to handle time-related data ('pre' or 'next').
            adj (bool, optional): Whether to adjust the price for splits, dividends, etc. Defaults to True.

        Returns:
            pl.Series: A series containing the calculated summary statistics.
        Raises:
            ValueError: If input parameters are incorrect or incomplete.
        """
        # 检查输入参数
        if summary_select not in ['min', 'max', 'mean', 'std', 'sum', 'median', 'skew']:
            raise ValueError("summary_select must be either 'min', 'max', 'mean', 'std', 'sum', 'median', 'skew'")
        check_if_tdate = TCalendar().check_tdate(date_series)
        if not check_if_tdate.all():
            raise ValueError("Some dates are not trade date. Please use 'get_closest_tdate' convert to trade date first")
        # 数据预处理
        input_df = pl.DataFrame({
            'symbol': symbol_series,
            'tdate': date_series
        })
        if not input_df['tdate'].dtype == pl.Date:
            input_df = input_df.with_columns(
                pl.col("tdate").str.to_date().alias("tdate")
            )
        # 复权处理
        if adj:
            price_df = self.price_df.join(self.adj_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
            price_df = price_df.with_columns(
                (pl.col(price_col) * pl.col('adj_factor')).alias('select_price')
            )
        else:
            price_df = self.price_df.with_columns(
                pl.col(price_col).alias('select_price')
            )
        # 处理价格
        if temporal_choice == 'pre':
            shift_days = windows
        elif temporal_choice == 'next':
            shift_days = windows * -1
        # 获取对应时间窗口的价格
        if summary_select == 'min':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_min(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'max':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_max(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'mean':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_mean(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'std':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_std(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'sum':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_sum(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'median':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_median(shift_days).over("symbol").alias("summary_price"),
            )
        elif summary_select == 'skew':
            price_df = price_df.with_columns(
                pl.col('select_price').rolling_skew(shift_days).over("symbol").alias("summary_price"),
            )

        # 返回对应时间窗口的价格
        price_df = price_df.select(['symbol', 'tdate', 'select_price'])
        input_df = input_df.join(price_df, left_on=['symbol', 'tdate'], right_on=['symbol', 'tdate'], how='left')
        return input_df['select_price']
        
    # def get_surplus_return()
        
if __name__ == "__main__":
    pa = PriceAnalyzer(start_date="2022-01-01", end_date="2023-06-05", symbol=["000001.SZ", "002548.SZ"])
    df = pa.get_summary_price(pl.Series(["000001.SZ", "002548.SZ"]), pl.Series(["2023-06-05", "2023-02-17"]), 'close', 'median', windows=5, temporal_choice='pre')
    print(df)
