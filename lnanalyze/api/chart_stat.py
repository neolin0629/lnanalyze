"""
Chart Analysis for Trade Date Return Calculation.

This module defines the ChartAnalysis class for performing chart analysis
on trade date returns.

Author: Johnny
Date: 2024/6/24
"""
import polars as pl
import qxdatac
from typing import Optional, Union, List
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from qxanalyze.api.tcalendar import shift
import numpy as np

class ChartAnalysis:
    """
    A class used to perform chart analysis for trade date return calculation.

    Attributes:
    ---------
        symbol (List[str]): List of unique stock symbols.
        stock_series (pl.Series): Series containing stock symbols.
        tdate_series (pl.Series): Series containing trade dates.
        price_col (str): Column name for stock prices.
        fdays (int): Number of forward days.
        pdays (int): Number of previous days.
        min_date (datetime): Minimum date for the analysis.
        max_date (datetime): Maximum date for the analysis.

    Methods:
    -------
        __init__(self, tdate_series: pl.Series, stock_series: pl.Series, price_col: str, fdays: int, pdays: int):
            Initializes the ChartAnalysis class with trade dates, stock symbols, price column, forward days, and previous days.

        get_cumulative_return_chart(self, output_type: Optional[str] = 'chart') -> Union[plt.Figure, pl.DataFrame]:
            Generates a cumulative return chart or DataFrame based on the specified output type.

        get_discrete_return_chart(self, output_type: Optional[str] = 'chart') -> Union[plt.Figure, pl.DataFrame]:
            Placeholder for generating a discrete return chart or DataFrame.
    """

    def __init__(
        self,
        tdate_series: pl.Series,
        stock_series: pl.Series,
        price_col: str,
        fdays: int,
        pdays: int,
        group_series: pl.Series = None,
    ):
        """
        Initializes the ChartAnalysis class.

        Args:
        -----

            tdate_series (pl.Series): Series containing trade dates.
            stock_series (pl.Series): Series containing stock symbols.
            group_series (pl.Series): Series containing group symbols.
            price_col (str): Column name for stock prices.
            fdays (int): Number of forward days.
            pdays (int): Number of previous days.
        """
        self.symbol = stock_series.unique().to_list()
        self.stock_series = stock_series
        self.tdate_series = tdate_series
        self.group_series = group_series
        self.price_col = price_col
        self.fdays = fdays
        self.pdays = pdays
        self.min_date = tdate_series.min() - datetime.timedelta(days=pdays+1)
        self.max_date = tdate_series.max() + datetime.timedelta(days=fdays+1)

    def _agg_return(
            self,
            data: pl.DataFrame,
            group_col: Union[str, List[str]],
            agg_col: str,
            stat_method: str = 'mean',
            agg_col_name: str = 'return',
    ) -> pl.DataFrame:
        '''
        Calculate the aggregated return based on the specified method.

        Args:
        ------
            data (pl.DataFrame): DataFrame containing the data.
            group_col (Union[str, List[str]]): Column name or list of column names to group by.
            agg_col (str): Column name to aggregate.
            stat_method (str): Statistic method, either 'mean' or 'median'. Default is 'mean'.
            agg_col_name (str): Name of the aggregated column. Default is 'return'.
        
        Returns:
        --------
            pl.DataFrame: DataFrame containing the aggregated return.

        '''
        if stat_method == 'mean':
            agg_df = data.group_by(group_col).agg(pl.col(agg_col).mean().alias(agg_col_name))
        elif stat_method == 'median':
            agg_df = data.group_by(group_col).agg(pl.col(agg_col).median().alias(agg_col_name))
        else:
            raise ValueError(f"Invalid stat method: {stat_method}, select from 'mean' or 'median'")
        return agg_df

    def get_cumulative_return_chart(
        self,
        stat_method: str = 'mean',
        benchmark: str = None,
        output_type: Optional[str] = 'chart'
    ) -> Union[plt.Figure, pl.DataFrame]:
        """
        Generates a cumulative return chart or DataFrame.

        Args:
        -------
            stat_method (str): Statistic method, either ```mean``` or ```median```. Default is ```mean```.
            benchmark (str): The benchmark symbol.
            output_type (Optional[str]): The output type, either ```chart``` or ```df```. Default is ```chart```.

        Returns:
        -------
            Union[plt.Figure, pl.DataFrame]: The cumulative return chart or DataFrame.
        """
        # Check if the output type is valid
        if output_type not in ['chart', 'df']:
            raise ValueError(f"Invalid output type: {output_type}")

        if self.group_series is not None:
            return self.get_cumulative_return_chart_with_group(benchmark=benchmark, output_type=output_type)
        
        # Create input DataFrame
        input_df = pl.DataFrame({'tdate': self.tdate_series, 'symbol': self.stock_series})
        
        # Get daily data
        daily_df = qxdatac.AShareDaily(columns=['symbol', 'tdate', self.price_col]).get(symbols=self.symbol, sdt=self.min_date, edt=self.max_date)
        daily_df = daily_df.rename({self.price_col: 'price'})
        if benchmark is not None:
            if self.price_col.endswith('_adj'):
                index_price_col = self.price_col.replace('_adj', '')
            else:
                index_price_col = self.price_col
            index_df = qxdatac.AShareIndexDaily(columns=['tdate', index_price_col]).get(sdt=self.min_date, edt=self.max_date, idxids=[benchmark])
            index_df = index_df.rename({index_price_col: 'index_price'})
            daily_df = daily_df.join(index_df, on='tdate', how='left')

        # Add row numbers to daily_df
        daily_df = daily_df.sort(['symbol','tdate'])
        daily_df = daily_df.with_row_count("row_num")

        def get_surrounding_rows(df: pl.DataFrame, input_row) -> pl.DataFrame:
            """
            Gets rows surrounding the target date within the specified range.

            Args:
            ---------
                df (pl.DataFrame): DataFrame containing daily data.
                input_row (dict): Row from input_df.

            Returns:
            ---------
                pl.DataFrame: DataFrame containing surrounding rows with additional columns.
            """
            symbol = input_row["symbol"]
            tdate = input_row["tdate"]
            
            # Get the row number for the target date
            tdate_row = df.filter((pl.col("symbol") == symbol) & (pl.col("tdate") == tdate))
            
            if tdate_row.shape[0] == 0:
                return pl.DataFrame()
            
            row_num = tdate_row.select("row_num")[0, 0]
            
            # Get rows within the specified range
            surrounding_rows = df.filter(
                (pl.col("symbol") == symbol) & 
                (pl.col("row_num") >= row_num - self.pdays) & 
                (pl.col("row_num") <= row_num + self.fdays)
            )
            # Add index_col and day1_price columns
            surrounding_rows = surrounding_rows.with_row_count("index_col")
            day1_price = surrounding_rows.sort("index_col").select("price")[0,0]

            surrounding_rows = surrounding_rows.with_columns(pl.lit(day1_price).alias("day1_price"))
            if benchmark is not None:
                day1_index_price = surrounding_rows.sort("index_col").select("index_price")[0,0]
                surrounding_rows = surrounding_rows.with_columns(pl.lit(day1_index_price).alias("day1_index_price"))
            return surrounding_rows

        # Apply get_surrounding_rows to each row in input_df and merge results
        result = [df for df in (get_surrounding_rows(daily_df, row) for row in tqdm(input_df.to_dicts())) if df.shape[1] > 0]
        final_df = pl.concat(result)

        # Calculate return column
        if benchmark is None:
            final_df = final_df.with_columns((pl.col("price") / pl.col("day1_price") - 1).alias("return"))
        else:
            final_df = final_df.with_columns(((pl.col("price") / pl.col("day1_price") - 1) - (pl.col("index_price") / pl.col("day1_index_price") - 1)).alias("return"))

        # Group by index_col and calculate mean return
        # final_df = final_df.group_by("index_col").agg(pl.col("return").mean().alias("return"))
        final_df: pl.DataFrame = self._agg_return(final_df, 'index_col', 'return', stat_method, 'return')

        # Sort final_df by index_col
        final_df = final_df.with_columns(pl.col("index_col").cast(pl.Int64)).sort("index_col")

        # Create date index for plotting
        date_index = list(range(self.pdays * -1, self.fdays + 1))
        final_df = final_df.with_columns(pl.Series(date_index).alias("index_col"))

        if output_type == 'df':
            return final_df

        # Plot the cumulative return chart
        plt.clf()
        plt.figure(figsize=(15, 7))
        plt.xlabel("Days to Cal Date")
        plt.ylabel("Return")
        plt.title("Cumulative Return Chart")
        plt.grid(True)
        plt.plot(final_df.to_pandas()["index_col"], final_df.to_pandas()["return"])
        plt.xticks(final_df.to_pandas()["index_col"])
        return plt
    
    def get_cumulative_return_chart_with_group(
        self,
        stat_method: str = 'mean',
        benchmark: str = None,
        output_type: Optional[str] = 'chart'
    ) -> Union[plt.Figure, pl.DataFrame]:
        """
        Generates a cumulative return chart or DataFrame, grouped by a column.

        Args:
        -----------
            stat_method (str): Statistic method, either ```mean``` or ```median```. Default is ```mean```.
            benchmark (str): The benchmark symbol.
            output_type (Optional[str]): The output type, either ```chart``` or ```df```. Default is ```chart```.

        Returns:
        -----------
            Union[plt.Figure, pl.DataFrame]: The cumulative return chart or DataFrame.
        """
        # Check if the output type is valid
        if output_type not in ['chart', 'df']:
            raise ValueError(f"Invalid output type: {output_type}")

        if not isinstance(self.group_series, pl.Series):
            raise ValueError("Group series is required for this method.")

        # Create input DataFrame
        input_df = pl.DataFrame({'tdate': self.tdate_series, 'symbol': self.stock_series, 'group': self.group_series})
        
        # Get daily data
        daily_df = qxdatac.AShareDaily(columns=['symbol', 'tdate', self.price_col]).get(symbols=self.symbol, sdt=self.min_date, edt=self.max_date)
        daily_df = daily_df.rename({self.price_col: 'price'})
        if benchmark is not None:
            if self.price_col.endswith('_adj'):
                index_price_col = self.price_col.replace('_adj', '')
            else:
                index_price_col = self.price_col
            index_df = qxdatac.AShareIndexDaily(columns=['tdate', index_price_col]).get(sdt=self.min_date, edt=self.max_date, idxids=[benchmark])
            index_df = index_df.rename({index_price_col: 'index_price'})
            daily_df = daily_df.join(index_df, on='tdate', how='left')

        # Add row numbers to daily_df
        daily_df = daily_df.sort(['symbol','tdate'])
        daily_df = daily_df.with_row_count("row_num")

        def get_surrounding_rows(df: pl.DataFrame, input_row) -> pl.DataFrame:
            """
            Gets rows surrounding the target date within the specified range.

            Args:
            -------------
                df (pl.DataFrame): DataFrame containing daily data.
                input_row (dict): Row from input_df.

            Returns:
            ------------
                pl.DataFrame: DataFrame containing surrounding rows with additional columns.
            """
            symbol = input_row["symbol"]
            tdate = input_row["tdate"]
            group = input_row["group"]
            
            # Get the row number for the target date
            tdate_row = df.filter((pl.col("symbol") == symbol) & (pl.col("tdate") == tdate))
            
            if tdate_row.shape[0] == 0:
                return pl.DataFrame()
            
            row_num = tdate_row.select("row_num")[0, 0]
            
            # Get rows within the specified range
            surrounding_rows = df.filter(
                (pl.col("symbol") == symbol) & 
                (pl.col("row_num") >= row_num - self.pdays) & 
                (pl.col("row_num") <= row_num + self.fdays)
            )
            
            # Add index_col and day1_price columns
            surrounding_rows = surrounding_rows.with_row_count("index_col")
            day1_price = surrounding_rows.sort("index_col").select("price")[0,0]
            surrounding_rows = surrounding_rows.with_columns(pl.lit(day1_price).alias("day1_price"), pl.lit(group).alias('group'))
            if benchmark is not None:
                day1_index_price = surrounding_rows.sort("index_col").select("index_price")[0,0]
                surrounding_rows = surrounding_rows.with_columns(pl.lit(day1_index_price).alias("day1_index_price"))

            return surrounding_rows

        # Apply get_surrounding_rows to each row in input_df and merge results
        result = [df for df in (get_surrounding_rows(daily_df, row) for row in tqdm(input_df.to_dicts())) if df.shape[1] > 0]
        final_df = pl.concat(result)

        # Calculate return column
        if benchmark is None:
            final_df = final_df.with_columns((pl.col("price") / pl.col("day1_price") - 1).alias("return"))
        else:
            final_df = final_df.with_columns(((pl.col("price") / pl.col("day1_price") - 1) - (pl.col("index_price") / pl.col("day1_index_price") - 1)).alias("return"))

        # Group by index_col and calculate mean return
        # final_df = final_df.group_by("index_col",'group').agg(pl.col("return").mean().alias("return"))
        final_df:pl.DataFrame = self._agg_return(final_df, ['index_col', 'group'], 'return', stat_method, 'return')

        # Sort final_df by index_col
        final_df = final_df.with_columns(pl.col("index_col").cast(pl.Int64)).sort("index_col")

        # Create date index for plotting
        date_index = list(range(self.pdays * -1, self.fdays + 1))
        date_index_dict = {i: date_index[i] for i in range(len(date_index))}
        replacement_df = pl.DataFrame({
            "index_col": list(date_index_dict.keys()),
            "new_index_col": list(date_index_dict.values())
        })
        final_df = final_df.join(replacement_df, on="index_col", how="left")
        final_df.drop_in_place("index_col")
        final_df = final_df.rename({"new_index_col": "index_col"})


        if output_type == 'df':
            return final_df

        # Draw a line chart, x-axis is the index_col, y-axis is the return, and the line is grouped by group
        plt.clf()
        from matplotlib.font_manager import FontProperties
        chinese_font = FontProperties(fname='/software/anaconda3/envs/dev/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/simhei.ttf')
        plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
        plt.figure(figsize=(15, 7))
        plt.xlabel("Days to Cal Date")
        plt.ylabel("Return")
        plt.title("Cumulative Return Chart")
        plt.grid(True)
        for group in final_df.to_pandas()['group'].unique():
            plt.plot(final_df.filter(pl.col('group') == group).to_pandas()["index_col"], final_df.filter(pl.col('group') == group).to_pandas()["return"], label=group)
        plt.xticks(final_df.to_pandas()["index_col"])
        plt.legend(prop=chinese_font)
        return plt
    
    def get_discrete_return_chart(
        self,
        stat_method = 'mean',
        benchmark: str = None,
        output_type: Optional[str] = 'chart'
    ) -> Union[plt.Figure, pl.DataFrame]:
        """
        Generates a discrete return chart or DataFrame.

        Args:
        ----------
            stat_method (str): Statistic method, either ```mean``` or ```median```. Default is ```mean```.
            benchmark (str): The benchmark symbol.
            output_type (Optional[str]): The output type, either ```chart``` or ```df```. Default is ```chart```.

        Returns:
        ----------
            Union[plt.Figure, pl.DataFrame]: The discrete return chart or DataFrame.
        """
        # Check if the output type is valid
        if output_type not in ['chart', 'df']:
            raise ValueError(f"Invalid output type: {output_type}")
        
        # Create input DataFrame
        if self.group_series is not None:
            input_df = pl.DataFrame({'tdate': self.tdate_series, 'symbol': self.stock_series, 'group': self.group_series})
        else:
            input_df = pl.DataFrame({'tdate': self.tdate_series, 'symbol': self.stock_series})

        # Initialize an empty DataFrame for concatenated dates
        date_concated_df = pl.DataFrame()

        # Iterate over the range of previous and forward days
        for i in range(self.pdays * -1, self.fdays + 1):
            temp_df = input_df.clone()
            if i < 0:
                temp_df = temp_df.with_columns(shift(temp_df['tdate'], n=i).alias('buy_date'))
                temp_df = temp_df.with_columns(pl.lit(i).alias('index_date'))
            elif i > 0:
                temp_df = temp_df.with_columns(shift(temp_df['tdate'], n=i).alias('buy_date'))
                temp_df = temp_df.with_columns(pl.lit(i).alias('index_date'))
            elif i == 0:
                temp_df = temp_df.with_columns(temp_df['tdate'].alias('buy_date'))
                temp_df = temp_df.with_columns(pl.lit(i).alias('index_date'))
            date_concated_df = pl.concat([date_concated_df, temp_df])

        # Calculate daily returns
        if benchmark is None:
            date_concated_df = date_concated_df.with_columns([
                qxdatac.ASharePriceCal().get_return_by_series(
                    symbol_series=date_concated_df['symbol'],
                    buy_date_series=date_concated_df['buy_date'],
                    windows=1,
                    buy_price_col=self.price_col,
                    sell_price_col=self.price_col
                ).alias('dod_return')
            ])
        else:
            date_concated_df = date_concated_df.with_columns([
            qxdatac.ASharePriceCal().get_excess_return_by_series(
                benchmark=benchmark,
                symbol_series=date_concated_df['symbol'],
                buy_date_series=date_concated_df['buy_date'],
                windows=1,
                buy_price_col=self.price_col,
                sell_price_col=self.price_col
            ).alias('dod_return')
        ])

        # Group by index_date and calculate mean return
        if self.group_series is None:
            # final_df = date_concated_df.group_by("index_date").agg(pl.col("dod_return").mean().alias("return"))
            final_df:pl.DataFrame = self._agg_return(date_concated_df, 'index_date', 'dod_return', stat_method, 'return')
        else:
            # final_df = date_concated_df.group_by("index_date",'group').agg(pl.col("dod_return").mean().alias("return"))
            final_df:pl.DataFrame = self._agg_return(date_concated_df, ['index_date', 'group'], 'dod_return', stat_method, 'return')

        # Sort final_df by index_date
        final_df = final_df.with_columns(pl.col("index_date").cast(pl.Int64)).sort("index_date")

        if output_type == 'df':
            return final_df


        # Plot the discrete return chart
        plt.clf()
        from matplotlib.font_manager import FontProperties
        chinese_font = FontProperties(fname='/software/anaconda3/envs/dev/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/simhei.ttf')
        plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
        plt.figure(figsize=(15, 7))
        plt.xlabel("Days to Cal Date")
        plt.ylabel("Return")
        plt.title("Discrete Return Chart")
        plt.grid(True)
        if self.group_series is None:
            plt.plot(final_df.to_pandas()["index_date"], final_df.to_pandas()["return"])
        else:
            for group in final_df.to_pandas()['group'].unique():
                plt.plot(final_df.filter(pl.col('group') == group).to_pandas()["index_date"], final_df.filter(pl.col('group') == group).to_pandas()["return"], label=group)
            plt.legend(prop=chinese_font)
        plt.xticks(final_df.to_pandas()["index_date"])
        return plt
    
def sample_bar_chart_by_group(
        group_col: str, 
        *dfs: pl.DataFrame
) -> plt.Figure:
    """
    自动生成柱状图，按照组别分组
    
    参数:
    ------------
        group (str): 列名
        *dfs: 任意数量的 DataFrame

    示例:
    ------------
        sample_bar_chart_by_group(group_col='forecastType', df1, df2, df3)

    返回:
    ------------
        plt.Figure: 柱状图
    """
    from matplotlib.font_manager import FontProperties
    import numpy as np
    chinese_font = FontProperties(fname='/software/anaconda3/envs/dev/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/simhei.ttf')

    # Prepare the data
    all_data = []
    group_labels = []
    for df in dfs:
        forecastType_count = df.group_by(group_col).agg(pl.count(group_col).alias('spec_count')).sort('spec_count', descending=True)
        forecastType_count = forecastType_count.to_pandas()
        all_data.append(forecastType_count['spec_count'])
        group_labels = forecastType_count[group_col].tolist()

    # Set up the plot
    x = np.arange(len(group_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each DataFrame's data
    for i, data in enumerate(all_data):
        ax.bar(x + i * width, data, width, label=f'DF {i + 1}')

    # Add labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('类型', fontproperties=chinese_font)
    ax.set_ylabel('样本数量', fontproperties=chinese_font)
    ax.set_title('样本数量分布', fontproperties=chinese_font)
    ax.set_xticks(x + width / 2 * (len(dfs) - 1))
    ax.set_xticklabels(group_labels, fontproperties=chinese_font)
    ax.legend()

    # Show numbers on the top of the bars
    for i, data in enumerate(all_data):
        for x_pos, y_pos in zip(x + i * width, data):
            ax.text(x_pos, y_pos, '%d' % y_pos, ha='center', va='bottom', fontsize=10)

    plt.show()

    # return fig

def simple_back_test(
        data: pl.DataFrame,
        benchmark: str = '000852.ZICN',
        excess_return: bool = True,
        symbol_col: str = 'symbol',
        date_col: str = 'date',
        weight_col: str = 'weight',
        price_col: str = 'close',
        group_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Perform a simple backtest on a given dataset to calculate cumulative returns.

    Args:
        data (pl.DataFrame): DataFrame containing the stock data for backtesting.
        benchmark (str): Benchmark symbol for calculating excess returns. Defaults to '000852.ZICN'.
        excess_return (bool): If True, calculate excess returns over the benchmark; otherwise, calculate simple returns. Defaults to True.
        symbol_col (str): Name of the column containing stock symbols. Defaults to 'symbol'.
        date_col (str): Name of the column containing dates. Defaults to 'date'.
        weight_col (str): Name of the column containing weights. Defaults to 'weight'.
        price_col (str): Name of the column containing prices. Defaults to 'close'.
        group_col (str): Name of the column containing group labels. Defaults to None.

    Returns:
        pl.DataFrame: DataFrame with cumulative returns for each date.
    """
    sim_return_df = data

    # Calculate either excess returns or simple returns
    if excess_return:
        # Calculate excess returns using the specified benchmark
        sim_return_df = sim_return_df.with_columns([
            qxdatac.ASharePriceCal().get_excess_return_by_series(
                benchmark=benchmark,
                symbol_series=sim_return_df[symbol_col],
                buy_date_series=sim_return_df[date_col],
                windows=1,
                buy_price_col=price_col,
                sell_price_col=price_col
            ).alias('pctChg')
        ])
    else:
        # Calculate simple returns
        sim_return_df = sim_return_df.with_columns([
            qxdatac.ASharePriceCal().get_return_by_series(
                symbol_series=sim_return_df[symbol_col],
                buy_date_series=sim_return_df[date_col],
                windows=1,
                buy_price_col=price_col,
                sell_price_col=price_col
            ).alias('pctChg')
        ])

    # Remove rows with null values
    sim_return_df = sim_return_df.drop_nulls()

    if group_col is None:

        # Calculate weighted returns
        sim_return_df = sim_return_df.with_columns([
            (pl.col('pctChg') * pl.col(weight_col)).alias('pctChg_per')
        ])

        # Aggregate returns by date and sort by date
        sim_return_df = sim_return_df.group_by(date_col).agg(pl.col('pctChg_per').sum()).sort(date_col)

        # Calculate cumulative returns
        sim_return_df = sim_return_df.with_columns([
            pl.col('pctChg_per').cum_sum().alias('cum_sum')
        ])

        # Plot cumulative returns
        plt.figure(figsize=(20, 10))
        plt.plot(sim_return_df[date_col].to_numpy(), sim_return_df['cum_sum'].to_numpy(), label='Cumulative Return')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Set major locator for x-axis to show all the months
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))

        # Set y-axis ticks with step size
        plt.yticks(np.arange(min(sim_return_df['cum_sum'].to_numpy()), 
                            max(sim_return_df['cum_sum'].to_numpy()), step=0.01))

        # Add labels and grid
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid()

        # Show legend
        plt.legend()

        # Display the plot
        plt.show()
    
    else:
        # Calculate weighted returns
        sim_return_df = sim_return_df.with_columns([
            (pl.col('pctChg') * pl.col(weight_col)).alias('pctChg_per')
        ])

        # Aggregate returns by date and group, then sort by date
        sim_return_df = sim_return_df.group_by([date_col, group_col]).agg(pl.col('pctChg_per').sum()).sort(date_col)

        # Calculate cumulative returns for each group
        sim_return_df = sim_return_df.with_columns([
            pl.col('pctChg_per').cum_sum().over(group_col).alias('cum_sum')
        ])

        # Plot cumulative returns for each group
        plt.figure(figsize=(20, 10))
        
        for group in sim_return_df[group_col].unique():
            group_data = sim_return_df.filter(pl.col(group_col) == group)
            plt.plot(group_data[date_col].to_numpy(), group_data['cum_sum'].to_numpy(), label=f'{group}')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Set major locator for x-axis to show all the months
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))

        # Add labels and grid
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid()

        # Show legend
        plt.legend()

        # Display the plot
        plt.show()

        return plt

    return sim_return_df