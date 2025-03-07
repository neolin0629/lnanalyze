"""
Functions that deal with transaction calendar
@author: Neo
@date: 2024/6/18
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional, Dict, Tuple, List

import pandas as pd
import polars as pl

from qxgentools.timeutils import adjust
from qxgentools.utils import Logger,Columns, is_valid_df_lib, is_valid_exchange, DatetimeLike, lists


log = Logger("qxanalyze.api.tcalendar")


@dataclass
class TCalendar(Columns):
    @property
    def column_map(self) -> Dict[str, str]:
        return {
            "calendarDate": "tdate",
            "isOpen": "is_open",
            "prevTradeDate": "prev",
            "nextTradeDate": "next",
            "current_trade_date_prev": "prev_le",
            "current_trade_date_next": "next_ge",
        }

    @staticmethod
    def get_columns() -> Tuple: 
        return ('tdate', 'is_open', 'prev', 'next', 'prev_le', 'next_ge')

    def get(self,
        sdt: Optional[DatetimeLike] = None,
        edt: Optional[DatetimeLike] = None,
        dtype: str = 'datetime'
    ) -> List[Union[datetime, str]]:
        """Get trading calendar dates list
        
        Parameters
        ----------
        sdt : Optional start date
        edt : Optional end date
        dtype : Return data type
            'datetime': returns list of datetime objects
            'str': returns list of date strings in '%Y-%m-%d' format
            
        Returns
        -------
        List[Union[datetime, str]]
            List of trading calendar dates
        """
        df = self.get_df(sdt, edt, is_open=True, exchange='XSHG', expand=False, df_lib='polars')
        if dtype == 'datetime':
            return df['tdate'].to_list()  # Return list of datetime objects
        elif dtype == 'str':
            return df['tdate'].dt.strftime('%Y-%m-%d').to_list()  # Return list of formatted date strings
        else:
            raise ValueError(f"Invalid dtype: {dtype}")  # Raise error for invalid dtype parameter

    def get_df(self,
        sdt: Optional[DatetimeLike] = None,
        edt: Optional[DatetimeLike] = None,
        is_open: bool = True,
        exchange: str = 'XSHG',
        expand: bool = False, 
        df_lib: str = 'polars',
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Get transaction calendar
        
        Parameters
        ----------
        sdt: Optional[DatetimeLike], default None
            start date
        edt: Optional[DatetimeLike], default None
            end date
        is_open: bool, default True, 
            1: only transaction date, 0: include non-transaction date
        exchange: str, default 'XSHG'
        expand : bool, optional
            Expand with additional columns or not, by default `False`. 
            * ``is_open`` : is transaction date or not
            * ``prev, next`` : previous & next trading day
            * ``prev_le, next_ge`` : previous less than or equal & next greater than or equal
            * ``monthlyn`` : nth trading day in this month
        df_lib: str, default 'polars', 'pandas' or 'polars'
        
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
        """
        is_valid_df_lib(df_lib)
        is_valid_exchange(exchange)

        try:
            from qxanalyze import get_qdata_client
        except Exception as e:
            from qxgentools.utils import missing_dependency
            missing_dependency("qdata_client")
        
        data = get_qdata_client().get_application_data('Calendar').filter(pl.col('exchangeCD') == exchange)
        if is_open:
            data = data.filter(pl.col('isOpen') == 1)
        if sdt: data = data.filter(pl.col('calendarDate') >= adjust(sdt).date())
        if edt: data = data.filter(pl.col('calendarDate') <= adjust(edt).date())
        data = data.rename(mapping=self.column_map)

        if df_lib == 'pandas':
            data = data.to_pandas()
        
        if expand and df_lib == 'pandas':
            data['monthlyn'] = data.groupby([data['tdate'].dt.year, data['tdate'].dt.month])['tdate'].cumcount() + 1
        elif expand and df_lib == 'polars':
            data = data.with_columns(pl.col("tdate").cum_count().over([pl.col("tdate").dt.year(), pl.col("tdate").dt.month()]).alias("monthlyn"))

        if (not expand) and (len(data.columns) > 1):
            data = data.loc[:, ["tdate"]] if df_lib == 'pandas' else data.select(["tdate"])
        
        if df_lib == "pandas": 
            data = data.set_index("tdate", drop=False)
            data.index.name = None
        
        return data
    

    def get_spec(self,
        data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series],
        is_open: bool = True,
        exchange: str = 'XSHG',
        expand: bool = False, 
        df_lib: str = 'polars',
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Get trading calendar specifications
        
        Parameters
        ----------
        dates : Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]
            The dates to get specifications for.
        exchange: str, default 'XSHG'
        df_lib: str, default 'polars'
            pandas or polars

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
        """
        is_valid_df_lib(df_lib)
        is_valid_exchange(exchange)
        
        # 获取时间序列的最小值和最大值
        if isinstance(data, List):
            sdt = min(data)
            edt = max(data)
        elif isinstance(data, pd.Series):
            sdt = data.min()
            edt = data.max()
        elif isinstance(data, pl.Series):
            sdt = data.min()
            edt = data.max()
        elif isinstance(data, pd.DataFrame):
            if 'tdate' in data.columns:
                sdt = data['tdate'].min()
                edt = data['tdate'].max()
            else:
                sdt = data.iloc[:,0].min()
                edt = data.iloc[:,0].max()
        elif isinstance(data, pl.DataFrame):
            if 'tdate' in data.columns:
                sdt = data.get_column('tdate').min()
                edt = data.get_column('tdate').max()
            else:
                sdt = data.get_column(data.columns[0]).min()
                edt = data.get_column(data.columns[0]).max()
                
        # 通过get_df获取交易日序列
        date_seq = self.get_df(sdt=sdt, edt=edt, is_open=is_open, exchange=exchange, expand=expand, df_lib=df_lib)
        
        if df_lib == 'pandas':
            date_seq = date_seq[date_seq['tdate'].isin(pd.to_datetime(data))]
            date_seq = date_seq.set_index('tdate', drop=False)
            date_seq.index.name = None
        else:
            date_seq = date_seq.filter(pl.col('tdate').is_in(data))
        
        return date_seq
    

# Initialize global variables
all_tdate_pd: Optional[pd.DataFrame] = None
all_tdate_pl: Optional[pl.DataFrame] = None

def fetch_all_tdate(is_open: bool = True, expand: bool = False, exchange: str = 'XSHG', df_lib: str = "polars") -> Union[pd.DataFrame, pl.DataFrame]:
    """Fetch all trading dates as a DataFrame from TCalendar.

    Args:
        is_open(bool): Filter by open trading days.
        expand (bool): Include all columns if True.
        df_lib (str) : Library to use ('pandas' or 'polars').

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame containing all trading dates.
    """
    try:
        if df_lib == "pandas":
            global all_tdate_pd
            if all_tdate_pd is None:
                all_tdate_pd = TCalendar().get_df(is_open=False, exchange=exchange, expand=True, df_lib="pandas")
            result = all_tdate_pd.loc[all_tdate_pd['is_open'] == 1] if is_open else all_tdate_pd
            return result if expand else result[["tdate"]]
        
        elif df_lib == "polars":
            global all_tdate_pl
            if all_tdate_pl is None:
                all_tdate_pl = TCalendar().get_df(is_open=False, exchange=exchange, expand=True, df_lib="polars")
            result = all_tdate_pl.filter(pl.col("is_open") == 1) if is_open else all_tdate_pl
            return result if expand else result.select(["tdate"])
        
        else:
            raise ValueError("Unsupported DataFrame library specified. Use 'pandas' or 'polars'.")
    except Exception as e:
        raise RuntimeError(f"Error fetching trading dates: {e}") from e


def get_all_tdate_pd(is_open: bool = True, exchange:str = 'XSHG', expand: bool = False) -> pd.DataFrame:
    """Wrapper to fetch trading dates as a pandas DataFrame."""
    return fetch_all_tdate(is_open=is_open, exchange=exchange, expand=expand, df_lib="pandas")


def get_all_tdate_pl(is_open: bool = True,exchange:str = 'XSHG', expand: bool = False) -> pl.DataFrame:
    """Wrapper to fetch trading dates as a polars DataFrame."""
    return fetch_all_tdate(is_open=is_open,exchange=exchange, expand=expand, df_lib="polars")


def check_tdate(
    data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series],
    date_series_name: str,
    is_open: bool = True,
    exchange: str = 'XSHG',
    expand: bool = False,
    df_lib: str = 'polars'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """检查日期是否为交易日
    
    Parameters
    ----------
    data : Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]
        需要检查的日期数据
    date_series_name : str
        日期序列的名称(用于错误日志)
    is_open : bool, default True
        是否只检查交易日
    exchange : str, default 'XSHG'
        交易所代码
    expand : bool, default False
        是否展开额外列
    df_lib : str, default 'polars'
        返回数据框架类型 ('pandas' 或 'polars')
    
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        包含检查结果的数据框
    """
    is_valid_df_lib(df_lib)
    is_valid_exchange(exchange)
    
    # 获取日期规范数据
    spec_df = get_Tcalendar().get_spec(
        data=data,
        is_open=False,  # 获取所有日期以便检查
        exchange=exchange,
        expand=True,    # 需要 is_open 列进行检查
        df_lib=df_lib
    )
    
    # 检查非交易日
    if df_lib == 'pandas':
        invalid_dates = spec_df[spec_df['is_open'] == 0]
        if len(invalid_dates) > 0:
            log.error(f"Ignore unavailable {date_series_name} rows, which are not trade dates: {lists(invalid_dates['tdate'].tolist())}")
    else:
        invalid_dates = spec_df.filter(pl.col('is_open') == 0)
        if invalid_dates.height > 0:
            log.error(f"Ignore unavailable {date_series_name} rows, which are not trade dates: {lists(invalid_dates.get_column('tdate').to_list())}")
    
    # 根据 is_open 参数过滤结果
    if is_open:
        if df_lib == 'pandas':
            spec_df = spec_df[spec_df['is_open'] == 1]
        else:
            spec_df = spec_df.filter(pl.col('is_open') == 1)
    
    # 根据 expand 参数决定返回列
    if not expand:
        if df_lib == 'pandas':
            spec_df = spec_df[['tdate']]
        else:
            spec_df = spec_df.select(['tdate'])
            
    return spec_df


TCALENDAR: Optional[TCalendar] = None

def get_Tcalendar():
    global TCALENDAR
    if TCALENDAR is None:
        TCALENDAR = TCalendar()
    return TCALENDAR


def all_trading(data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series], exchange: str = "XSHG") -> bool:        
    """
    Determine if the provided dates are all trading days.
    
    Parameters
    ----------
    data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]
        The data containing dates to check if they are trading days.
    exchange: str, default `XSHG`

    Returns
    -------
    bool
    """
    df = get_Tcalendar().get_spec(data, is_open=False, exchange=exchange, expand=True, df_lib="pandas")
    return df["is_open"].eq(1).all()


def any_trading(data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series], exchange: str = "XSHG") -> bool:        
    """
    Determine if the provided dates have any trading day.
    """
    df = get_Tcalendar().get_spec(data, is_open=False, exchange=exchange, expand=True, df_lib="pandas")
    return df["is_open"].eq(1).any()


def get_closest( 
    data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series],
    adjustment: str = "prev", 
    exchange: str = "XSHG",
) -> Union[pd.Series, pl.Series]:
    """
    Retrieve the closest trading day based on the specified adjustment direction.

    
    Parameters
    ----------
    data : Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]
        The data containing dates for which the closest trading day is required.
    exchange: str, default `XSHG`
    adjustment : str, optional
        Direction of adjustment; 'prev' for previous trading day or 'next' for next trading day.
        Defaults to 'prev'.

    Returns
    -------
    Union[pd.Series, pl.Series]
        A series (either pandas or polars) containing the closest trading days to the dates provided in `data`.

    Raises
    ------
    ValueError
        If `adjustment` is not 'next' or 'prev', or if the data does not contain a 'tdate' column when expected.

    """
    if adjustment not in ('next', 'prev'):
        raise ValueError("adjustment must be either 'next' or 'prev'")
    
    df_lib = "pandas" if isinstance(data, (List, pd.DataFrame, pd.Series)) else "polars"

    dates = get_Tcalendar().get_spec(data, is_open=False, exchange=exchange, expand=True, df_lib=df_lib)
    col = "prev_le" if adjustment == "prev" else "next_ge"

    if isinstance(data, (List, pd.Series)): data = pd.DataFrame({"tdate": data})
    if isinstance(data, pd.DataFrame): 
        if not "tdate" in data.columns: raise ValueError("data must contain 'tdate' column")
        data = data.merge(dates, on="tdate", how="left")
    if isinstance(data, pl.Series): data = data.to_frame("tdate")
    if isinstance(data, pl.DataFrame): 
        if not "tdate" in data.columns: raise ValueError("data must contain 'tdate' column")
        data = data.join(dates, on="tdate", how="left", coalesce=True)

    return data.loc[:, col] if df_lib == 'pandas' else data.get_column(col)


def offsets(
    date: Optional[DatetimeLike] = None, 
    n: int = 0, 
    adjustment: str = "prev",
    exchange: str = "XSHG",
) -> Union[pd.Timestamp, str]:
    """
    Adjusts a given date to the nearest trading date in the A-share market, considering specified offsets and directions.

    Parameters
    ----------
    date : DatetimeLike, optional
        The date to be adjusted. If None, today's date is used as default.
    n : int, optional
        The number of trading days to offset. Positive values move forward in time, negative values move backward. Defaults to 0, which means no offset.
    exchange: str, default `XSHG`
    adjustment : str, optional
        Specifies the direction of the adjustment:
        'prev' for adjusting to the previous trading day,
        'next' for adjusting to the next trading day.
        Defaults to 'prev'.

    Returns
    -------
    Union[pd.Timestamp, str]
        The adjusted trading date. The return type is pd.Timestamp if the input is a datetime-like object, or str if the input is a string.

    Notes
    -----
    The function relies on a trading calendar to determine open trading days. If the offset moves the date beyond available data in the calendar, it may result in an error or unexpected behavior.

    """
    adjusted_date = adjust(date or "today", date_only=True)

    trading_dates = get_all_tdate_pd(is_open=False, exchange=exchange, expand=True)

    if adjustment == "prev":
        adjusted_date = trading_dates.loc[adjusted_date, "prev_le"]
    elif adjustment == "next":
        adjusted_date = trading_dates.loc[adjusted_date, "next_ge"]

    if n != 0:
        trading_dates = trading_dates[trading_dates["is_open"] == 1]
        index = trading_dates.index.get_loc(adjusted_date) + n
        res: pd.Timestamp = trading_dates.iloc[index]["tdate"]
    else:
        res: pd.Timestamp = adjusted_date

    return res.strftime("%Y-%m-%d") if isinstance(date, str) else res


def trading(date: Optional[DatetimeLike] = None, exchange: str = "XSHG") -> bool:
    """
    Determine if the specified date is a trading day for the A-share market.

    Parameters
    ----------
    date : DatetimeLike, optional
        The date to check for trading status. If None, today's date is assumed. The function converts the input to a pandas Timestamp.
    exchange: str, default `XSHG`

    Returns
    -------
    bool
        Returns True if the given date is a trading day, otherwise False.

    Notes
    -----
    The function adjusts the given date to ensure it is in the correct format for comparison and uses a pre-defined calendar of trading days to check the date's status.
    """
    adjusted_date = adjust(date, date_only=True) # make sure the date is a pd.Timestamp
    trading_dates = get_all_tdate_pd(is_open=False, exchange=exchange, expand=True)
    return trading_dates.loc[adjusted_date, "tdate"] == trading_dates.loc[adjusted_date, "prev_le"]


def shift(
    data: Union[List, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series],
    n: int = 1,
    adjustment: str = "prev",
    exchange: str = "XSHG",
):
    """Shift Given Trading Dates by Trading Calendar

    Parameters
    ----------
    data : List | pd.Series | pd.DataFrame | pl.DataFrame
        Trading dates : 

        * Single trading date as str or int
        * Multiple trading dates as : 

            * list of pandas.Timestamp
            * pandas.Series of pandas.Timestamp
            * pandas.DataFrame including a "tdate" column
            * polars.Series of polas.Date
            * polars.DataFrame including a "tdate" column
    n : int, optional
        How many days to shift, by default +1. 
    exchange: str, default `XSHG`
    adjustment: str, optional, default to `next`
        `prev` for forward adjustment, `next` for backward adjustment when not trading date


    Returns
    -------
    DatetimeLike | list | pd.Series | pd.DataFrame | pl.DataFrame
        Shifted trading dates.    
        The return type would be according to your input. 
    
    Notes
    -----
    .. warning::

        Your input would be modified in-place if it's a pandas.DataFrame or polars.DataFrame. 
        Make deep copy before passing if you want to keep the original. 
    """
    if n == 0: return data

    if isinstance(data, (List, pd.Series, pd.DataFrame)): 
        pdf: pd.DataFrame = pd.DataFrame({"tdate": data}) if isinstance(data, (List, pd.Series)) else data.copy()
        if "tdate" not in pdf.columns: raise ValueError("Missing 'tdate' column")

        from qxgentools.timeutils import is_date_pd
        if not is_date_pd(pdf["tdate"]): pdf["tdate"] = pd.to_datetime(pdf["tdate"])

        if not all_trading(pdf["tdate"]): pdf["tdate"] = get_closest(pdf["tdate"], exchange=exchange, adjustment=adjustment).to_numpy()

        trading_dates = get_all_tdate_pd(exchange=exchange)
        trading_dates["shifted_date"] = trading_dates["tdate"]
        from qxanalyze.utils import merge_pd
        
        pdf = merge_pd(pdf, trading_dates, n=n, left_id=None, right_id=None)
        pdf["tdate"] = pdf["shifted_date"]
        pdf = pdf.drop(["shifted_date"], axis=1)

        if isinstance(data, List): return pdf["tdate"].to_list()
        if isinstance(data, pd.Series): return pdf["tdate"]
        if isinstance(data, pd.DataFrame): return pdf

    else:
        pdf: pl.DataFrame = data.to_frame(name="tdate") if isinstance(data, pl.Series) else data.clone()
        if "tdate" not in pdf.columns: raise ValueError("Missing 'tdate' column")

        from qxgentools.timeutils import is_date_pl
        if not is_date_pl(pdf, "tdate"): pdf = pdf.with_columns(pl.col("tdate").str.to_date())

        if not all_trading(pdf.get_column("tdate")): pdf = pdf.with_columns(get_closest(pdf["tdate"], exchange=exchange, adjustment=adjustment).alias("tdate"))

        trading_dates = get_all_tdate_pl(exchange=exchange).with_columns(pl.col("tdate").alias("shifted_date"))
        from qxanalyze.utils import merge_pl
        pdf = merge_pl(pdf, trading_dates, n=n, left_id=None, right_id=None)
        pdf = pdf.with_columns(pl.col("shifted_date").alias("tdate")).drop(["shifted_date"])

        if isinstance(data, pl.Series): return pdf.get_column("tdate")
        if isinstance(data, pl.DataFrame): return pdf


if __name__=='__main__':

    from pprint import pprint as print

    # print(TCalendar().get_columns())
    t = TCalendar()  # can pass in the required columns ["tdate", "is_open", "prev_le", "next_ge"]

    # # ! test get
    # df_str = t.get(sdt=20210101, edt=20210131, dtype="str")
    # print(df_str)
    # df_dt = t.get(sdt=20210101, edt=20210131, dtype="datetime")
    # print(df_dt)

    # # ! test get_df
    # pl_all_noexpand = t.get_df(sdt=20210101, edt=20211231, is_open=False, df_lib="polars")
    # print(pl_all_noexpand)

    # # * use `tdate` column if exist or the First one
    # # ! test get_spec
    # pl_sample = pl_all_noexpand.sample(n=10)
    # pl_spec = t.get_spec(pl_sample, is_open=False, expand=True)  
    # print(pl_sample)
    # print(pl_spec)
    
    # # ! test get_closest
    # print(get_closest(pl_spec, 'prev'))
    # print(get_closest(pl_spec, 'next'))

    # # ! test offsets
    # dt_str = "2021-01-01"
    # print(get_Tcalendar().get_df("20201228", "20210107", is_open=False, expand=True, df_lib="pandas"))
    # print(offsets(dt_str))
    # print(offsets(dt_str, n = 1))
    # print(offsets(dt_str, n = -2))
    # print(offsets(dt_str, adjustment="next"))
    # print(offsets(dt_str, n = 1, adjustment="next"))
    # print(offsets(dt_str, n = -2, adjustment="next"))

    # dt_pdt = pd.Timestamp("2021-01-01")
    # print(offsets(dt_pdt))
    # print(offsets(dt_pdt, n = 1))
    # print(offsets(dt_pdt, n = -2))
    # print(offsets(dt_pdt, adjustment="next"))
    # print(offsets(dt_pdt, n = 1, adjustment="next"))
    # print(offsets(dt_pdt, n = -2, adjustment="next"))

    # # ! test trading
    # dt_str = "2021-01-01"
    # print(trading(dt_str))
    # dt_pdt = pd.Timestamp("2021-01-04")
    # print(trading(dt_pdt))
    # print(trading("today"))

    # #! test shift
    # df_open_expand = get_Tcalendar().get_df(sdt=20210101, edt=20211231, is_open=False, df_lib="pandas")
    # df_sample = df_open_expand.sample(10) #.sort_values(by="tdate")
    # print(df_sample)
    # # print(get_Tcalendar().get_spec(df_sample, is_open=False, expand=True, df_lib="pandas"))
    # print(shift(df_sample, n=2))
    # print(shift(df_sample, n=2, adjustment="next"))
    # print(df_sample["tdate"].apply(lambda x: offsets(x, n=2)))
    # print(df_sample["tdate"].apply(lambda x: offsets(x, n=2, adjustment="next")))

    # pl_open_expand = get_Tcalendar().get_df(sdt=20210101, edt=20211231, is_open=False, df_lib="polars")
    # pl_sample = pl_open_expand.sample(10) #.sort(by="tdate")
    # print(pl_sample)
    # # print(get_Tcalendar().get_spec(pl_sample, is_open=False, expand=True, df_lib="polars"))
    # print(shift(pl_sample, n=2))
    # print(shift(pl_sample, n=2, adjustment="next"))
    # print(pl_sample["tdate"].map_elements(lambda x: offsets(x, n=2), return_dtype=pl.Date))
    # print(pl_sample["tdate"].map_elements(lambda x: offsets(x, n=2, adjustment="next"), return_dtype=pl.Date))
    pass