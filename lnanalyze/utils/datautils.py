"""
Tools for data processing
@author: Neo
@date: 2024-06-15
"""

import pandas as pd
import polars as pl
import concurrent.futures
from typing import Union, Optional, List, Dict

from qxgentools.timeutils import is_date_pd

from qxdatac.config import CONFIG


def merge_pd(
    A: pd.DataFrame,
    B: pd.DataFrame,
    n: int = 0,
    left_id : Optional[str] = "symbol",
    right_id: Optional[str] = "symbol",
    how     : str = "left",
    tdays   : Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge two time-series table with given date lags

    Args:
        A | B: pd.DataFrame
            A & B should include "tdate" column, in ["object" | "pd.Timestamp"] types
        n: int, default to 0
            Day offset for merging; positive means B's date is n days after A's.
            n>0: B is future information to A 
            n<0: A is future information to B
        left_id: str, optional
            Column name to join on in A, default to "symbol"
        right_id: str, optional
            Column name to join on in B, default to "symbol"
        how: str, optional
            Type of merge operation, default to "left"
        tdays: pd.DataFrame
            DataFrame of all trading dates.

    Returns:
        pd.DataFrame: Merged DataFrame based on specified conditions.
    """
    assert how in ("left", "right", "outer", "cross", "inner"), "Unsupported join method."

    l_lst = ["refdate", left_id] if left_id else ["refdate"]
    r_lst = ["refdate", right_id] if right_id else ["refdate"]  

    B.rename(columns={"tdate": "refdate"}, inplace=True)
    if not is_date_pd(B["refdate"]):
        B['refdate'] = pd.to_datetime(B['refdate'])

    if n==0:        
        A['refdate'] = pd.to_datetime(A['tdate']) if not is_date_pd(A["tdate"]) else A['tdate']

    else:
        if tdays is None:
            from qxdatac.research import get_all_tdate_pd
            tdays = get_all_tdate_pd()
        offset_dates = tdays['tdate'].shift(-n)
        alldays = pd.DataFrame({'_tdt': tdays['tdate'], 'refdate': offset_dates}).dropna()

        A['_tdt'] = pd.to_datetime(A['tdate']) if not is_date_pd(A["tdate"]) else A['tdate']   
        A = A.merge(alldays, on="_tdt", how="left")

    merged_df = pd.merge(A, B, left_on=l_lst, right_on=r_lst, how=how)
    merged_df = merged_df.drop(columns=["_tdt", "refdate"], errors='ignore')

    return merged_df


def merge_pl(
    A: Union[pl.LazyFrame, pl.DataFrame],
    B: Union[pl.LazyFrame, pl.DataFrame],
    n: int = 0,
    left_id : Optional[str] = "symbol",
    right_id: Optional[str] = "symbol",
    how     : str = "left",
    tdays   : Optional[pl.DataFrame] = None
) -> pl.LazyFrame | pl.DataFrame:
    """
    Merge two time-series tables with given date lags in polars.

    Args:
        A | B: pl.DataFrame | pl.LazyFrame
            A & B should include "tdate" column, in ["object" | "pl.Timestamp"] types
        n: int, default to 0
            Day offset for merging; positive means B's date is n days after A's.
            n>0: B is future information to A 
            n<0: A is future information to B
        left_id: str, optional
            Column name to join on in A, default to "symbol"
        right_id: str, optional
            Column name to join on in B, default to "symbol"
        how: str, optional
            Type of merge operation, default to "left"
        tdays: pl.DataFrame
            DataFrame of all trading dates.

    Returns:
        pl.LazyFrame | pl.DataFrame: Merged DataFrame based on specified conditions.
    """
    assert how in ("inner", "left", "outer", "semi", "anti", "cross", "outer_coalesce"), "Unsupported join method."

    l_lst = ["refdate", left_id] if left_id else ["refdate"]
    r_lst = ["refdate", right_id] if right_id else ["refdate"]


    flag_lazy:bool = True if isinstance(A, pl.LazyFrame) else False

    # handled in a lazy manner, improving efficiency.
    if not isinstance(A, pl.LazyFrame): A = A.lazy()
    if not isinstance(B, pl.LazyFrame): B = B.lazy()

    B = B.with_columns(pl.col("tdate").cast(pl.Date)).rename({"tdate": "refdate"})

    if n == 0:
        A = A.with_columns(pl.col("tdate").cast(pl.Date).alias("refdate"))
    
    else:
        if tdays is None:
            from qxdatac.research import get_all_tdate_pl
            tdays = get_all_tdate_pl()
        if not isinstance(tdays, pl.LazyFrame):
            tdays = tdays.lazy()

        alldays = tdays.with_columns(
            pl.col("tdate").shift(-n).alias("refdate"),
        ).drop_nulls().rename({"tdate": "_tdt"})

        A = A.with_columns(pl.col("tdate").cast(pl.Date).alias("_tdt"))
        A = A.join(alldays, on="_tdt", how="left", coalesce=True).drop("_tdt")

    merged_df = A.join(B, left_on=l_lst, right_on=r_lst, how=how, coalesce=True).drop("refdate")
    return merged_df if flag_lazy else merged_df.collect()


def read_sql(
    url: str, sql: Union[str, List[str]], df_lib: str = CONFIG.df_lib, params: Optional[Union[List, Dict]]=None
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Executes SQL query or queries against a DuckDB database and returns the result as a DataFrame.
    
    Parameters:
        url (str): The connection string to the DuckDB database.
        sql (Union[str, List[str]]): SQL query or list of SQL queries to be executed. 
                                     If a list is provided, queries will be executed in parallel.
        df_lib (str): The library to use for DataFrame operations, 'polars' or 'pandas'.
                      Default is read from a global CONFIG object.
        params (Optional[Union[List, Dict]]): Optional parameters for the SQL queries.
                                           Can be a list or dictionary.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame object containing the results of the query.
                                         The type of DataFrame depends on the `df_lib` specified.
    
    Raises:
        ImportError: If the required 'duckdb' module is not installed.
    """
    try:
        import duckdb
    except ModuleNotFoundError as e:
        raise ImportError(f"Missing optional dependency: {e.name}")
    
    def execute_query(sql: str):
        with duckdb.connect(url, read_only=True) as con:
            result = con.sql(sql) if params is None else con.sql(sql, params)
            return result.pl() if df_lib == 'polars' else result.df()
    

    if isinstance(sql, list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_query, s) for s in sql]
            results = [f.result() for f in futures]

        return pl.concat(results) if df_lib == 'polars' else pd.concat(results, ignore_index=True)
    else:
        return execute_query(sql)