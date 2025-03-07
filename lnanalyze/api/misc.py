"""
Some functions to help with research
@author: Neo
@date: 2024/6/19
"""
from typing import List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import polars as pl

from qxgentools.utils import DatetimeLike, PathLike, Logger

log = Logger("qxanalyze.api.misc")


def _get_formatted_files(
    sdt: DatetimeLike, 
    edt: DatetimeLike, 
    path: PathLike,
    file_pattern: str = "{date}.csv",
    date_format: str = "%Y-%m-%d",
    use_tcal: bool = True
) -> List[str]: 
    """Retrieve a list of formatted file names between two dates within a directory.

    Args:
        sdt: Starting date.
        edt: Ending date.
        path: Directory to search.
        file_pattern: Format for the file names, defaults to "{date}.csv".
        date_format: Date format used in file names, defaults to "%Y-%m-%d".
        use_tcal: Boolean to use trading calendar, defaults to True.

    Returns:
        A list of file paths that exist according to the specified pattern.
    """
    from qxgentools.timeutils import dt2str

    if use_tcal:
        from qxdatac import TCalendar
        dates = TCalendar(["tdate"]).get(sdt, edt, df_lib="pandas").to_list()
    else:
        from qxgentools.timeutils import get
        dates = get(sdt, edt)
    
    # Format the dates and check file existence.
    files: List[str] = [file_pattern.format(date=dt2str(date, date_format)) for date in dates]
    existing_files = [str(Path(path) / file) for file in files if Path(path / file).exists()]

    # Logging missing files.
    missing_files = list(set(files) - set(existing_files))
    if missing_files:
        from qxgentools.utils.human import lists
        log.warning(f"Missing files: {lists(missing_files)}")

    if not existing_files:
        raise RuntimeError(f"No files matching '{file_pattern}' found in {path}")

    return existing_files

def read_directory(
    path: PathLike,
    reader: Optional[Callable] = None,
    sdt: Optional[DatetimeLike] = None, 
    edt: Optional[DatetimeLike] = None, 
    file_pattern: str = "{date}.csv",
    date_format: str = "%Y-%m-%d",
    use_tcal: bool = True,
    df_lib: str = "polars", 
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Read all files in a directory that match a given pattern or date range.

    Args:
        path: Directory path.
        reader: Function to read files, defaults to qxgentools.utils.read_file.
        sdt: Start date for filtering files.
        edt: End date for filtering files.
        file_pattern, date_format, use_tcal: See _get_formatted_files.
        df_lib: Library to use ('pandas' or 'polars').

    Returns:
        A DataFrame containing data from all read files.
    """
    
    from qxdatac.utils import is_valid_df_lib
    is_valid_df_lib(df_lib)
    
    
    if reader is None:
        from qxgentools.utils import read_file
        reader = read_file

    from qxgentools.utils import get_files
    files = get_files(path) if (sdt is None or edt is None) else _get_formatted_files(
        sdt, edt, path, file_pattern, date_format, use_tcal
    )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda f: reader(f, df_lib=df_lib, **kwargs), files))
        data = pl.concat(results) if df_lib == "polars" else pd.concat(results)

    return data