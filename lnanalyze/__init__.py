"""
@author neo
@time 2024/6/18
"""
import pandas as pd
# Enable copy_on_write to optimize memory usage during chained assignment
pd.options.mode.copy_on_write = True

_qdata_client = None

def get_qdata_client():
    """get qdata_client instance
    
    Returns
    -------
    qdata_client
        qdata_client instance
    """
    global _qdata_client
    if _qdata_client is None:
        try:
            import qdata_client
            qdata_client.CLI_CONFIG.load_config()
            _qdata_client = qdata_client
        except Exception as e:
            from qxgentools.utils import missing_dependency
            missing_dependency("qdata_client")
    return _qdata_client


from .api import (
    # tcalendar
    TCalendar, get_Tcalendar,
    fetch_all_tdate, get_all_tdate_pd, get_all_tdate_pl, check_tdate,
    all_trading, any_trading, get_closest, offsets, trading, shift,
    # chart_stat
    ChartAnalysis, sample_bar_chart_by_group, 
    # statistics
    calculate_stats, extreme_mad, extreme_nsigma, extreme_percentile, 
    zscore, neutralization_by_OLS, neutralization_by_inv, 
    sharpe, ic, ic_batch, factor_stat,
    QCut, simple_back_test, QCutSpeedTools, DataPreProcess, TopBottomTool, lgbm_train_fit, xgboost_train_fit, regression_train_fit_weight_control
)
