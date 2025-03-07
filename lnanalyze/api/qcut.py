"""
QCut Function for Quantile Single Factor Analysis
@author: Johnny
@date: 2024/6/20
"""

import polars as pl
from typing import List, Optional, Union, Callable, Tuple


class QCut:
    """
    A class used to perform quantile-based discretization.

    Attributes:
        boundaries (pl.DataFrame): DataFrame containing the boundaries for each group.
    
    Methods:
        train(factors_column: pl.Series, target_column: pl.Series, groups: int = 5):
            Trains the quantile cutter and sets the boundaries based on the input data.
        fit(factors_column: pl.Series):
            Fits the input factors to the trained boundaries and returns the group labels.
    """
    def __init__(self): #Internal Use
        self.boundaries = None

    def train(self, factors_column: pl.Series, groups: int = 5):
        """
        Trains the quantile cutter and sets the boundaries based on the input data.

        Args:
            factors_column (pl.Series): Series containing the factor data.
            target_column (pl.Series): Series containing the target data.
            groups (int, optional): Number of quantile groups to divide the data into. Defaults to 5.
        """
        df = pl.DataFrame({
            'factors': factors_column,
        })

       #qcut 
        df = df.with_columns([
            df['factors'].qcut(groups, labels=[str(i + 1) for i in range(groups)], allow_duplicates=True).alias("group")
        ])

        boundary = df['factors'].sort().qcut(groups, allow_duplicates=True)
        #unique to list
        self.boundaries = boundary.unique().to_list() #output eg ['(-inf, 0.1]', '(0.1, 0.3]', '(0.3, inf]']
        return df



    def fit(self, factors_column: pl.Series):
        """
        Fits the input factors to the trained boundaries and returns the group labels.

        Args:
            factors_column (pl.Series): Series containing the factor data.

        Returns:
            pl.Series: Series containing the group labels for each factor.
        """

        if self.boundaries is None:
            raise ValueError("The QCut instance has not been trained. Please call the 'train' method first.")

        def parse_bins(bin_strings):
            bins = []
            for bin_str in bin_strings:
                left, right = bin_str[1:-1].split(', ')
                left = float('-inf') if left == '-inf' else float(left)
                right = float('inf') if right == 'inf' else float(right)
                bins.append((left, right))
            bins = bins[1:-1]
            return bins
        
        groups = len(self.boundaries)
        parsed_bins = parse_bins(self.boundaries)
        bins = sorted(set([b for bin_pair in parsed_bins for b in bin_pair]))

        df = pl.DataFrame({
            'factors': factors_column
        })

        df = df.with_columns([
            df['factors'].cut(breaks=bins, labels=[str(i + 1) for i in range(groups)], left_closed=False).alias("group")
        ])
        return df["group"]

########## Test ##########
if __name__ == '__main__':
    data = {
    "factors_column": [5,30,1,2,7,8,10,20,30]
        }
    df = pl.DataFrame(data)
    qcut = QCut()
    print(qcut.train(df['factors_column'], 5))
    df = df.with_columns([
        qcut.fit(df['factors_column']).alias("group")])
    print (df)

class QCutSpeedTools:
    def __init__(self,
                 training_df: pl.DataFrame,
                 testing_df: pl.DataFrame,
                 validation_df: pl.DataFrame):
        """
        Initialize QCutSpeedTools with training, testing, and validation DataFrames.

        Args:
            training_df (pl.DataFrame): DataFrame used for training the QCut model.
            testing_df (pl.DataFrame): DataFrame used for testing the QCut model.
            validation_df (pl.DataFrame): DataFrame used for validating the QCut model.
        """
        self.training_df = training_df
        self.testing_df = testing_df
        self.validation_df = validation_df

    def qcut(self,
             factor_column: str,
             group_column: str = None,
             groups: int = 5) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Apply quantile-based discretization to the specified factor column, dividing data into specified groups.

        Args:
            factor_column (str): Name of the column to apply qcut on.
            group_column (str): Optional name for the new column that holds group labels. Defaults to None,
                                in which case a name is generated based on the factor_column.
            groups (int): Number of quantile groups to create. Defaults to 5.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: DataFrames with the new qcut group column added.

        Raises:
            ValueError: If the specified factor_column does not exist in the DataFrames.
        """
        # If no group_column name is specified, create one based on the factor_column
        if group_column is None:
            group_column = factor_column + "_qcut"

        qcut = QCut()

        # Ensure the factor_column exists in the DataFrames
        if factor_column not in self.training_df.columns:
            raise ValueError(f"Column '{factor_column}' does not exist in the training DataFrame.")
        if factor_column not in self.testing_df.columns:
            raise ValueError(f"Column '{factor_column}' does not exist in the testing DataFrame.")
        if factor_column not in self.validation_df.columns:
            raise ValueError(f"Column '{factor_column}' does not exist in the validation DataFrame.")

        # Train QCut model on the training DataFrame
        qcut.train(self.training_df[factor_column], groups)

        # Apply QCut model to training, testing, and validation DataFrames
        self.training_df = self.training_df.with_columns([
            qcut.fit(self.training_df[factor_column]).alias(group_column)
        ])
        self.testing_df = self.testing_df.with_columns([
            qcut.fit(self.testing_df[factor_column]).alias(group_column)
        ])
        self.validation_df = self.validation_df.with_columns([
            qcut.fit(self.validation_df[factor_column]).alias(group_column)
        ])

        return self.training_df, self.testing_df, self.validation_df

    def fit(self,
            filter_dict: dict) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Filter DataFrames based on a dictionary of column-value pairs.

        Args:
            filter_dict (dict): A dictionary where keys are column names and values are lists of allowed values.
            Example:
            ```
            qcut_dict = {'JUMP_qcut': [2,3], 'VOL_qcut': [1,2]}
            ```
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Filtered DataFrames.

        Raises:
            ValueError: If any specified key in filter_dict does not exist in the DataFrames.
        """
        # Make copies of the original DataFrames for filtering
        training_df = self.training_df
        testing_df = self.testing_df
        validation_df = self.validation_df

        # Iterate over the filter dictionary and apply filters
        for key, value in filter_dict.items():
            # Ensure the key exists in the DataFrames
            if key not in training_df.columns:
                raise ValueError(f"Column '{key}' does not exist in the training DataFrame.")
            if key not in testing_df.columns:
                raise ValueError(f"Column '{key}' does not exist in the testing DataFrame.")
            if key not in validation_df.columns:
                raise ValueError(f"Column '{key}' does not exist in the validation DataFrame.")

            # Convert all filter values to strings
            value = [str(i) for i in value]

            # Filter each DataFrame based on the current key-value pair
            training_df = training_df.filter(pl.col(key).is_in(value))
            testing_df = testing_df.filter(pl.col(key).is_in(value))
            validation_df = validation_df.filter(pl.col(key).is_in(value))

        return training_df, testing_df, validation_df


    
def TopBottomTool(
            df: pl.DataFrame,
            symbol_column: str,
            date_column: str,
            factor_column: str,
            top_n: int,
            bottom_n: int) -> pl.DataFrame:
    """
    Create a Series indicating top and bottom performers based on a factor.

    Args:
        df (pl.DataFrame): Input DataFrame.
        date_column (str): Name of the column containing dates.
        factor_column (str): Name of the column containing factor values.
        top_n (int): Number of top performers to select.
        bottom_n (int): Number of bottom performers to select.

    Returns:
        pl.Series: A Series with 1 for top performers, -1 for bottom performers, and 0 for others.
    """
    # Sort the DataFrame by date and factor value
    sorted_df = df.sort([date_column, factor_column], descending=[False, True])

    # Add row numbers within each date group
    sorted_df = sorted_df.with_columns(
        pl.col(factor_column).rank().over(date_column).alias("rank")
    )

    # Calculate the total count for each date
    date_counts = sorted_df.group_by(date_column).agg(pl.count().alias("total"))
    sorted_df = sorted_df.join(date_counts, on=date_column)

    # Assign values based on rank
    result = sorted_df.with_columns(
        pl.when(pl.col("rank") <= top_n).then(1)
        .when(pl.col("rank") > pl.col("total") - bottom_n).then(-1)
        .otherwise(0)
        .alias("result")
    )

    result = result.select(date_column, symbol_column, "result")
    df = df.join(result, on=[date_column, symbol_column], how='left')

    df = df.with_columns([
        pl.when(pl.col('result') == 1).then(pl.lit('top')).otherwise(
            pl.when(pl.col('result') == -1).then(pl.lit('bottom')).otherwise(pl.lit('others'))
        ).alias('group')
    ])
    return df
        
########## Deprecated Code ##########
# class QuantileSingleFactorAnalysis:
#     def __init__(self):
#         self.boundaries = None

#     def train(self, training_data=None, feature_column=None, target_column=None, num_groups=5):
#         #Checker
#         if not isinstance(training_data, pd.DataFrame):
#             raise ValueError("Input Should be Pandas Dataframe")
#         if not isinstance(feature_column, str):
#             raise ValueError("Feature Column should be string")
#         if not isinstance(target_column, str):
#             raise ValueError("Target Column should be string")
#         if not isinstance(num_groups, int):
#             raise ValueError("Number of Groups should be integer")

#         # Determine grouping boundaries using quantile cut
#         training_data['group'] = pd.qcut(training_data[feature_column], num_groups, labels=False)
#         self.boundaries = pd.qcut(training_data[feature_column], num_groups, retbins=True)[1]

#         self.boundaries[0] = float('-inf')  # Set the minimum boundary to negative infinity
#         self.boundaries[-1] = float('inf')  # Set the maximum boundary to positive infinity

#         self.num_groups = num_groups
#         self.feature_column = feature_column
#         self.target_column = target_column

#     def fit(self, test_data=None, boundaries=None):
#         #Checker
#         if not isinstance(test_data, pd.DataFrame):
#             raise ValueError("Input Should be Pandas Dataframe")
#         if self.feature_column not in test_data.columns:
#             raise ValueError("Feature Column Not Found")

#         # Apply the trained boundaries to the test data
#         if boundaries is not None:
#             self.boundaries = boundaries
#         test_data['group'] = pd.cut(
#             test_data[self.feature_column], 
#             bins=self.boundaries, 
#             labels=np.arange(self.num_groups), 
#             right=False, 
#             include_lowest=True
#         )
#         return test_data['group']

#     def statistics(self, data_frame=None, group_column='group'):
#         if not isinstance(data_frame, pd.DataFrame):
#             raise ValueError("Input Should be Pandas Dataframe")
#         if not isinstance(group_column, str):
#             raise ValueError("Group Column should be string")
#         elif group_column not in data_frame.columns:
#             raise ValueError("Group Column Not Found")
        
#         group_stats = data_frame.groupby(group_column).agg(
#             average_return=(self.target_column, 'mean'),
#             return_std_dev=(self.target_column, 'std'),
#             sample_size=(self.target_column, 'count'),
#             win_rate=(self.target_column, lambda x: (x > 0).mean())
#         ).reset_index()
#         group_stats['sharpe_ratio'] = group_stats['average_return'] / group_stats['return_std_dev']
#         return group_stats

#     def best_sharpe_ratio(self, data_frame, group_column='group'):
#         if data_frame is not pd.DataFrame:
#             raise ValueError("Input Should be Pandas Dataframe")
#         if group_column is not str:
#             raise ValueError("Group Column should be string")
#         elif group_column not in data_frame.columns:
#             raise ValueError("Group Column Not Found")
        
#         group_stats = self.statistics(data_frame, group_column)
#         return group_stats.loc[group_stats['sharpe_ratio'].idxmax()]