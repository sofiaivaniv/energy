import pandas as pd 
pd.set_option('display.float_format', '{:.2f}'.format)
import numpy as np 
from typing import Tuple, Optional, Union 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
 
class EnergyForecaster: 
    def __init__(self): 
        """
        Initialize the EnergyForecaster class.

        This constructor initializes an EnergyForecaster object, but does not initialize any model at this stage.
        """
        self.model = None  # The forecasting model is not initialized yet.
 
    def _detect_date_format(self, date_series: pd.Series) -> str: 
        """
        Detect the date format in the input series.

        This method analyzes the first value in the series to determine whether the dates are in a year-based 
        format (YYYY) or in a full datetime format.

        Args:
            date_series (pd.Series): A series containing the date data.

        Returns:
            str: 'year' if the format is YYYY or 'datetime' if it is a full datetime format.
        
        Raises:
            ValueError: If the date format is unsupported.
        """
        # Convert the first element to a string to handle both string and numeric types
        sample = str(date_series.iloc[0]) 

        # Check if it's a 4-digit year
        if sample.isdigit() and len(sample) == 4: 
            return 'year'  # If it's a year

        # Try parsing as datetime
        try: 
            pd.to_datetime(date_series.iloc[0]) 
            return 'datetime' 
        except:
            raise ValueError("Unsupported date format") 

    def prepare_data(self, 
                 data: pd.DataFrame, 
                 target_column: str = 'quantity', 
                 test_size: float = 0.2 
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for forecasting.

        This method preprocesses the input data by handling missing values, detecting the date format, 
        and splitting the data into training and test sets based on the specified test size.

        Args:
            data (pd.DataFrame): Input DataFrame containing the data.
            target_column (str): The column name that contains the target values (default is 'quantity').
            test_size (float): Proportion of data to use for testing, between 0 and 1 (default is 0.2).

        Returns:
            Tuple: A tuple containing the training DataFrame and test DataFrame.
        
        Raises:
            ValueError: If test_size is not between 0 and 1.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        # Create a copy of the data
        df = data.copy()

        # Convert the target column to numeric
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

        # Remove NaN values
        df = df.dropna(subset=[target_column])

        # Sort by date if it exists
        if 'date' in df.columns:
            # Convert 'date' column to numeric if it's year-based or datetime
            if self._detect_date_format(df['date']) == 'year':
                df['date'] = pd.to_numeric(df['date'], errors='coerce')
            else:
                df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        # Calculate the sizes for the train-test split
        total_size = len(df)
        test_size_count = round(total_size * test_size)  # Use rounding for better precision
        train_size = total_size - test_size_count

        # Adjust for any rounding issues
        if train_size + test_size_count != total_size:
            train_size = total_size - test_size_count

        # Split the data using iloc
        train_data = df.iloc[:train_size].copy()
        test_data = df.iloc[train_size:].copy()

        # Final assertion to verify the split
        assert len(train_data) + len(test_data) == total_size, (
            f"Split sizes mismatch: train={len(train_data)}, test={len(test_data)}, total={total_size}"
        )

        return train_data, test_data

    def preprocess_raw_data(self, data: pd.DataFrame, date_column: str = 'date', value_column: str = 'quantity') -> pd.Series: 
        """
        Preprocess raw data into a time series.

        This method detects the date format, processes the values into a time series, and handles missing 
        values by either forward filling or interpolating.

        Args:
            data (pd.DataFrame): Raw input data.
            date_column (str): The name of the column containing date values (default is 'date').
            value_column (str): The name of the column containing the values to forecast (default is 'quantity').

        Returns:
            pd.Series: A time series with processed and aggregated values.
        """
        # Create a copy of the data
        df = data.copy()

        # Detect the date format
        date_format = self._detect_date_format(df[date_column]) 

        if date_format == 'year': 
            # If the data is annual, use years as the index
            df['proper_date'] = pd.to_numeric(df[date_column], errors='coerce') 
            freq = 'Y'  # Yearly frequency
        else: 
            # If dates are daily
            df['proper_date'] = pd.to_datetime(df[date_column]) 
            freq = 'D'  # Daily frequency

        # Convert the value column to numeric
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        # Group by date and sum the values
        time_series = df.groupby('proper_date')[value_column].sum()

        # Reindex to include all original date ranges
        time_series = time_series.reindex(df['proper_date'].drop_duplicates().sort_values())

        # Handle missing values
        if freq == 'Y': 
            time_series = time_series.ffill().bfill()
        else: 
            time_series = time_series.ffill(limit=7)  # Forward fill for up to 7 days
            time_series = time_series.interpolate(method='linear')  # Linear interpolation

        return time_series

    def forecast_consumption(self, 
                             data: Union[pd.Series, pd.DataFrame], 
                             periods: int, 
                             method: str = 'holt-winters', 
                             seasonal_periods: Optional[int] = 7 
                             ) -> pd.Series: 
        """ 
        Forecast future energy consumption.

        This method uses the specified forecasting method to predict future energy consumption 
        for a given number of periods.

        Args:
            data (Union[pd.Series, pd.DataFrame]): The input data to forecast (must be a pandas Series).
            periods (int): The number of periods to forecast.
            method (str): The forecasting method to use (default is 'holt-winters').
            seasonal_periods (Optional[int]): The number of seasonal periods (default is 7 for weekly data).

        Returns:
            pd.Series: The forecasted energy consumption for the given periods.
        
        Raises:
            ValueError: If the input data is not a pandas Series or if an unsupported method is specified.
        """
        if isinstance(data, pd.DataFrame): 
            raise ValueError("Input data must be a pandas Series") 

        # Handle different index types
        if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)): 
            try: 
                # Try to convert to PeriodIndex for yearly data 
                if all(str(idx).isdigit() and len(str(idx)) == 4 for idx in data.index): 
                    data.index = pd.PeriodIndex(data.index, freq='Y') 
                else: 
                    data.index = pd.DatetimeIndex(data.index) 
            except: 
                raise ValueError("Could not process index format") 

        # Convert to numeric if needed
        data = pd.to_numeric(data, errors='coerce') 

        # Handle missing values
        data = data.interpolate(method='linear') 

        # Sort index if needed
        if not data.index.is_monotonic_increasing: 
            data = data.sort_index() 

        # Validate data length
        is_yearly = isinstance(data.index, pd.PeriodIndex) and data.index.freq == 'Y' 
        min_required_points = 3 if is_yearly else 5 
        if len(data) < min_required_points: 
            raise ValueError(f"Need at least {min_required_points} data points, but got {len(data)}") 

        # Adjust seasonal_periods based on frequency 
        if is_yearly: 
            seasonal_periods = 1  # No seasonality for yearly data 

        if method == 'holt-winters': 
            model = ExponentialSmoothing( 
                data, 
                seasonal_periods=seasonal_periods, 
                trend='add', 
                seasonal='add' if seasonal_periods > 1 else None, 
                initialization_method='estimated' 
            ) 
            self.model = model.fit(optimized=True, use_brute=True) 
            forecast = self.model.forecast(periods) 
            return forecast 
        else: 
            raise ValueError(f"Unsupported forecasting method: {method}")










