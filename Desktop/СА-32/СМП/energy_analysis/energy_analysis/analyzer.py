import pandas as pd
from typing import Dict


class EnergyAnalyzer:
    '''
    A class for analyzing energy consumption patterns and statistics.

    This class provides methods for statistical analysis and pattern
    detection in energy consumption data.
    '''

    def __init__(self):
        """
        Initialize the EnergyAnalyzer.

        This method initializes the EnergyAnalyzer class, which is responsible for
        processing and analyzing energy consumption data.
        """
        pass

    def _ensure_numeric_column(self,
                               data: pd.DataFrame,
                               column: str) -> pd.Series:
        """
        Ensure a column contains numeric values, converting if necessary.

        This method checks if the specified column in the DataFrame contains numeric values.
        If the column is not numeric, it will attempt to convert the values into numeric format,
        handling any non-numeric values by coercing them into NaN.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the data to be processed.
        column : str
            Name of the column to convert to numeric.

        Returns
        -------
        pd.Series
            A pandas Series containing the numeric values from the specified column.
        """
        series = data[column].copy()

        if series.dtype == object:
            # Replace any non-breakable spaces or other whitespace
            series = series.str.strip()
            # Replace comma with dot for decimal numbers
            series = series.str.replace(',', '.')
            # Convert to numeric, coercing errors to NaN
            series = pd.to_numeric(series, errors='coerce')

        return series

    def _parse_date(self,
                    data: pd.DataFrame,
                    date_column: str) -> pd.Series:
        """
        Parse the date column to handle both year-only and full-date formats.

        This method attempts to convert the specified date column into a pandas datetime object.
        If the conversion fails (for a full date), it assumes that the column represents only years
        and converts it accordingly.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the date information.
        date_column : str
            The name of the column to convert to a datetime format.

        Returns
        -------
        pd.Series
            A pandas Series with parsed date or year information.

        Raises
        ------
        ValueError
            If neither full date conversion nor year-only conversion can be performed.
        """
        # Try to convert the column to datetime
        try:
            # If the column is in a full date format (e.g., 'YYYY-MM-DD')
            parsed_date = pd.to_datetime(data[date_column], errors='coerce')
            if parsed_date.isna().all():
                raise ValueError("Full date conversion failed")
        except Exception as e:
            # If not a full date, assume it's just a year
            print(f"Warning: {e}, assuming it's just a year")
            parsed_date = pd.to_datetime(data[date_column], format='%Y', errors='coerce')

        return parsed_date

    def calculate_basic_stats(self,
                              data: pd.DataFrame,
                              consumption_column: str = 'quantity'
                              ) -> Dict[str, float]:
        """
        Calculate basic statistical measures of energy consumption.

        This method calculates common statistical measures such as mean, median,
        standard deviation, minimum, and maximum for the specified energy consumption column.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing energy consumption data.
        consumption_column : str, optional
            The name of the column containing consumption data. Default is 'quantity'.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the calculated statistical measures.

        Raises
        ------
        ValueError
            If the specified consumption column is not found in the data or if no valid data is available
            after cleaning.
        """
        if consumption_column not in data.columns:
            raise ValueError(f"Column {consumption_column} not found in data")

        # Convert consumption column to numeric
        numeric_consumption = self._ensure_numeric_column(data, consumption_column)

        # Remove NaN values before calculating statistics
        clean_consumption = numeric_consumption.dropna()

        if len(clean_consumption) == 0:
            raise ValueError("No valid numeric data found after cleaning")

        stats_dict = {
            'mean': float(clean_consumption.mean()),
            'median': float(clean_consumption.median()),
            'std': float(clean_consumption.std()),
            'min': float(clean_consumption.min()),
            'max': float(clean_consumption.max()),
            'valid_entries': len(clean_consumption),
            'total_entries': len(numeric_consumption)
        }

        return stats_dict

    def detect_patterns(self,
                        data: pd.DataFrame,
                        consumption_column: str = 'quantity',
                        date_column: str = 'date') -> Dict[str, pd.Series]:
        """
        Detect patterns in energy consumption data based on the format of the date column.

        This method identifies patterns in the energy consumption data by analyzing the date column.
        If the date column contains only years, it will generate yearly consumption patterns.
        If the date column contains full dates, it will generate monthly, weekly, and daily patterns.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing energy consumption data.
        consumption_column : str, optional
            The name of the column containing consumption data. Default is 'quantity'.
        date_column : str, optional
            The name of the column containing date information. Default is 'date'.

        Returns
        -------
        Dict[str, pd.Series]
            A dictionary containing the detected patterns for the energy consumption.

        Raises
        ------
        ValueError
            If neither the date nor the consumption column is found in the data.
        """
        if consumption_column not in data.columns:
            raise ValueError(f"Column {consumption_column} not found in data")
        if date_column not in data.columns:
            raise ValueError(f"Column {date_column} not found in data")

        # Create a copy of the data to avoid modifying the original
        data = data.copy()

        # Check if the date column is numeric (i.e., contains only years) or full dates
        if pd.api.types.is_numeric_dtype(data[date_column]):
            # If it's only a year, create patterns for the year
            data[date_column] = pd.to_datetime(data[date_column].astype(str) + '-01-01', errors='coerce')
            print("Detected year-only format, generating yearly patterns.")
            patterns = {
             'yearly_pattern': data.groupby(data[date_column].dt.year)[consumption_column].mean()
            }
        elif pd.api.types.is_datetime64_any_dtype(data[date_column]):
            # If it's a full date, create patterns for month, week, and day
            print("Detected full date format, generating monthly, weekly, and daily patterns.")
            patterns = {}
            if data[date_column].dt.month.notna().any():
                patterns['monthly_pattern'] = data.groupby(data[date_column].dt.month)[consumption_column].mean()
            if data[date_column].dt.weekday.notna().any():
                patterns['weekly_pattern'] = data.groupby(data[date_column].dt.dayofweek)[consumption_column].mean()
            if data[date_column].dt.hour.notna().any():
                patterns['daily_pattern'] = data.groupby(data[date_column].dt.hour)[consumption_column].mean()
        else:
            raise ValueError(f"Unsupported date format in column {date_column}")

        return patterns

