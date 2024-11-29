import pandas as pd
import numpy as np
from typing import Optional

class DataProcessor:
    '''
    A class for processing and cleaning energy consumption data.

    This class provides methods for data loading, cleaning, validation,
    and preprocessing of energy consumption time series data.
    '''

    def __init__(self):
        """
        Initialize the DataProcessor instance.

        Attributes
        ----------
        data : pd.DataFrame or None
            The internally stored DataFrame containing loaded and processed data.
        """
        self.data = None

    def load_data(self,
                  file_path: str,
                  date_column: str = 'date',
                  consumption_column: str = 'quantity'
                  ) -> pd.DataFrame:
        """
        Load energy consumption data from a file.

        This method reads a data file, validates the presence of required columns,
        processes the date column, and ensures the consumption column is numeric.

        Parameters
        ----------
        file_path : str
            Path to the data file (e.g., Excel file).
        date_column : str
            Name of the column containing timestamps or years.
        consumption_column : str
            Name of the column containing consumption values.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the loaded and validated data.

        Raises
        ------
        ValueError
            If required columns are missing or all consumption values are invalid.
        Exception
            For general errors during data loading or processing.
        """
        try:
            # Load the data without setting index
            self.data = pd.read_excel(file_path)

            # Print debug information
            print(f"Total rows loaded: {len(self.data)}")
            print(f"Columns found: {self.data.columns.tolist()}")
            print(f"First few rows of {consumption_column}:\n{self.data[consumption_column].head()}")
            print(f"Data types: {self.data.dtypes}")

            # Verify required columns exist
            if date_column not in self.data.columns:
                raise ValueError(f"Required column {date_column} not found")
            if consumption_column not in self.data.columns:
                raise ValueError(f"Required column {consumption_column} not found")

            # Check and process the date column
            if pd.api.types.is_numeric_dtype(self.data[date_column]):
                print(f"Column '{date_column}' is numeric. Interpreting as year.")
                self.data[date_column] = self.data[date_column].astype(int)  # Keep as year
            else:
                print(f"Column '{date_column}' is non-numeric. Interpreting as datetime.")
                self.data[date_column] = pd.to_datetime(self.data[date_column])

            # Ensure consumption column is numeric and handle any conversion errors
            self.data[consumption_column] = pd.to_numeric(
                self.data[consumption_column].replace(',', '.', regex=True),
                errors='coerce'
            )

            # Print validation info
            valid_entries = self.data[consumption_column].notna().sum()
            total_entries = len(self.data)
            print(f"Valid entries: {valid_entries} out of {total_entries}")

            # Basic data validation
            if self.data[consumption_column].isna().all():
                raise ValueError(f"All values in {consumption_column} are NaN after conversion")

            return self.data

        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def clean_data(self,
                   data: Optional[pd.DataFrame] = None,
                   handle_missing: str = 'interpolate'
                   ) -> pd.DataFrame:
        """
        Clean and preprocess the energy consumption data.

        This method handles missing values using various strategies,
        removes duplicates, and ensures data consistency.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame to clean. If None, uses the internally stored data.
        handle_missing : str, optional
            The strategy to handle missing values:
            - 'interpolate': Fills missing values using linear interpolation.
            - 'drop': Removes rows with missing values.
            - 'zero': Replaces missing values with zero.

        Returns
        -------
        pd.DataFrame
            A cleaned and preprocessed DataFrame.

        Raises
        ------
        ValueError
            If no data is provided or loaded, or an invalid handle_missing method is specified.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided or loaded")
            data = self.data.copy()
        else:
            data = data.copy()

        # Print initial statistics
        print(f"Initial missing values:\n{data.isna().sum()}")

        # Handle missing values
        if handle_missing == 'interpolate':
            # First handle non-numeric columns with forward and backward fill
            object_cols = data.select_dtypes(include=['object']).columns
            if not object_cols.empty:
                data[object_cols] = data[object_cols].ffill().bfill()

            # Then handle numeric columns with interpolation
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                data[numeric_cols] = data[numeric_cols].interpolate(method='linear')
                data[numeric_cols] = data[numeric_cols].ffill().bfill()  # Handle edges

        elif handle_missing == 'drop':
            data = data.dropna()
        elif handle_missing == 'zero':
            data = data.fillna(0)
        else:
            raise ValueError("Invalid handle_missing method")

        # Remove duplicates
        data = data.drop_duplicates()

        # Print final statistics
        print(f"Final missing values:\n{data.isna().sum()}")

        return data

    def validate_data(self, data: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate the data for common issues.

        This method checks the presence of required columns, verifies data types,
        and ensures the data meets basic integrity requirements.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame to validate. If None, uses the internally stored data.

        Returns
        -------
        bool
            True if validation passes, False otherwise.

        Raises
        ------
        ValueError
            If no data is provided or loaded.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided or loaded")
            data = self.data

        # Check for required columns
        required_columns = ['date', 'quantity', 'organizationName']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False

        # Check for data types
        if not pd.api.types.is_numeric_dtype(data['date']) and not pd.api.types.is_datetime64_any_dtype(data['date']):
            print("'date' column must be numeric (year) or datetime type")
            return False

        if not pd.api.types.is_numeric_dtype(data['quantity']):
            print("'quantity' column is not numeric type")
            return False
        
        return True

