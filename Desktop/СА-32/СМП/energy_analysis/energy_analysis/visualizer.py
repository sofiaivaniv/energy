import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple

class EnergyVisualizer:
    """
    A class for visualizing energy consumption data.

    Provides methods for generating various plots to explore
    and understand energy consumption patterns.
    """

    def __init__(self):
        """
        Initialize the EnergyVisualizer.

        Sets the visualization theme using seaborn and initializes
        the data attribute as None.
        """
        sns.set_theme(style="whitegrid")
        self.data = None

    def load_data(self, data: pd.DataFrame):
        """
        Load data for visualization.

        This method takes a pandas DataFrame containing energy consumption data
        and stores it for further visualization.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing energy consumption data, including columns
            such as date and consumption quantity.
        """
        self.data = data.copy()

    def plot_time_series(self, 
                         date_column: str = 'date', 
                         consumption_column: str = 'quantity',
                         title: str = 'Energy Consumption Over Time',
                         figsize: Tuple[int, int] = (12, 6)):
        """
        Plot a time-series graph of energy consumption.

        This method generates a line plot of energy consumption over time,
        showing how consumption varies over the specified date range.

        Parameters
        ----------
        date_column : str, optional
            Name of the column containing dates (default is 'date').
        consumption_column : str, optional
            Name of the column containing consumption values (default is 'quantity').
        title : str, optional
            Title of the plot (default is 'Energy Consumption Over Time').
        figsize : Tuple[int, int], optional
            Figure size in inches (default is (12, 6)).

        Raises
        ------
        ValueError
            If no data has been loaded for visualization.
        """
        if self.data is None:
            raise ValueError("No data loaded for visualization.")

        plt.figure(figsize=figsize)
        sns.lineplot(data=self.data, x=date_column, y=consumption_column, marker='o', color='blue')
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy Consumption', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, 
                       consumption_column: str = 'quantity', 
                       bins: int = 20, 
                       title: str = 'Distribution of Energy Consumption',
                       figsize: Tuple[int, int] = (10, 6)):
        """
        Plot a histogram of energy consumption.

        This method generates a histogram to visualize the distribution of energy
        consumption values, with an optional kernel density estimate (KDE) for smoothness.

        Parameters
        ----------
        consumption_column : str, optional
            Name of the column containing consumption values (default is 'quantity').
        bins : int, optional
            Number of bins for the histogram (default is 20).
        title : str, optional
            Title of the plot (default is 'Distribution of Energy Consumption').
        figsize : Tuple[int, int], optional
            Figure size in inches (default is (10, 6)).

        Raises
        ------
        ValueError
            If no data has been loaded for visualization.
        """
        if self.data is None:
            raise ValueError("No data loaded for visualization.")

        plt.figure(figsize=figsize)
        sns.histplot(self.data[consumption_column].dropna(), bins=bins, kde=True, color='teal')
        plt.title(title, fontsize=16)
        plt.xlabel('Energy Consumption', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, 
                 consumption_column: str = 'quantity', 
                 category_column: Optional[str] = None,
                 title: str = 'Boxplot of Energy Consumption',
                 figsize: Tuple[int, int] = (18, 12)):
        """
        Plot a boxplot of energy consumption.

        This method generates a boxplot to visualize the distribution and potential outliers
        in energy consumption. It can optionally group the data by a categorical column.

        Parameters
        ----------
        consumption_column : str, optional
            Name of the column containing consumption values (default is 'quantity').
        category_column : Optional[str], optional
            Name of the column for categorical grouping (default is None).
        title : str, optional
            Title of the plot (default is 'Boxplot of Energy Consumption').
        figsize : Tuple[int, int], optional
            Figure size in inches (default is (18, 12)).

        Raises
        ------
        ValueError
            If no data has been loaded for visualization.
        """
        if self.data is None:
            raise ValueError("No data loaded for visualization.")

        plt.figure(figsize=figsize)
        if category_column:
            sns.boxplot(
                data=self.data, 
                x=category_column, 
                y=consumption_column, 
                hue=category_column,
                palette='Set3'
            )
        else:
            sns.boxplot(
                data=self.data, 
                y=consumption_column, 
                color='lightblue'  # Fixed color for single data set
        )

        plt.title(title, fontsize=16)
        plt.xlabel(category_column if category_column else 'Category', fontsize=12)
        plt.ylabel('Energy Consumption', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
 
    def plot_heatmap(self, 
                    date_column: str = 'date', 
                    consumption_column: str = 'quantity', 
                    aggfunc: str = 'mean',
                    title: str = 'Heatmap of Energy Consumption Patterns'):
        """
        Plot a heatmap to visualize aggregated energy consumption patterns for months.

        This method creates a heatmap based on the specified date and consumption columns.
        It aggregates the consumption data by year and month or year if the date column
        is numeric.

        Parameters
        ----------
        date_column : str, optional
            Name of the column containing date or year data (default is 'date').
        consumption_column : str, optional
            Name of the column containing consumption values (default is 'quantity').
        aggfunc : str, optional
            Aggregation function to use, such as 'mean', 'sum', etc. (default is 'mean').
        title : str, optional
            Title of the plot (default is 'Heatmap of Energy Consumption Patterns').

        Raises
        ------
        ValueError
            If no data has been loaded for visualization.
        """
        if self.data is None:
            raise ValueError("No data loaded for visualization")

        # Check if 'date' column is datetime
        if pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            # Extract year and month if 'date' is datetime
            self.data['year'] = self.data[date_column].dt.year
            self.data['month'] = self.data[date_column].dt.month

            # Create pivot table for months
            pivot_table_month = self.data.pivot_table(
                values=consumption_column, 
                index=['year', 'month'],  # Year and month as indices
                aggfunc=aggfunc
            )
            data_for_heatmap_month = pivot_table_month  # Heatmap data for months

            # Plot heatmap for months
            plt.figure(figsize=(12, 6))
            sns.heatmap(data_for_heatmap_month,  # Data for the heatmap
                        annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Consumption'})
            plt.title(f'{title} - Month Based', fontsize=16)
            plt.xlabel('Year and Month', fontsize=12)
            plt.ylabel('Year', fontsize=12)
            plt.tight_layout()
            plt.show()

        elif pd.api.types.is_numeric_dtype(self.data[date_column]):
            # If date column is numeric, use it as the year
            self.data['year'] = self.data[date_column]
            pivot_table = self.data.pivot_table(
                values=consumption_column, 
                index='year',  # Use year as index
                aggfunc=aggfunc
            )
            data_for_heatmap = pivot_table
            # Plot heatmap for years
            plt.figure(figsize=(12, 6))
            sns.heatmap(data_for_heatmap,  # Data for the heatmap
                        annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Consumption'})
            plt.title(f'{title} - Year Based', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Year', fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Invalid column type for {date_column}.")


