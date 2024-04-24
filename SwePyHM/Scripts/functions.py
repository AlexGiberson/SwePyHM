import pandas as pd
from pyproj import CRS
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Constants
LAPSE_RATE = 6.5  # Temperature lapse rate in °C per 1km
FREEZING_TEMP = 0  # Freezing temperature Tf
BASE_MELT_FACTOR = 1  # Melt factor k (Based on the 1-4 mm/°C/day)

def calculate_melt_factor(temp, base_temp=FREEZING_TEMP, base_melt_factor=BASE_MELT_FACTOR):
    if temp <= base_temp:
        return 0
    else:
        # For example, melt factor increases by 0.1 for each degree above freezing

        return base_melt_factor + 0.18 * (temp - base_temp)

# Define functions for temperature adjustment, snow accumulation, and snow melt
def adjust_temp(obs_temp, base_elev, target_elev):
    return obs_temp - LAPSE_RATE / 1000 * (target_elev - base_elev)

def snow_accumulation(precip, adjusted_temp):
    return precip if adjusted_temp < FREEZING_TEMP else 0

def snow_melt(adjusted_temp):
    melt_factor = calculate_melt_factor(adjusted_temp)
    return melt_factor * (adjusted_temp - FREEZING_TEMP) if adjusted_temp > FREEZING_TEMP else 0

def plot_swe_all_elevations(df):
    plt.figure(figsize=(12, 6))
    elevation_groups = df.groupby('Elevation')
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], label=f'{name} m')

    plt.title('SWE Over Time for All Elevation Bands')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend(title='Elevation', loc='upper right')
    plt.grid(True)
    plt.show()

def plot_swe_all_elevations_with_mean(df):
    plt.figure(figsize=(12, 6))
    elevation_groups = df.groupby('Elevation')
    color = 'gray'  # Single color for all elevation bands

    # Plot each elevation band using the same color
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], color=color, label='_nolegend_')  # Exclude from legend

    # Calculate the mean SWE for each date across all elevations
    mean_swe = df.groupby('Date')['SWE'].mean()

    # Plot the mean SWE with a distinguished color and line style
    plt.plot(mean_swe.index, mean_swe, label='Mean SWE', color='red', linestyle='--')

    # Add a dummy line for 'Individual Elevation Bands' for the legend
    plt.plot([], [], color='gray', label='Individual Elevation Bands')

    # Handle legend
    plt.legend(loc='upper right')

    plt.title('SWE Over Time for All Elevation Bands')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.grid(True)
    plt.show()

def plot_swe_all_elevations_range(df, start_date='2010-08-01', end_date='2013-08-30'):
    # Convert 'Date' to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame for the specified date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]

    # Plotting
    plt.figure(figsize=(12, 6))
    elevation_groups = df_filtered.groupby('Elevation')
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], label=f'{name} m')

    plt.title(f'SWE Over Time for All Elevation Bands ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend(title='Elevation', loc='upper right')
    plt.grid(True)
    plt.show()

def plot_swe_all_elevations_range_with_mean(df, start_date='2010-08-01', end_date='2013-08-30'):
    # Convert 'Date' to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame for the specified date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]

    # Plotting
    plt.figure(figsize=(12, 6))
    elevation_groups = df_filtered.groupby('Elevation')
    
    # Single color for individual elevation bands
    color = 'gray'

    # Plot each elevation group with the same gray color and exclude them from the legend
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], color=color, label='_nolegend_')

    # Calculate the mean SWE for each date within the filtered range
    mean_swe = df_filtered.groupby('Date')['SWE'].mean()

    # Plot the mean SWE with a distinguished color and line style
    plt.plot(mean_swe.index, mean_swe, label='Mean SWE', color='red', linestyle='-')

    # Add a dummy line for 'Individual Elevation Bands' for the legend
    plt.plot([], [], color=color, label='Individual Elevation Bands')

    plt.title(f'SWE Over Time for All Elevation Bands ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def plot_swe_all_elevations_lumped(df):
    lumped_model_df = pd.read_csv('D:/Masters/Classes/ENCI_619/Model_scripts/Data/Results/lumped_HYPE_results.csv')
    # Ensure 'Date' columns are in datetime format for both DataFrames
    lumped_model_df['Date'] = pd.to_datetime(lumped_model_df['Date'])
    
    plt.figure(figsize=(12, 6))
    elevation_groups = df.groupby('Elevation')
    color = 'gray'  # Single color for all elevation bands

    # Plot each elevation band using the same color
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], color=color, label='_nolegend_')  # Exclude from legend

    # Calculate the mean SWE for each date across all elevations
    mean_swe = df.groupby('Date')['SWE'].mean()

    # Plot the mean SWE
    plt.plot(mean_swe.index, mean_swe, label='Mean SWE', color='red', linestyle='--')

    # Plot the lumped model results
    plt.plot(lumped_model_df['Date'], lumped_model_df['Lumped_Model_SWE'], label='Lumped Model Results', color='blue', linestyle='-.')

    # Add a dummy line for 'Individual Elevation Bands' for the legend
    plt.plot([], [], color=color, label='Individual Elevation Bands')

    plt.title('SWE Over Time for All Elevation Bands')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_swe_all_elevations_range_lumped(df, start_date='2010-08-01', end_date='2013-08-30'):
    lumped_model_df = pd.read_csv('D:/Masters/Classes/ENCI_619/Model_scripts/Data/Results/lumped_HYPE_results.csv')
    # Ensure 'Date' columns are in datetime format for both DataFrames
    lumped_model_df['Date'] = pd.to_datetime(lumped_model_df['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame for the specified date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]

    # Filter the lumped model results for the same date range
    lumped_mask = (lumped_model_df['Date'] >= start_date) & (lumped_model_df['Date'] <= end_date)
    lumped_filtered = lumped_model_df.loc[lumped_mask]

    # Plotting
    plt.figure(figsize=(12, 6))
    elevation_groups = df_filtered.groupby('Elevation')
    for name, group in elevation_groups:
        plt.plot(group['Date'], group['SWE'], color='gray', label='_nolegend_')

    # Calculate the mean SWE for each date within the filtered range
    mean_swe = df_filtered.groupby('Date')['SWE'].mean()

    # Plot the mean SWE
    plt.plot(mean_swe.index, mean_swe, label='Mean SWE', color='red', linestyle='--')

    # Plot the filtered lumped model results
    plt.plot(lumped_filtered['Date'], lumped_filtered['Lumped_Model_SWE'], label='Lumped Model Results', color='blue', linestyle='-.')

    # Add a dummy line for 'Individual Elevation Bands' for the legend
    plt.plot([], [], color='gray', label='Individual Elevation Bands')

    plt.title(f'SWE Over Time for All Elevation Bands ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('SWE (mm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()





