import tkinter as tk
from tkinter import filedialog, simpledialog
import subprocess
from functions import *

# Function to call the hydrological model simulation
def run_hydrological_model(temp_path, 
                           precip_path, 
                           dem_path, 
                           number_of_bands,
                           shapefile_outpath,
                           gdf_outpath,
                           results_outpath
                           ):
    import pandas as pd
    import rasterio
    from pyproj import CRS
    import geopandas as gpd
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt

    
    #number_of_bands = 5

    # File paths
    #temp_path = 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Tobs.txt'
    #precip_path = 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Pobs.txt'
    #dem_path = 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/DEM_METRE_1023144.tif'
    #shapefile_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/shapefiles/elevation_bands.shp'
    #gdf_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/shapefiles/elevation_bands_with_area.shp'
    #results_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Results/adjusted_area_output_swe.csv'

    # Load data
    temperature = pd.read_csv(temp_path, sep='\t', parse_dates=True, index_col='Date')
    precipitation = pd.read_csv(precip_path, sep='\t', parse_dates=True, index_col='Date')

    # Open the DEM file and compute elevation bands
    with rasterio.open(dem_path) as src:
        # Read the data for band 1
        dem_data = src.read(1)
        
        # Retrieve the nodata value from the source metadata
        nodata = src.nodata
        
        # Mask out the nodata values by setting them to NaN for further processing
        dem_data[dem_data == nodata] = np.nan
        
        # Find the valid min and max elevation values excluding NoData values
        min_elevation = np.nanmin(dem_data)
        max_elevation = np.nanmax(dem_data)
        
        # Divide the valid elevation range the number of bands specified
        elevation_ranges = np.linspace(min_elevation, max_elevation, number_of_bands+1)

    average_elevation = []    
    for i in range(len(elevation_ranges) - 1):  # subtract 1 because there are n-1 bands
        band_avg_elev = (elevation_ranges[i] + elevation_ranges[i + 1]) / 2
        average_elevation.append(band_avg_elev)

    average_elevation = np.array(average_elevation)

    interval = round(average_elevation[1] - average_elevation[0])
    interval = str(interval + 1)

    min_elevation_dem = elevation_ranges[0]
    min_elevation_dem = str(min_elevation_dem)

    command = [
        'gdal_contour',
        '-p',  # Create polygons instead of lines
        '-amax', 'ELEV_MAX',
        '-amin', 'ELEV_MIN',
        '-b', '1',  # Band 1
        '-i', interval,  # Interval between contours
        '-off', min_elevation_dem,
        '-f', 'GPKG',  # Output format
        dem_path,  # Input DEM
        shapefile_outpath  # Output shapefile
    ]


    subprocess.run(command, check=True)
    gdf = gpd.read_file(shapefile_outpath)

    # Assume your data is initially in WGS84 (EPSG:4326)
    gdf = gdf.set_crs('EPSG:4326')

    # Find the appropriate UTM zone for your dataset
    utm_zone = CRS(gdf.estimate_utm_crs()).to_string()

    # Project the GeoDataFrame to the UTM CRS
    gdf_utm = gdf.to_crs(utm_zone)

    # Now calculate the area in square meters
    gdf_utm['area_m2'] = gdf_utm['geometry'].area

    # Save the GeoDataFrame with the new 'area' column to a shapefile
    gdf_utm.to_file(gdf_outpath)

    area_values = gdf_utm['area_m2'].values

    elevation_band_areas = np.array(area_values)

    # Calculate the total area of the basin (sum of all band areas)
    total_basin_area = elevation_band_areas.sum()

    # Calculate fractional area for each elevation band
    area_fractions = elevation_band_areas / total_basin_area

    # Initialize SWE storage, one entry for each elevation band
    swe_storage = np.zeros(len(elevation_ranges) - 1)
    # Simulation loop
    results = []
    for date, row in temperature.iterrows():
        daily_temp = row['1023144']  # Make sure to use the correct column name
        daily_precip = precipitation.loc[date]['1023144']  # Make sure to use the correct column name
        for i in range(len(elevation_ranges)-1):
            band_avg_elev = (elevation_ranges[i] + elevation_ranges[i+1]) / 2
            adjusted_temp = adjust_temp(daily_temp, min_elevation, band_avg_elev)
            
            # Adjust precipitation for the fractional area of the elevation band
            adjusted_precip = daily_precip * area_fractions[i]
            
            a = snow_accumulation(adjusted_precip, adjusted_temp)  # Use adjusted_temp here
            m = snow_melt(adjusted_temp)  # And here
            # Accumulate or deplete the SWE storage for this band
            swe_storage[i] += a - m
            swe_storage[i] = max(swe_storage[i], 0)  # SWE cannot be negative
            melt_factor = calculate_melt_factor(adjusted_temp)
            # Append the current SWE, not just the change
            results.append((date, round(band_avg_elev, 1), round(swe_storage[i], 2),round(melt_factor,2),round(adjusted_temp,2)))

    # Save results
    result_df = pd.DataFrame(results, columns=['Date', 'Elevation', 'SWE', 'Melt Rate', 'Adjusted Temp'])

    plot_swe_all_elevations(result_df)
    plot_swe_all_elevations_with_mean(result_df)
    plot_swe_all_elevations_range(result_df, '2013-08-01', '2014-08-01')
    plot_swe_all_elevations_range_with_mean(result_df,'2013-08-01', '2014-08-01')
    plot_swe_all_elevations_lumped(result_df)
    plot_swe_all_elevations_range_lumped(result_df,'2013-08-01', '2014-08-01')
    result_df.to_csv(results_outpath, index=False)
    
    pass

# Function to handle file browsing and path setting
def browse_file(entry):
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)

# Function to initiate the model run
def run_script():
    temp_path = entry_temp.get()
    precip_path = entry_precip.get()
    dem_path = entry_dem.get()
    results_outpath = entry_results.get()
    number_of_bands = simpledialog.askinteger("Input", "Enter the number of elevation bands:",
                                              parent=root, minvalue=1, maxvalue=100)
    shapefile_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/shapefiles/elevation_bands1.shp'
    gdf_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/shapefiles/elevation_bands_with_area1.shp'
    #results_outpath = 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Results/adjusted_area_output_swe1.csv'
    
    run_hydrological_model(temp_path, precip_path, dem_path, number_of_bands, 
                           shapefile_outpath, gdf_outpath, results_outpath)

# Setup the main window
root = tk.Tk()
root.title("Hydrological Model Simulation")

# Temperature file path entry
label_temp = tk.Label(root, text="Temperature Data File:")
label_temp.pack()
entry_temp = tk.Entry(root, width=50)
entry_temp.insert(0, 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Tobs.txt')  # Default path
entry_temp.pack()
browse_button_temp = tk.Button(root, text="Browse", command=lambda: browse_file(entry_temp))
browse_button_temp.pack()

# Precipitation file path entry
label_precip = tk.Label(root, text="Precipitation Data File:")
label_precip.pack()
entry_precip = tk.Entry(root, width=50)
entry_precip.insert(0, 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Pobs.txt')  # Default path
entry_precip.pack()
browse_button_precip = tk.Button(root, text="Browse", command=lambda: browse_file(entry_precip))
browse_button_precip.pack()

# DEM file path entry
label_dem = tk.Label(root, text="DEM File:")
label_dem.pack()
entry_dem = tk.Entry(root, width=50)
entry_dem.insert(0, 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/DEM_METRE_1023144.tif')  # Default path
entry_dem.pack()
browse_button_dem = tk.Button(root, text="Browse", command=lambda: browse_file(entry_dem))
browse_button_dem.pack()

# Results file path entry
label_results = tk.Label(root, text="Results File Path:")
label_results.pack()
entry_results = tk.Entry(root, width=50)
entry_results.insert(0, 'D:/Masters/Classes/ENCI_619/Model_scripts/Data/Results/adjusted_area_output_swe1.csv')  # Default path
entry_results.pack()
browse_button_results = tk.Button(root, text="Browse", command=lambda: browse_file(entry_results))
browse_button_results.pack()

# Button to run the simulation
run_button = tk.Button(root, text="Run Simulation", command=run_script)
run_button.pack()

# Start the GUI event loop
root.mainloop()
