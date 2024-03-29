#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:20:11 2023

@author: ozanbaris

This script is for the RC and RQ identification of houses from ecobee dataset using curve_fit. 
Cooling season is computed for each house by checking the first time cooling turned on and the last time it was on. 
"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, to_hex


# File names
file_names = ['Mar_clean.nc','Apr_clean.nc', 'May_clean.nc', 'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc', 'Sep_clean.nc','Oct_clean.nc','Nov_clean.nc']



# Initialize an empty DataFrame
df = pd.DataFrame()

# Load each file and append it to df
for file_name in file_names:
    # Load the .nc file
    data = xr.open_dataset(file_name)

    # Convert it to a DataFrame
    temp_df = data.to_dataframe()

    # Reset the index
    temp_df = temp_df.reset_index()

    # Append the data to df
    df = pd.concat([df, temp_df], ignore_index=True)

# Now df is a pandas DataFrame and you can perform any operation you want
print(df.head())

#%%
house_data = {}

unique_house_ids = df['id'].unique()

for house_id in unique_house_ids:
    house_data[house_id] = df[df['id'] == house_id]
#%%# Prepare dictionaries to store cooling and heating season data for each house
cooling_season_dict = {}

for house_id, single_house_data in house_data.items():
    # Identify when the HVAC system is in cooling or heating mode
    single_house_data.set_index('time', inplace=True)

    single_house_data['Cooling_Mode'] = single_house_data['CoolingEquipmentStage1_RunTime'].notna()


    # Identify the periods of cooling and heating
    cooling_start = single_house_data.loc[single_house_data['Cooling_Mode']].index.min()
    cooling_end = single_house_data.loc[single_house_data['Cooling_Mode']].index.max()


    # Extract cooling and heating season data
    cooling_season_data = single_house_data.loc[cooling_start : cooling_end]


    # Store in dictionaries
    cooling_season_dict[house_id] = cooling_season_data



#%%
# Create empty dictionaries for each category
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}


#for house_id, data in house_data.items():
for house_id, data in cooling_season_dict.items():
    # Count the number of non-empty temperature sensors
    num_sensors = sum([1 for i in range(1, 6) if not np.isnan(data[f'RemoteSensor{i}_Temperature']).all()])
    
    # Add to relevant dictionary
    if num_sensors == 1:
        one_houses[house_id] = data
    elif num_sensors == 2:
        two_houses[house_id] = data
    elif num_sensors == 3:
        three_houses[house_id] = data
    elif num_sensors == 4:
        four_houses[house_id] = data
    elif num_sensors == 5:
        five_houses[house_id] = data
        
print(f"Number of houses with 1 sensor: {len(one_houses)}")
print(f"Number of houses with 2 sensors: {len(two_houses)}")
print(f"Number of houses with 3 sensors: {len(three_houses)}")
print(f"Number of houses with 4 sensors: {len(four_houses)}")
print(f"Number of houses with 5 sensors: {len(five_houses)}")

#%%


for house_id, single_house_data in cooling_season_dict.items():
    single_house_data.reset_index(inplace=True)

# Prepare dictionary to store valid periods for each house and each sensor
valid_periods_dict = {}

# For each sensor, check if its temperature is more than 20 F above Indoor_CoolSetpoint
sensor_columns = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']

for house_id, single_house_data in five_houses.items():
    # Ensure data is sorted by time
    single_house_data.sort_values(by='time', inplace=True)

    # Initialize house entry in valid_periods_dict
    valid_periods_dict[house_id] = {}

    for sensor in sensor_columns:
        # Identify when HVAC is off (Cooling and Heating)
        single_house_data['HVAC_Off'] = single_house_data['CoolingEquipmentStage1_RunTime'].isna() & single_house_data['HeatingEquipmentStage1_RunTime'].isna()

        # Identify groups of rows where HVAC is off
        single_house_data['Off_Period'] = (single_house_data['HVAC_Off'] != single_house_data['HVAC_Off'].shift()).cumsum()

        # Filter periods that are at least 2 hours long and Outdoor_Temperature is higher than the sensor temperature
        # Also filter for times between 10 AM and 5 PM and where sensor temperature increased by at least 2 F
        valid_periods = single_house_data[single_house_data['HVAC_Off']].groupby('Off_Period').filter(
            lambda x: (len(x) >= 12)
                      and (x['Outdoor_Temperature'] > x[sensor]).all()
                      and (x['time'].dt.hour >= 10).all()
                      and (x['time'].dt.hour < 17).all()
                      and ((x[sensor].max() - x[sensor].min()) >= 2))

        # Add to the dictionary
        valid_periods_dict[house_id][sensor] = valid_periods



#%%

hist_val_periods=[]
for house_id, house_periods in valid_periods_dict.items():
    num_valid_periods = sum(len(periods) for periods in house_periods.values())
    hist_val_periods.append(num_valid_periods)

# Create a histogram of the number of valid periods
plt.figure(figsize=(10, 6))
plt.hist(hist_val_periods, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Number of Periods')
plt.ylabel('Number of Houses')
plt.title('Histogram of Number of Filtered Free Floating Periods')
plt.show()


#%%

total_houses = len(valid_periods_dict)

# Count houses with no valid periods
houses_with_zero_valid_periods = sum(1 for house_periods in valid_periods_dict.values() if not any(not periods.empty for periods in house_periods.values()))


# Compute the total length of the dataset for all house-sensor pairs
total_length = sum(sum(len(periods) for periods in house_periods.values()) for house_periods in valid_periods_dict.values())

# Compute average and std deviation of the number of valid periods
num_valid_periods = [sum(len(periods) for periods in house_periods.values()) for house_periods in valid_periods_dict.values()]
average_num_valid_periods = np.mean(num_valid_periods)
std_dev_num_valid_periods = np.std(num_valid_periods)

# Compute average and std deviation of the length of the valid periods
length_valid_periods = [len(periods) for house_periods in valid_periods_dict.values() for periods in house_periods.values()]
average_length_valid_periods = np.mean(length_valid_periods)
std_dev_length_valid_periods = np.std(length_valid_periods)


# Print the computed stats
print("Total number of houses:", total_houses)
print("Number of houses with zero valid periods:", houses_with_zero_valid_periods)
print("Total length of the dataset combined for all house-sensor pairs:", total_length)
print("Average number of valid periods for all house-sensor combinations in total:", average_num_valid_periods)
print("Std deviation of the number of valid periods for all house-sensor combinations in total:", std_dev_num_valid_periods)
print("Average length of the valid periods for all house-sensor combinations in total:", average_length_valid_periods)
print("Std deviation of the length of the valid periods for all house-sensor combinations in total:", std_dev_length_valid_periods)

#%%


# Dictionary to store RC and RQ for each house and each sensor
RC_values = {}
RQ_values = {}
errors = {}

# Function to remove outliers
def remove_outliers(data):
    if len(data) > 1:
        data = np.array(data)  # Convert list of numpy arrays to numpy array
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        data_no_outlier = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
        return data_no_outlier.tolist()
    else:
        return data

# Iterate over houses and sensors and valid periods
for house_id, sensors in valid_periods_dict.items():
    RC_values[house_id] = {}
    RQ_values[house_id] = {}
    errors[house_id] = {}
    
    for sensor, periods in sensors.items():
        # Initialize list to store RC values, RQ values and errors for this sensor
        RC_values_sensor = []
        RQ_values_sensor = []
        errors_sensor = []
        
        for period, data in periods.groupby('Off_Period'):
            # Define the function to fit, now with RQ and RC as parameters
            def model_func(t, RQ, RC):
                return (T_diff0- RQ)* np.exp(-t/RC) 
            
            # Prepare input data for curve fitting
            # Here, we assume that the timestep is 5 min (or 1/12 hour)
            t = np.arange(len(data)) / 12

            T_diff0 = data[sensor].iloc[0] - data['Outdoor_Temperature'].iloc[0]
            T_diff_t = data[sensor] - data['Outdoor_Temperature'].mean()

            # Perform curve fitting with initial guesses for RQ and RC
            popt, _ = curve_fit(model_func, t, T_diff_t, p0=[1, 20], bounds=(0, 200))
            RQ_values_sensor.append(popt[0])
            RC_values_sensor.append(popt[1])

            # Make predictions
            T_diff_pred = model_func(t, *popt)
            # Calculate error
            error = np.sqrt(np.mean((T_diff_t - T_diff_pred) ** 2))
            errors_sensor.append(error)

        # Remove outliers from RC_values_sensor, RQ_values_sensor and errors_sensor
        RC_values_sensor = remove_outliers(RC_values_sensor)
        RQ_values_sensor = remove_outliers(RQ_values_sensor)
        errors_sensor = remove_outliers(errors_sensor)

        # Store average RC value, average RQ and average error for this sensor
        RQ_values[house_id][sensor] = np.mean(RQ_values_sensor)
        RC_values[house_id][sensor] = np.mean(RC_values_sensor)
        errors[house_id][sensor] = np.mean(errors_sensor)


#%%
sensor_names = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']
def outlier_removal(values_dict, sensor_names):
    def remove_outliers_std(data):
        # Calculate mean and standard deviation
        data_mean = np.mean(data)
        data_std = np.std(data)
    
        # Calculate the bounds for 2 standard deviations
        lower_bound = data_mean - 2 * data_std
        upper_bound = data_mean + 2 * data_std
    
        # Filter values that lie within 2 standard deviations
        data_out = [x for x in data if lower_bound <= x <= upper_bound]
    
        return data_out
    
    # For each sensor, gather all values across all houses
    values_sensors = {sensor: [values_dict[house_id][sensor] for house_id in values_dict if sensor in values_dict[house_id] and np.isnan(values_dict[house_id][sensor]) == False] for sensor in sensor_names}

    # Count of non-NaN values before outlier removal
    count_before = sum([len(values) for values in values_sensors.values()])
    print(f"Count of non-NaN values before outlier removal: {count_before}")

    # Remove outliers from the values for each sensor using standard deviations
    values_sensors_no_outliers = {sensor: remove_outliers_std(values_sensors[sensor]) for sensor in sensor_names}

    # Initialize the new dictionary
    values_dict_no_outliers = {house_id: {sensor: np.nan for sensor in sensor_names} for house_id in values_dict}

    # Then assign the outlier-removed data back to the new dictionary
    for house_id in values_dict:
        for sensor in sensor_names:
            if sensor in values_sensors_no_outliers and values_dict[house_id][sensor] in values_sensors_no_outliers[sensor]:
                values_dict_no_outliers[house_id][sensor] = values_dict[house_id][sensor]

    # Gather all filtered values into a single list
    values_no_outliers = [value for sensor_values in values_dict_no_outliers.values() for value in sensor_values.values() if not np.isnan(value)]

    # Count of non-NaN values after outlier removal
    count_after = len(values_no_outliers)
    print(f"Count of non-NaN values after outlier removal: {count_after}")

    return values_dict_no_outliers, values_sensors_no_outliers, values_no_outliers

RC_values_no_outliers, RC_values_sensors_no_outliers, RC_values_no_outliers_list = outlier_removal(RC_values, sensor_names)
RQ_values_no_outliers, RQ_values_sensors_no_outliers, RQ_values_no_outliers_list = outlier_removal(RQ_values, sensor_names)

#%%
def filter_values_with_errors(values_dict, errors_dict):
    # Create copies of the input dictionaries to avoid modifying the originals
    values_dict_filtered = values_dict.copy()
    errors_dict_filtered = errors_dict.copy()

    # Iterate over the houses in the values dictionary
    for house_id, sensors in values_dict_filtered.items():
        # Check if the house is also in the errors dictionary
        if house_id in errors_dict_filtered:
            # Iterate over the sensor data for the house
            for sensor, value in sensors.items():
                # If the sensor is also in the errors dictionary for the house, check the error
                if sensor in errors_dict_filtered[house_id] and errors_dict_filtered[house_id][sensor] > 1:
                    # If the error is larger than 1, set the value to NaN
                    values_dict_filtered[house_id][sensor] = np.nan

    return values_dict_filtered

final_RC_values=filter_values_with_errors(RC_values_no_outliers, errors)
final_RQ_values=filter_values_with_errors(RQ_values_no_outliers, errors)
#%%

# Read the CSV files into DataFrames
final_df_RC = pd.read_csv('BuildSys/final_RC_values.csv', index_col=0)
final_df_RQ = pd.read_csv('BuildSys/final_RQ_values.csv', index_col=0)

def dataframe_to_dict(df):
    """Converts a DataFrame to a nested dictionary."""
    nested_dict = df.transpose().to_dict()
    for house_id, sensors in nested_dict.items():
        for sensor, value in sensors.items():
            if np.isnan(value):
                nested_dict[house_id][sensor] = None
            else:
                nested_dict[house_id][sensor] = float(value)
    return nested_dict

# Convert the DataFrames back to dictionary form
final_RC_values = dataframe_to_dict(final_df_RC)
final_RQ_values = dataframe_to_dict(final_df_RQ)
#%%
# Extract the house_ids from the final_RC_values dictionary
house_ids_from_final_RC = list(final_RC_values.keys())

# Initialize the state_dict
state_dict = {}


for house_id in house_ids_from_final_RC:
    # Filter the dataframe for the current house_id
    single_house_data = df[df['id'] == house_id]
    
    # Get unique non-na states for the current house
    state_values = single_house_data['State'].dropna().unique()
    
    # Initialize the state for the current house as 'Unknown'
    assigned_state = 'Unknown'
    for state in state_values:
        if state != '':
            assigned_state = state
            break
    
    # Add the state to the state_dict
    state_dict[house_id] = assigned_state

print(state_dict)

#%%

# Function to darken a color
def darken(color, amount=0.1):
    # Convert color to RGB
    rgb = to_rgb(color)
    
    # Reduce each of the RGB values
    rgb = [x * (1 - amount) for x in rgb]
    
    # Convert RGB back to hex
    color = to_hex(rgb)
    
    return color

# Darken each color in state_colors
state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}
state_colors_dark = {state: darken(color, 0.1) for state, color in state_colors.items()}

state_dict = {}


for house_id, single_house_data in five_houses.items():
    # Get the cooling data from the cooling_season_dict
    # Save the state for each house
    state_values = single_house_data['State'].dropna().unique()
    assigned_state = 'Unknown'
    for state in state_values:
        if state != '':
            assigned_state = state
            break
    state_dict[house_id] = assigned_state
    

#%%

# Darken each color in state_colors
state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}
state_colors_dark = {state: darken(color, 0.1) for state, color in state_colors.items()}

def plot_data(df, var_name, title, unit, thermostat_df, filename='plot.pdf'):
    # Set the overall font size to 26
    plt.rcParams.update({'font.size': 26})

    # Create a gridspec
    gs = plt.GridSpec(2, 1, height_ratios=[1, 5], hspace=0)

    # Create figure
    fig = plt.figure(figsize=(15, 15))

    # Map house_id to states and add as a new column
    df['mapped_state'] = df['house_id'].map(state_dict)
    

    # Get the ordered list of unique house_ids from df
    ordered_house_ids = df['house_id'].unique()

    # Convert house_id in df into a categorical variable with its current order
    df['house_id'] = pd.Categorical(df['house_id'], categories=ordered_house_ids, ordered=True)

    # Create the histogram in the top subplot
    ax0 = plt.subplot(gs[0])
    # Plot histograms for each state
    sns.histplot(data=df, x=var_name, hue='mapped_state', palette=state_colors_dark, multiple="stack", bins=50, edgecolor=".3", linewidth=.5, ax=ax0, legend=False)
    ax0.set_ylabel('Count')  # Update y-axis label
    ax0.set_xlabel('')  # Remove x-axis label

    # Create the boxplot in the bottom subplot
    ax1 = plt.subplot(gs[1], sharex=ax0)  # share x-axis with the histogram
    bp = sns.boxplot(y='house_id', x=var_name, data=df, hue='mapped_state', palette=state_colors, dodge=False, ax=ax1)  # save the axis to bp
    # Filter out house_ids not in df
    thermostat_df = thermostat_df[thermostat_df['house_id'].isin(ordered_house_ids)]

    # Convert house_id in thermostat_df into a categorical variable with the same order as in df
    thermostat_df['house_id'] = pd.Categorical(thermostat_df['house_id'], categories=ordered_house_ids, ordered=True)

    # Now, you can sort the thermostat_df by 'house_id' in the same order as in df:
    thermostat_df = thermostat_df.sort_values('house_id')
    
    # You can add a rank column to thermostat_df, just like in df
    thermostat_df['rank'] = thermostat_df['house_id'].cat.codes

    # Plot markers over the boxplot
    ax1.scatter(thermostat_df['Thermostat_Temperature'], thermostat_df['rank'], color='cyan', marker="D", s=100)


    # Customize labels and legend
    ax1.set_xlabel(f'{title} Value ({unit})')  # Update x-axis label
    ax1.set_ylabel('House Rank')  # Remove y-axis label
    ax1.legend(title='State', bbox_to_anchor=(1, 0), loc='lower right')  # Place legend at the lower right end
    # Customize ytick labels
    yticks_locs = ax1.get_yticks()  # get current ytick locations
    new_yticks = [i+1 if i%10==0 else '' for i in range(len(yticks_locs))]  # show every 10th house number, starting from 1
    new_yticks[0] = ''  # Remove the tick on the first value of the House Number axis
    ax1.set_yticks(yticks_locs)  # set new ytick locations
    ax1.set_yticklabels(new_yticks)  # set new ytick labels
    
    plt.tight_layout()  # Make sure everything fits
    
    
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%%
def compute_boxplot_data(final_values, sensor_names, var_name, state_dict):
    # Convert the final_values dict to a DataFrame
    df_values = pd.DataFrame(final_values).T
    df_values.reset_index(inplace=True)
    df_values.rename(columns={'index': 'house_id'}, inplace=True)

    # Melt the DataFrame to a long format
    boxplot_data = df_values.melt(id_vars='house_id', var_name='sensor', value_name=var_name)
    boxplot_data.dropna(subset=[var_name], inplace=True)
    
    # Sort the houses by the maximum value of var_name
    sorted_houses = boxplot_data.groupby("house_id")[var_name].max().sort_values(ascending=False).index
    boxplot_data["house_id"] = boxplot_data["house_id"].astype("category")
    boxplot_data["house_id"].cat.set_categories(sorted_houses, inplace=True)
    
    # Map the house_id to the state
    boxplot_data['state'] = boxplot_data['house_id'].map(state_dict)
    
    # Set the font size for the plot
    plt.rcParams.update({'font.size': 26})
    
    # Compute the range of var_name for each house
    var_range = boxplot_data.groupby("house_id")[var_name].max() - boxplot_data.groupby("house_id")[var_name].min()
    
    # Rank the houses by the range of var_name
    sorted_houses = var_range.sort_values(ascending=True).index
    rank_mapping = {house_id: i+1 for i, house_id in enumerate(sorted_houses)}
    
    # Assign ranks, giving a default value (max rank + 1) to houses not present in rank_mapping
    max_rank = len(rank_mapping)
    boxplot_data['rank'] = boxplot_data['house_id'].map(lambda x: rank_mapping.get(x, max_rank+1))
    
    # Convert rank to integer
    boxplot_data['rank'] = boxplot_data['rank'].astype(int)
    
    # Sort the data by the rank
    boxplot_data.sort_values('rank', inplace=True)

    # Create the thermostat_data DataFrame
    thermostat_data = pd.DataFrame({house_id: [final_values[house_id]['Thermostat_Temperature']] for house_id in final_values}).T
    thermostat_data.reset_index(inplace=True)
    thermostat_data.rename(columns={'index': 'house_id', 0: 'Thermostat_Temperature'}, inplace=True)
    

    return boxplot_data, thermostat_data


 #%%
sensor_names = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']
boxplot_data_RC, thermostat_data_RC=compute_boxplot_data(final_RC_values,sensor_names,'RC', state_dict)

plot_data(boxplot_data_RC, 'RC', 'RC', 'h', thermostat_data_RC, filename="RC_cooling.pdf")
 #%%
boxplot_data_RQ, thermostat_data_RQ=compute_boxplot_data(final_RQ_values,sensor_names,'RQ', state_dict) 
plot_data(boxplot_data_RQ, 'RQ', 'RQ', 'F', thermostat_data_RQ, filename="RQ_cooling.pdf")


  #%%
def compute_statistics(boxplot_data, value_name):

    # Average value for all houses combined with its standard deviation
    average_all = boxplot_data[value_name].mean()
    std_dev_all = boxplot_data[value_name].std()
    print(f"Average {value_name} value for all houses: {average_all}")
    print(f"Standard deviation of {value_name} values for all houses: {std_dev_all}")

    # Maximum differences in each house and statistical results of those differences
    boxplot_data['max_difference'] = boxplot_data.groupby('house_id')[value_name].transform(lambda x: x.max() - x.min())
    difference_counts = boxplot_data['max_difference'].value_counts()

    # Compute probabilities for differences
    for diff in [5, 10, 15]:
        probability = (boxplot_data['max_difference'] >= diff).mean()
        print(f"\nProbability of having an {value_name} value in the same house {diff} more than the other {value_name} value: {probability}")

    # Compute likelihood of maximum difference within house
    sorted_diff = np.sort(boxplot_data['max_difference'].unique())
    median_diff = sorted_diff[len(sorted_diff) // 2]

    print(f"\nMore than 50% of the time, there will be a difference of at least {median_diff} in the {value_name} values of the sensors in the same house.")

    # Calculating probabilities of getting a value for a sensor that is x% more than the minimum value for a sensor in that house.
    for perc in [50, 75, 100, 150, 200]:
        boxplot_data[f'min_plus_{perc}'] = boxplot_data.groupby('house_id')[value_name].transform(lambda x: x.min() * (1 + perc / 100.0))
        probability = (boxplot_data[value_name] >= boxplot_data[f'min_plus_{perc}']).mean()
        print(f"\nProbability of getting an {value_name} value for a sensor that is {perc}% more than the minimum {value_name} value for a sensor in that house: {probability}")

    unique_houses = boxplot_data["house_id"].unique()

    diff_percentiles = []
    for house in unique_houses:
        house_values = boxplot_data[boxplot_data["house_id"] == house][value_name]
        min_val = house_values.min()
        max_val = house_values.max()
        diff_percent = ((max_val - min_val) / min_val) * 100
        diff_percentiles.append(diff_percent)

    upper_percentiles = [100 - p for p in [25, 50, 75]]
    upper_diffs = [np.percentile(diff_percentiles, p) for p in upper_percentiles]

    for p, diff in zip(upper_percentiles, upper_diffs):
        print(f"With {100-p}% probability, the percentile difference from the minimum to maximum {value_name} value in the same house is more than approximately {diff}%.")

compute_statistics(boxplot_data_RC, 'RC')
compute_statistics(boxplot_data_RQ, 'RQ')

  #%%
def compute_percent_identification(dataset, values_dict, values_no_outliers_dict, errors_dict, value_name):
    total_data = len(dataset) * 6
    non_nan_couples = 0
    error_filtered_non_nan_couples = 0

    # Calculate the number of house-sensor couples that are non-NaN for the given values
    for house_id, house_data in values_dict.items():
        for sensor, sensor_value in house_data.items():
            if not np.isnan(sensor_value):
                non_nan_couples += 1

    # Create the error-filtered values dictionary by removing values where the error is larger than 1
    error_filtered_values_dict = {house_id: {sensor: np.nan if np.isnan(values_no_outliers_dict[house_id][sensor]) or errors_dict[house_id][sensor] > 1 else values_no_outliers_dict[house_id][sensor] for sensor in sensor_names} for house_id in values_dict}

    # Calculate the number of house-sensor couples that are non-NaN after error filtering
    for house_id, house_data in error_filtered_values_dict.items():
        for sensor, sensor_value in house_data.items():
            if not np.isnan(sensor_value):
                error_filtered_non_nan_couples += 1

    # Compute the percentage of identification
    percentage_identification = (error_filtered_non_nan_couples / total_data) * 100

    # Store everything in a dictionary
    results_dict = {
        'total_data': total_data,
        'non_nan_couples': non_nan_couples,
        'error_filtered_non_nan_couples': error_filtered_non_nan_couples,
        'percentage_identification': percentage_identification
    }

    print(f"For {value_name} values:")
    print(f"Total data: {results_dict['total_data']}")
    print(f"Number of non-NaN house-sensor couples: {results_dict['non_nan_couples']}")
    print(f"Number of non-NaN house-sensor couples after error filtering: {results_dict['error_filtered_non_nan_couples']}")
    print(f"Percentage of identification: {results_dict['percentage_identification']}%")

    return results_dict

rc_results = compute_percent_identification(five_houses, RC_values, RC_values_no_outliers, errors, 'RC')
rq_results = compute_percent_identification(five_houses, RQ_values, RQ_values_no_outliers, errors, 'RQ')


#%%

# Combine outlier removal and error filtering
final_values_dict_RC = {house_id: {sensor: np.nan if np.isnan(RC_values_no_outliers[house_id][sensor]) or errors[house_id][sensor] > 1 else RC_values_no_outliers[house_id][sensor] for sensor in sensor_names} for house_id in RC_values_no_outliers}

final_values_dict_RQ = {house_id: {sensor: np.nan if np.isnan(RQ_values_no_outliers[house_id][sensor]) or errors[house_id][sensor] > 1 else RQ_values_no_outliers[house_id][sensor] for sensor in sensor_names} for house_id in RQ_values_no_outliers}

# Convert the final dictionaries to DataFrames
final_df_RC = pd.DataFrame(final_values_dict_RC).transpose()
final_df_RQ = pd.DataFrame(final_values_dict_RQ).transpose()

# Save the DataFrames to CSV files
final_df_RC.to_csv('final_RC_values.csv')
final_df_RQ.to_csv('final_RQ_values.csv')




