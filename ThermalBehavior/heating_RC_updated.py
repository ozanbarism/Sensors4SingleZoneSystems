#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:20:11 2023

@author: ozanbaris
This script estimates the RC values for the heating season using curve_fit.
Heating season is computed for each house by checking the first time heating turned on and the last time it was on. 
"""
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, to_hex

# File names
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 
              'May_clean.nc', 
              'Sep_clean.nc', 'Oct_clean.nc', 'Nov_clean.nc', 'Dec_clean.nc']

df_heat = pd.DataFrame()

# Load each file and append it to df_heat
for file_name in file_names:
    # Load the .nc file
    data = xr.open_dataset(file_name)

    # Convert it to a DataFrame
    temp_df_heat = data.to_dataframe()

    # Reset the index
    temp_df_heat = temp_df_heat.reset_index()

    # Append the data to df_heat
    df_heat = pd.concat([df_heat, temp_df_heat], ignore_index=True)

# Now df_heat is a pandas DataFrame and you can perform any operation you want
print(df_heat.head())

#%% Since we do not have a full winter period, we remove 1 year so that data goes from december to the january. 
house_heat_data = {}

unique_house_ids = df_heat['id'].unique()

for house_id in unique_house_ids:
    single_house_heat_data = df_heat[df_heat['id'] == house_id]
    
    # Subtract 1 year from 'time' for months September through December
    single_house_heat_data.loc[single_house_heat_data['time'].dt.month >= 9, 'time'] = single_house_heat_data.loc[single_house_heat_data['time'].dt.month >= 9, 'time'].apply(lambda x: x.replace(year=x.year-1))
    
    # Sort DataFrame by 'time'
    single_house_heat_data.sort_values(by='time', inplace=True)
    
    house_heat_data[house_id] = single_house_heat_data

#%%# 
# Prepare dictionaries to store cooling and heating season data for each house
heating_season_dict = {}

for house_id, single_house_heat_data in house_heat_data.items():
    # Make a copy of the DataFrame to keep the original one intact
    single_house_heat_data_copy = single_house_heat_data.copy()

    # Set 'time' column as index
    single_house_heat_data_copy.set_index('time', inplace=True)

    # Identify when the HVAC system is in heating mode
    single_house_heat_data_copy['Heating_Mode'] = single_house_heat_data_copy['HeatingEquipmentStage1_RunTime'].notna()

    # Identify the periods of heating
    heating_start = single_house_heat_data_copy.loc[single_house_heat_data_copy['Heating_Mode']].index.min()
    heating_end = single_house_heat_data_copy.loc[single_house_heat_data_copy['Heating_Mode']].index.max()

    # Extract heating season data
    heating_season_data = single_house_heat_data.loc[(single_house_heat_data['time'] >= heating_start) & (single_house_heat_data['time'] <= heating_end)]

    # Store in dictionaries
    heating_season_dict[house_id] = heating_season_data



#%%
# Create empty dictionaries for each category
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}


#for house_id, data in house_heat_data.items():
for house_id, data in heating_season_dict.items():
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

valid_periods_dict = {}

# For each sensor, check if its temperature is more than 20 F above Indoor_CoolSetpoint
sensor_columns = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']

for house_id, single_house_data_raw in five_houses.items():
    # --- FIX IS HERE ---
    # Create a copy AND reset the index. 
    # This moves 'time' from the index back to a regular column.
    single_house_data = single_house_data_raw.copy().reset_index()

    # 1. Create 'Night' column
    single_house_data.loc[:, 'Night'] = single_house_data['time'].apply(
        lambda x: x.date() if x.hour >= 7 else (x - pd.Timedelta(days=1)).date()
    )

    # Initialize dictionary for the house
    valid_periods_dict[house_id] = {}

    for sensor in sensor_columns:
        # 2. Identify when HVAC is off and when CoolingEquipmentStage1_RunTime is NaN
        single_house_data.loc[:, 'HVAC_Off'] = (
            single_house_data['HeatingEquipmentStage1_RunTime'].isna() & 
            single_house_data['CoolingEquipmentStage1_RunTime'].isna()
        )

        # Identify groups of rows where HVAC is off. 
        single_house_data.loc[:, 'Off_Period'] = (
            single_house_data['HVAC_Off'] & 
            (~single_house_data['HVAC_Off'].shift(1).fillna(False))
        ).cumsum()

        # 3. Filter specific periods
        # Group by Night and Off_Period
        for (night, period), period_data in single_house_data.groupby(['Night', 'Off_Period']):
            
            # --- FILTERING LOGIC ---
            
            # A. Basic Checks
            is_night_time = ((period_data['time'].dt.hour >= 22) | (period_data['time'].dt.hour < 7))
            is_correct_length = len(period_data) >= 12 and len(period_data) <= 12 * 9
            is_temp_change_sufficient = (period_data[sensor].max() - period_data[sensor].min()) >= 2
            no_nan_values = period_data[sensor].isnull().sum() == 0
            
            # B. Check Time Continuity
            # Check if time steps are roughly 5 minutes (300s +/- 10s buffer)
            time_diffs = period_data['time'].diff().dt.total_seconds().dropna()
            is_continuous = ((time_diffs >= 290) & (time_diffs <= 310)).all()

            # C. Check Monotonic Decrease (Heat Source Check)
            # Diff > 0 means temperature rose. We want all Diff <= 0.
            # Using <= 0.0 to be strict, or <= 0.05 to allow tiny sensor noise
            temp_diffs = period_data[sensor].diff().dropna()
            is_monotonically_decreasing = (temp_diffs <= 0).all() 

            if (is_night_time.all() and is_correct_length and is_temp_change_sufficient 
                and no_nan_values and is_continuous and is_monotonically_decreasing):
                
                # Add to the dictionary using house-sensor pair as key
                if sensor not in valid_periods_dict[house_id]:
                    valid_periods_dict[house_id][sensor] = []
                valid_periods_dict[house_id][sensor].append(period_data)




#%%

hist_val_periods=[]
for house_id, house_periods in valid_periods_dict.items():
    num_valid_periods = sum(len(periods) for periods in house_periods.values())
    hist_val_periods.append(num_valid_periods)

# Create a histogram of the number of valid periods
plt.figure(figsize=(10, 6))
plt.hist(hist_val_periods, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Number of Houses')
plt.ylabel('Number of Buildings')
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

# Dictionary to store RC_heat and RQ for each house and each sensor
RC_values = {}
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

for house_id, sensors in valid_periods_dict.items():
    RC_values[house_id] = {}
    errors[house_id] = {}

    for sensor, periods in sensors.items():
        # Initialize list to store RC_heat values, RQ values and error_heats for this sensor
        RC_values_sensor = []
        errors_sensor = []

        for data in periods:
            # Define the function to fit, now with RQ and RC_heat as parameters
            def model_func(t, RC_heat):
                return T_diff0 * np.exp(-t / RC_heat)

            # Drop the rows where either sensor or Outdoor_Temperature data is NaN
            data = data.dropna(subset=[sensor, 'Outdoor_Temperature'])
            
            m=len(data[sensor])
            t = np.arange(len(data[sensor])) / 12

            T_diff0 = data[sensor].iloc[0] - data['Outdoor_Temperature'].iloc[:m].mean()
            T_diff_t = data[sensor] - data['Outdoor_Temperature'].iloc[:m].mean()

            # Perform curve fitting with initial guesses for RQ and RC_heat
            popt, _ = curve_fit(model_func, t, T_diff_t, p0=20, bounds=(0, 200))
            RC_values_sensor.append(popt[0])

            # Make predictions
            T_diff_pred = model_func(t, *popt)
            # Calculate error_heat
            error = np.sqrt(np.mean((T_diff_t - T_diff_pred) ** 2))
            errors_sensor.append(error)
        # Remove outliers from RC_heat_values_sensor and error_heats_sensor
        RC_values_sensor = remove_outliers(RC_values_sensor)
        errors_sensor = remove_outliers(errors_sensor)

        # Store average RC_heat value, average RQ and average error_heat for this sensor
        RC_values[house_id][sensor] = np.mean(RC_values_sensor)
        errors[house_id][sensor] = np.mean(errors_sensor)
#%%


# Dictionary to store RC_heat and RQ for each house and each sensor
RC_values = {}
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

for house_id, sensors in valid_periods_dict.items():
    RC_values[house_id] = {}
    errors[house_id] = {}

    for sensor, periods in sensors.items():
        # Initialize list to store RC_heat values, RQ values and error_heats for this sensor
        RC_values_sensor = []
        errors_sensor = []

        for data in periods:
            # Define the function to fit, now with RQ and RC_heat as parameters
            def model_func(t, RC_heat):
                return T_diff0 * np.exp(-t / RC_heat)

            # Drop the rows where either sensor or Outdoor_Temperature data is NaN
            data = data.dropna(subset=[sensor, 'Outdoor_Temperature'])
            
 
            t = np.arange(len(data[sensor])) / 12

            T_diff0 = data[sensor].iloc[0] - data['Outdoor_Temperature'].mean()
            T_diff_t = data[sensor] - data['Outdoor_Temperature'].mean()

            # Perform curve fitting with initial guesses for RQ and RC_heat
            popt, _ = curve_fit(model_func, t, T_diff_t, p0=20, bounds=(0, 200))
            RC_values_sensor.append(popt[0])

            # Make predictions
            T_diff_pred = model_func(t, *popt)
            # Calculate error_heat
            error = np.sqrt(np.mean((T_diff_t - T_diff_pred) ** 2))
            errors_sensor.append(error)
        # Remove outliers from RC_heat_values_sensor and error_heats_sensor
        RC_values_sensor = remove_outliers(RC_values_sensor)
        errors_sensor = remove_outliers(errors_sensor)

        # Store average RC_heat value, average RQ and average error_heat for this sensor
        RC_values[house_id][sensor] = np.mean(RC_values_sensor)
        errors[house_id][sensor] = np.mean(errors_sensor)
#%%
num_not_nan_couples = 0

# Iterate over the dictionary
for house_id, sensors in RC_values.items():
    for sensor, value in sensors.items():
        # Check if the value is not NaN
        if not pd.isnull(value):
            num_not_nan_couples += 1

print(f"Number of house-sensor couples that are not NaN: {num_not_nan_couples}")
num_houses_all_sensors_not_nan = 0

# Iterate over the dictionary
for house_id, sensors in RC_values.items():
    # Check if all sensor values for the house are not NaN
    if all(not pd.isnull(value) for value in sensors.values()):
        num_houses_all_sensors_not_nan += 1

print(f"Number of houses with all sensor values that are not NaN: {num_houses_all_sensors_not_nan}")

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
    # Then assign the outlier-removed data back to the new dictionary
    for house_id in values_dict:
        for sensor in sensor_names:
            if sensor in values_dict[house_id] and sensor in values_sensors_no_outliers and values_dict[house_id][sensor] in values_sensors_no_outliers[sensor]:
                values_dict_no_outliers[house_id][sensor] = values_dict[house_id][sensor]
    
        # Gather all filtered values into a single list
    values_no_outliers = [value for sensor_values in values_dict_no_outliers.values() for value in sensor_values.values() if not np.isnan(value)]

    # Count of non-NaN values after outlier removal
    count_after = len(values_no_outliers)
    print(f"Count of non-NaN values after outlier removal: {count_after}")

    return values_dict_no_outliers, values_sensors_no_outliers, values_no_outliers

RC_values_no_outliers, RC_values_sensors_no_outliers, RC_values_no_outliers_list = outlier_removal(RC_values, sensor_names)


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
    
    
def plot_data(df, var_name, title, unit):
    # Set the overall font size to 26
    plt.rcParams.update({'font.size': 26})

    # Create a gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5], hspace=0) 

    # Create figure
    fig = plt.figure(figsize=(15, 15))

    # Create the histogram in the top subplot
    ax0 = plt.subplot(gs[0])
    # Plot histograms for each state
    sns.histplot(data=df, x=var_name, hue='state', palette=state_colors_dark, multiple="stack", bins=50, edgecolor=".3", linewidth=.5, ax=ax0, legend=False)
    ax0.set_ylabel('Count')  # Update y-axis label
    ax0.set_xlabel('')  # Remove x-axis label

    # Create the boxplot in the bottom subplot
    ax1 = plt.subplot(gs[1], sharex=ax0)  # share x-axis with the histogram
    sns.boxplot(y='rank', x=var_name, data=df, hue='state', palette=state_colors, dodge=False, ax=ax1)

    ax1.legend(title='State', bbox_to_anchor=(1, 0), loc='lower right')  # Place legend at the lower right end

    # Customize ytick labels
    yticks_locs = ax1.get_yticks()  # get current ytick locations
    new_yticks = [i+1 if i%10==0 else '' for i in range(len(yticks_locs))]  # show every 10th house number, starting from 1
    new_yticks[0] = ''  # Remove the tick on the first value of the House Number axis
    ax1.set_yticks(yticks_locs)  # set new ytick locations
    ax1.set_yticklabels(new_yticks)  # set new ytick labels

    ax1.set_xlabel(f'{title} Value ({unit})')  # Update x-axis label
    ax1.set_ylabel('House Number')  # Update y-axis label

    plt.tight_layout()  # Make sure everything fits
    plt.show()



 #%%
def compute_boxplot_data(five_houses, sensor_names, errors, var_name, state_dict, values_dict, values_sensors_no_outliers):

    # Compute the data for boxplot
    values_no_outliers_dict = {house_id: {sensor: values_dict[house_id][sensor] if (sensor in values_sensors_no_outliers and sensor in values_dict[house_id] and values_dict[house_id][sensor] in values_sensors_no_outliers[sensor]) else np.nan for sensor in sensor_names} for house_id in values_dict}
    
    df_values = pd.DataFrame(values_no_outliers_dict).T
    df_values.reset_index(inplace=True)
    df_values.rename(columns={'index': 'house_id'}, inplace=True)

    df_errors = pd.DataFrame(errors).T
    df_errors.reset_index(inplace=True)
    df_errors.rename(columns={'index': 'house_id'}, inplace=True)

    df_values_filtered = df_values.loc[:, df_values.columns[1:]].where(df_errors.iloc[:, 1:] <= 1)
    df_values_filtered.insert(0, 'house_id', df_values['house_id'])
    df_values_filtered.dropna(how='all', subset=df_values_filtered.columns[1:], inplace=True)
    df_values_filtered.index = range(1, len(df_values_filtered) + 1)
    
    boxplot_data = df_values_filtered.melt(id_vars='house_id', var_name='sensor', value_name=var_name)
    boxplot_data.dropna(subset=[var_name], inplace=True)
    
    sorted_houses = boxplot_data.groupby("house_id")[var_name].max().sort_values(ascending=False).index
    boxplot_data["house_id"] = boxplot_data["house_id"].astype("category")
    boxplot_data["house_id"].cat.set_categories(sorted_houses, inplace=True)
    boxplot_data['state'] = boxplot_data['house_id'].map(state_dict)
    
    plt.rcParams.update({'font.size': 26})
    var_range = boxplot_data.groupby("house_id")[var_name].max() - boxplot_data.groupby("house_id")[var_name].min()
    sorted_houses = var_range.sort_values(ascending=True).index
    rank_mapping = {house_id: i+1 for i, house_id in enumerate(sorted_houses)}
    boxplot_data["rank"] = boxplot_data["house_id"].map(rank_mapping)
    boxplot_data.sort_values('rank', inplace=True)

    return boxplot_data



boxplot_data_RC = compute_boxplot_data(five_houses, sensor_names, errors, 'RC', state_dict, RC_values, RC_values_sensors_no_outliers)
plot_data(boxplot_data_RC, 'RC', 'RC', 'h')

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

#%%
# Combine outlier removal and error filtering
final_values_dict_RC = {house_id: {sensor: np.nan if np.isnan(RC_values_no_outliers[house_id][sensor]) or errors[house_id][sensor] > 1 else RC_values_no_outliers[house_id][sensor] for sensor in sensor_names} for house_id in RC_values_no_outliers}


# Convert the final dictionaries to DataFrames
final_df_RC = pd.DataFrame(final_values_dict_RC).transpose()


# Save the DataFrames to CSV files
final_df_RC.to_csv('final_RC_heat_values.csv')
#%%

def compute_rc_statistics(final_values_dict_RC):
    all_values = []
    differences = {}

    for house_id, sensor_data in final_values_dict_RC.items():
        sensor_values = [value for value in sensor_data.values() if not np.isnan(value)]
        if sensor_values:
            all_values.extend(sensor_values)
            smallest = min(sensor_values)
            largest = max(sensor_values)
            differences[house_id] = largest - smallest

    smallest_rc_value = min(all_values)
    largest_rc_value = max(all_values)
    max_difference = max(differences.values())

    return smallest_rc_value, largest_rc_value, max_difference

smallest_rc_value, largest_rc_value, max_difference = compute_rc_statistics(final_values_dict_RC)

print(f"Smallest RC value across all homes: {smallest_rc_value}")
print(f"Largest RC value across all homes: {largest_rc_value}")
print(f"Largest difference among individual house differences: {max_difference}")

#%%

# Create a dictionary to store RC values for each house
RC_values_houses = {house_id: [] for house_id in RC_values.keys()}

# Iterate over the RC_values dictionary
for house_id, sensors in RC_values.items():
    for sensor, RC in sensors.items():
        RC_values_houses[house_id].append(RC)

# Remove outliers from the RC values for each house
RC_values_houses_no_outliers = {house_id: remove_outliers(RC_values_houses[house_id]) for house_id in RC_values_houses.keys()}

# Plot a boxplot for each house without outliers
plt.figure(figsize=(10, 6))
plt.boxplot(RC_values_houses_no_outliers.values(), vert=False, patch_artist=True, labels=RC_values_houses_no_outliers.keys())
plt.xlabel('RC Values')
plt.title('Boxplot of RC Values for Each House (No Outliers)')
plt.show()

# Calculate the difference between the maximum and minimum RC value for each house
RC_diff_houses = {house_id: max(RC_values_houses[house_id]) - min(RC_values_houses[house_id]) for house_id in RC_values_houses.keys()}

# Plot a bar plot of the RC differences
plt.figure(figsize=(10, 6))
plt.bar(RC_diff_houses.keys(), RC_diff_houses.values())
plt.xlabel('House')
plt.ylabel('RC Difference')
plt.title('Difference between Maximum and Minimum RC Value for Each House')
plt.xticks(rotation=90)
plt.show()



#%% TO PLOT EACH HOUSE ACCORDING TO THEIR STATE
# Calculate the cooling days for each house and store it in a dictionary

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

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Create a dictionary of RC values without outliers with house_id as keys and sensor values as sub-dictionaries


df_errors = pd.DataFrame(errors).T
df_errors.reset_index(inplace=True)
df_errors.rename(columns={'index': 'house_id'}, inplace=True)

# Filter values based on the error condition, excluding the first column
df_RC_values_filtered = df_RC_values_no_outliers.loc[:, df_RC_values_no_outliers.columns[1:]].where(df_errors.iloc[:, 1:] <= 1)

# Include the 'house_id' column back into the filtered dataframe
df_RC_values_filtered.insert(0, 'house_id', df_RC_values_no_outliers['house_id'])

# Drop houses with all NaN values
df_RC_values_filtered.dropna(how='all', subset=df_RC_values_filtered.columns[1:], inplace=True)

# Replace house_id with numbers
df_RC_values_filtered.index = range(1, len(df_RC_values_filtered) + 1)

# Prepare data for boxplot. The resulting DataFrame will have two columns: 'house' and 'RC'.
boxplot_data = df_RC_values_filtered.melt(id_vars='house_id', var_name='sensor', value_name='RC')

# Drop rows with NaN values
boxplot_data.dropna(subset=['RC'], inplace=True)

# Sort houses by maximum RC value
sorted_houses = boxplot_data.groupby("house_id")["RC"].max().sort_values(ascending=False).index
boxplot_data["house_id"] = boxplot_data["house_id"].astype("category")
boxplot_data["house_id"].cat.set_categories(sorted_houses, inplace=True)


state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}

# Map state for each house
boxplot_data['state'] = boxplot_data['house_id'].map(state_dict)

plt.figure(figsize=(15, 10))

# Create boxplot
boxplot = sns.boxplot(x='house_id', y='RC', data=boxplot_data, hue='state', palette=state_colors, dodge=False)

# Create an array of sequential numbers with the length of unique house_ids
xticks_labels = np.arange(1, len(boxplot_data['house_id'].unique())+1)

# Set new xticks
boxplot.set_xticks(np.arange(len(xticks_labels)))

# Set new xtick labels and make every 10th label visible
boxplot.set_xticklabels(xticks_labels)
for ind, label in enumerate(boxplot.get_xticklabels()):
    label.set_visible(False)
    if ind % 10 == 0:
        label.set_visible(True)

# Set fontsize for xticks and yticks
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Set title and labels with font size 24
plt.title('Boxplot of RC Values for Each House', fontsize=24)
plt.xlabel('House Number', fontsize=24)
plt.ylabel('RC Value (h)', fontsize=24)

# Create and format the legend
legend = plt.legend(title='State', bbox_to_anchor=(1, 1), fontsize=24)
plt.setp(legend.get_title(), fontsize=24)  # Set fontsize of the legend title

plt.show()




#%%
# Define file paths
rc_file_path = "RC_heat_values.csv"

# Save df_RC_values_filtered dataframe as CSV
df_RC_values_filtered.to_csv(rc_file_path)



