# Databricks notebook source
pip install tensorflow pandas numpy matplotlib seaborn scipy scikit-learn openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the data path and results directory
data_path = r"/Workspace/Users/shruti.shreya-1@syngenta.com/weather_data_2020-25/Plantix_data_2023_2024_merged.csv"
results_dir = 'std_vals_tel_rice_main_23_24'

# COMMAND ----------


# Create or empty the results directory
os.makedirs(results_dir, exist_ok=True)
for file in os.listdir(results_dir):
    os.remove(os.path.join(results_dir, file))
logging.info(f"Directory '{results_dir}' has been created/emptied.")

# COMMAND ----------

# 1. Load the data
main_data = pd.read_csv(data_path)
logging.info("Data loaded successfully.")
logging.info(f"Original data shape: {main_data.shape}")


# COMMAND ----------

# 2. Clean and format the data
for col in main_data.columns:
    if main_data[col].dtype == 'object':
        main_data[col] = main_data[col].fillna(main_data[col].mode()[0])
    else:
        main_data[col] = main_data[col].fillna(main_data[col].mean())

# Convert date column to datetime
main_data['date'] = pd.to_datetime(main_data['date'], dayfirst=True)
logging.info("Data cleaning and formatting completed.")


# COMMAND ----------

main_data = main_data.rename(columns={'plantix_count_3days': 'plantix_count_7days'})
main_data.head()

# COMMAND ----------

# 3. Select the data based on criteria
def select_data(data, state=None, crop=None, disease_name=None):
    mask = pd.Series(True, index=data.index)
    if state:
        mask &= data['state'].isin(state)
    if crop:
        mask &= data['crop'].isin(crop)
    if disease_name:
        mask &= data['disease_name'].isin(disease_name)
    return data[mask]

# 3.1 Add state, crop, and disease name filters
state = ['Telangana']
crop = ['RICE']
disease_name = ['Bacterial Blight of Rice', 'Bacterial Panicle Blight', 'Blast of Rice', 'Rice Sheath Blight', 'Stem Rot of Rice']
main_data = select_data(main_data, state, crop, disease_name)
logging.info(f"Data filtered. New shape: {main_data.shape}")
selected_data = main_data



# COMMAND ----------

main_data

# COMMAND ----------

# 4. Identify weather parameters and count column
weather_params = [col for col in main_data.columns if col.startswith(('Temperature', 'Relative Humidity', 'Precipitation', 'Leaf wetness', 'Soil Moisture', 'Soil Temperature', 'Wind Speed', 'Wind Direction', 'Sunshine Duration'))]
count_column = 'plantix_count_7days'
weather_params

# COMMAND ----------

# 5. Create monthly aggregate data
monthly_data = main_data.groupby(main_data['date'].dt.to_period('M')).agg({
    **{param: 'mean' for param in weather_params},
    count_column: 'sum'
}).reset_index()
monthly_data['date'] = monthly_data['date'].dt.to_timestamp()
monthly_data['year'] = monthly_data['date'].dt.year
monthly_data['month'] = monthly_data['date'].dt.month

# COMMAND ----------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_new_features(df, standard_values_file, is_example=False, 
                        rolling_windows=[3, 7, 14, 30], 
                        interaction_pairs=None, 
                        high_temp_threshold=25, 
                        high_humidity_threshold=80,
                        include_advanced_features=True):
    """
    Create new features including rolling averages, interactions, deviations, and more advanced weather-related features.
    Args:
    df (pd.DataFrame): Input dataframe
    standard_values_file (str): Path to the Excel file with standard values
    is_example (bool): Whether this is an example run
    rolling_windows (list): List of window sizes for rolling averages
    interaction_pairs (list): List of tuples with column names to create interaction features
    high_temp_threshold (float): Threshold for high temperature
    high_humidity_threshold (float): Threshold for high humidity
    include_advanced_features (bool): Whether to include advanced weather-related features
    """
    try:
        # Load standard values from Excel file
        standard_values = pd.read_excel(standard_values_file)
        if 'disease_name' not in standard_values.columns:
            possible_disease_columns = [col for col in standard_values.columns if 'disease' in col.lower()]
            if possible_disease_columns:
                disease_column = possible_disease_columns[0]
                standard_values.set_index(disease_column, inplace=True)
            else:
                raise ValueError("No suitable column found to use as disease name.")
        else:
            standard_values.set_index('disease_name', inplace=True)
 
        if not is_example:
            df = df.sort_values(['state', 'crop', 'disease_name', 'date'])
        # Identify weather columns
        weather_cols = [col for col in df.columns if any(param in col for param in ['Temperature', 'Relative Humidity', 'Leaf wetness', 'Precipitation', 'Soil', 'Sunshine'])]
        # Create rolling averages
        for col in weather_cols:
            for window in rolling_windows:
                if is_example:
                    df[f'{col}_{window}day_avg'] = df[col]
                else:
                    df[f'{col}_{window}day_avg'] = df.groupby(['state', 'crop', 'disease_name'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        # Create interaction features
        if interaction_pairs is None:
            interaction_pairs = [('Temperature_max', 'Relative Humidity_max')]
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_{col2}_Interaction'] = df[col1] * df[col2]
                df[f'High_{col1}_High_{col2}'] = ((df[col1] > high_temp_threshold) & (df[col2] > high_humidity_threshold)).astype(int)
        # Add season feature
        df['month'] = df['date'].dt.month
        df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
        df = pd.get_dummies(df, columns=['season'])
        # Create deviation features using standard values
        for col in weather_cols:
            if col in standard_values.columns:
                df[f'{col}_deviation'] = df.apply(lambda row: row[col] - standard_values.loc[row['disease_name'], col] if row['disease_name'] in standard_values.index else np.nan, axis=1)
                df[f'{col}_relative_deviation'] = df.apply(lambda row: 
                    (row[col] - standard_values.loc[row['disease_name'], col]) / standard_values.loc[row['disease_name'], col] 
                    if row['disease_name'] in standard_values.index and standard_values.loc[row['disease_name'], col] != 0 else np.nan, axis=1)
        # Create squared and cube terms for original weather parameters only
        for col in weather_cols:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cube'] = df[col] ** 3
 
        if include_advanced_features:
            # Advanced features
            if 'Temperature_max' in df.columns and 'Temperature_min' in df.columns:
                df['Diurnal_Temperature_Range'] = df['Temperature_max'] - df['Temperature_min']
            if 'Rainfall_sum' in df.columns:
                df['Dry_Day'] = (df['Rainfall_sum'] < 1).astype(int)
                df['Consecutive_Dry_Days'] = df.groupby(['state', 'crop', 'disease_name'])['Dry_Day'].transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
            if 'Relative Humidity_max' in df.columns and 'Relative Humidity_min' in df.columns:
                df['Humidity_Range'] = df['Relative Humidity_max'] - df['Relative Humidity_min']
            if 'Temperature_mean' in df.columns and 'Relative Humidity_mean' in df.columns:
                df['Vapor_Pressure_Deficit'] = 0.611 * np.exp((17.27 * df['Temperature_mean']) / (df['Temperature_mean'] + 237.3)) * (1 - df['Relative Humidity_mean'] / 100)
                df['Heat_Index'] = -42.379 + 2.04901523 * df['Temperature_mean'] + 10.14333127 * df['Relative Humidity_mean'] - 0.22475541 * df['Temperature_mean'] * df['Relative Humidity_mean'] - 6.83783e-3 * df['Temperature_mean']**2 - 5.481717e-2 * df['Relative Humidity_mean']**2 + 1.22874e-3 * df['Temperature_mean']**2 * df['Relative Humidity_mean'] + 8.5282e-4 * df['Temperature_mean'] * df['Relative Humidity_mean']**2 - 1.99e-6 * df['Temperature_mean']**2 * df['Relative Humidity_mean']**2
 
        logger.info("Features created successfully")
        return df
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        raise

# COMMAND ----------

standard_values_file= '/Workspace/Users/shruti.shreya-1@syngenta.com/weather_data_2020-25/standard_val.xlsx'

# COMMAND ----------

main_data = create_new_features(selected_data, standard_values_file)


# COMMAND ----------

main_data.head()

# COMMAND ----------

# 9. Daily aggregate
daily_data = main_data.groupby(['date', 'state', 'crop', 'disease_name']).agg({
    **{param: 'mean' for param in weather_params if 'Precipitation' not in param},
    **{param: 'sum' for param in weather_params if 'Precipitation' in param},
    count_column: 'sum'
}).reset_index()


# COMMAND ----------

# correlations = daily_data[weather_params + ['plantix_count_7days']].corr(method= 'spearman')['plantix_count_7days'].drop('plantix_count_7days')

# # Sort correlations
# sorted_correlations = correlations.abs().sort_values(ascending=False)

# # Print correlations
# print("Correlations with plantix_count_7days:")
# print(sorted_correlations)

# COMMAND ----------

daily_data = create_new_features(daily_data, standard_values_file)


# COMMAND ----------

daily_data.head()

# COMMAND ----------

new_features_daily= ['Leaf wetness probability_mean_3day_avg',
       'Leaf wetness probability_mean_7day_avg',
       'Leaf wetness probability_mean_14day_avg',
       'Leaf wetness probability_mean_30day_avg',
       'Relative Humidity_max_3day_avg', 'Relative Humidity_max_7day_avg',
       'Relative Humidity_max_14day_avg', 'Relative Humidity_max_30day_avg',
       'Relative Humidity_min_3day_avg', 'Relative Humidity_min_7day_avg',
       'Relative Humidity_min_14day_avg', 'Relative Humidity_min_30day_avg',
       'Soil Moisture_mean_3day_avg', 'Soil Moisture_mean_7day_avg',
       'Soil Moisture_mean_14day_avg', 'Soil Moisture_mean_30day_avg',
       'Soil Temperature_mean_3day_avg', 'Soil Temperature_mean_7day_avg',
       'Soil Temperature_mean_14day_avg', 'Soil Temperature_mean_30day_avg',
       'Sunshine Duration_mean_3day_avg', 'Sunshine Duration_mean_7day_avg',
       'Sunshine Duration_mean_14day_avg', 'Sunshine Duration_mean_30day_avg',
       'Temperature_max_3day_avg', 'Temperature_max_7day_avg',
       'Temperature_max_14day_avg', 'Temperature_max_30day_avg',
       'Temperature_min_3day_avg', 'Temperature_min_7day_avg',
       'Temperature_min_14day_avg', 'Temperature_min_30day_avg',
       'Precipitation Total_mean_3day_avg',
       'Precipitation Total_mean_7day_avg',
       'Precipitation Total_mean_14day_avg',
       'Precipitation Total_mean_30day_avg',
       'Temperature_max_Relative Humidity_max_Interaction',
       'High_Temperature_max_High_Relative Humidity_max', 'month',
       'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall',
       'Leaf wetness probability_mean_deviation',
       'Leaf wetness probability_mean_relative_deviation',
       'Relative Humidity_max_deviation',
       'Relative Humidity_max_relative_deviation',
       'Relative Humidity_min_deviation',
       'Relative Humidity_min_relative_deviation',
       'Soil Moisture_mean_deviation', 'Soil Moisture_mean_relative_deviation',
       'Soil Temperature_mean_deviation',
       'Soil Temperature_mean_relative_deviation',
       'Sunshine Duration_mean_deviation',
       'Sunshine Duration_mean_relative_deviation',
       'Temperature_max_deviation', 'Temperature_max_relative_deviation',
       'Temperature_min_deviation', 'Temperature_min_relative_deviation',
       'Precipitation Total_mean_deviation',
       'Precipitation Total_mean_relative_deviation',
       'Leaf wetness probability_mean_squared',
       'Leaf wetness probability_mean_cube', 'Relative Humidity_max_squared',
       'Relative Humidity_max_cube', 'Relative Humidity_min_squared',
       'Relative Humidity_min_cube', 'Soil Moisture_mean_squared',
       'Soil Moisture_mean_cube', 'Soil Temperature_mean_squared',
       'Soil Temperature_mean_cube', 'Sunshine Duration_mean_squared',
       'Sunshine Duration_mean_cube', 'Temperature_max_squared',
       'Temperature_max_cube', 'Temperature_min_squared',
       'Temperature_min_cube', 'Precipitation Total_mean_squared',
       'Precipitation Total_mean_cube', 'Diurnal_Temperature_Range',
       'Humidity_Range']

# COMMAND ----------

daily_data.head()

# COMMAND ----------

# 11. Select each disease data datewise

def select_data(data, disease_name=None):
    return data[data['disease_name'].isin(disease_name)] if disease_name else data

diseases = {
    'Bacterial Blight': ('Bacterial Blight of Rice', daily_data),
    'Bacterial Panicle Blight': ('Bacterial Panicle Blight', daily_data),
    'Blast':('Blast of Rice', daily_data),
    'Stem Rot': ('Stem Rot of Rice', daily_data),
    'Sheath Blight':('Rice Sheath Blight', daily_data)
}

disease_data = {}

for disease, (disease_name, source_data) in diseases.items():
    filtered_data = select_data(source_data, [disease_name])
    disease_data[disease] = filtered_data
    logging.info(f"{disease} data filtered. Shape: {filtered_data.shape}")

# Access the filtered data
bacterial_blight_daily = disease_data['Bacterial Blight']
bacterial_panicle_blight_daily = disease_data['Bacterial Panicle Blight']
blast_daily = disease_data['Blast']
stem_rot_daily= disease_data['Stem Rot']
sheath_blight_daily = disease_data['Sheath Blight']

# COMMAND ----------

bacterial_blight_daily.head()

# COMMAND ----------

# # 12. Calculate and plot correlations for all disease-daily-main datasets

# def calculate_correlations(data, target_column, features):
#     correlations = []
    
#     for feature in features:
#         if feature not in data.columns:
#             continue
        
#         # Spearman's correlation
#         spearman, _ = stats.spearmanr(data[feature], data[target_column])
        
#         correlations.append({
#             'Feature': feature,
#             "Spearman's rho": spearman
#         })
    
#     return pd.DataFrame(correlations)
# def plot_correlations(data, title_prefix, file_prefix, features):
#     correlations = calculate_correlations(data, count_column, features)
#     correlations = correlations.sort_values(by="Spearman's rho", key=abs, ascending=False)
    
#     # Save correlations to CSV
#     correlations.to_csv(os.path.join(results_dir, f'{file_prefix}_spearman_correlations.csv'), index=False)
    
#     # Plot heatmap
#     plt.figure(figsize=(10, len(correlations) * 0.3))
#     sns.heatmap(correlations.set_index('Feature'), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
#     plt.title(f'{title_prefix} Spearman Correlations with Plantix Count', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_dir, f'{file_prefix}_spearman_correlations_heatmap.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     logging.info(f"{title_prefix} Spearman correlations saved.")


# logging.info("Analysis complete. All results have been saved to the results directory.")

# data_list = [
#     (main_data, "Main Data", "main_data", weather_params),
    
#     (daily_data, "daily data", "daily_data", weather_params),

# # daily disease

#     (bacterial_blight_daily, "bacterial blight daily orig", "bacterial_blight_daily", weather_params),
#     (bacterial_panicle_blight_daily, "bacterial panicle blight daily orig", "bacterial_panicle_blight_daily", weather_params),
#     (blast_daily, "blast daily orig", "blast_daily", weather_params),
#     (stem_rot_daily, "stem rot daily orig", "stem_rot_daily", weather_params),
#     (sheath_blight_daily, "sheath blight daily orig", "sheath_blight_daily", weather_params)
    
# ]

# # Loop through the data and create correlation plots
# for df, title, filename, features in data_list:
#     plot_correlations(df, title, filename, features)

# COMMAND ----------

# 11. Save processed data
main_data.to_csv(os.path.join(results_dir, 'processed_main_data.csv'), index=False)


daily_data.to_csv(os.path.join(results_dir, 'daily_data.csv'), index=False)

# five diease daily data save
bacterial_blight_daily.to_csv(os.path.join(results_dir, 'bacterial_blight_daily.csv'), index=False)
bacterial_panicle_blight_daily.to_csv(os.path.join(results_dir, 'bacterial_panicle_blight_daily.csv'), index=False)
blast_daily.to_csv(os.path.join(results_dir, 'blast_daily.csv'), index=False)
stem_rot_daily.to_csv(os.path.join(results_dir, 'stem_rot_daily.csv'), index=False)
sheath_blight_daily.to_csv(os.path.join(results_dir, 'sheath_blight_daily.csv'), index=False)

# COMMAND ----------

#12. correln check between plantix_count & weather params of single disease daily data

bacterial_blight_daily_correlations = bacterial_blight_daily[weather_params + new_features_daily + [count_column]].corr(method='spearman')[count_column].drop(count_column).sort_values(ascending=False)
print(f"bacterial blight daily correlations with '{count_column}':")
print(bacterial_blight_daily_correlations)


# COMMAND ----------

# 12. Calculate and plot correlations for all disease-daily-main datasets

def calculate_correlations(data, target_column, features):
    correlations = []
    
    for feature in features:
        if feature not in data.columns:
            continue
        
        # Spearman's correlation
        spearman, _ = stats.spearmanr(data[feature], data[target_column])
        
        correlations.append({
            'Feature': feature,
            "Spearman's rho": spearman
        })
    
    return pd.DataFrame(correlations)
def plot_correlations(data, title_prefix, file_prefix, features):
    correlations = calculate_correlations(data, count_column, features)
    correlations = correlations.sort_values(by="Spearman's rho", key=abs, ascending=False)
    
    # Save correlations to CSV
    correlations.to_csv(os.path.join(results_dir, f'{file_prefix}_spearman_correlations.csv'), index=False)
    
    # Plot heatmap
    plt.figure(figsize=(10, len(correlations) * 0.3))
    sns.heatmap(correlations.set_index('Feature'), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(f'{title_prefix} Spearman Correlations with Plantix Count', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{file_prefix}_spearman_correlations_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"{title_prefix} Spearman correlations saved.")


logging.info("Analysis complete. All results have been saved to the results directory.")

data_list = [
    (main_data, "Main Data", "main_data", weather_params + new_features_daily),
    
    (daily_data, "daily data", "daily_data", weather_params + new_features_daily),

# daily disease

    (bacterial_blight_daily, "bacterial blight daily", "bacterial_blight_daily", weather_params + new_features_daily),
    (bacterial_panicle_blight_daily, "bacterial panicle blight daily", "bacterial_panicle_blight_daily", weather_params + new_features_daily),
    (blast_daily, "blast daily", "blast_daily", weather_params + new_features_daily),
    (stem_rot_daily, "stem rot daily", "stem_rot_daily", weather_params + new_features_daily),
    (sheath_blight_daily, "sheath blight daily", "sheath_blight_daily", weather_params + new_features_daily)
    
]

# Loop through the data and create correlation plots
for df, title, filename, features in data_list:
    plot_correlations(df, title, filename, features)

# COMMAND ----------

# 13. Put correlations in a sorted manner for all data

def print_sorted_correlations(data, target_column, features, title):
    correlations = calculate_correlations(data, target_column, features)
    sorted_features = correlations.sort_values(by="Spearman's rho", key=abs, ascending=False)['Feature'].tolist()
    
    print(f"\n{title} correlations from highest to lowest:")
    print(sorted_features)

# Define the datasets and their corresponding features
datasets = {
    "Main Data": (main_data, weather_params + new_features_daily),
    "Bacterial Blight Daily": (bacterial_blight_daily, weather_params + new_features_daily),
    "Bacterial Panicle Blight Daily": (bacterial_panicle_blight_daily, weather_params + new_features_daily),
    'blast daily': (blast_daily, weather_params + new_features_daily),
    "Stem Rot Daily": (stem_rot_daily, weather_params + new_features_daily),
    "Sheath Blight Daily": (sheath_blight_daily, weather_params + new_features_daily),
}

# Print sorted correlations for each dataset
for title, (data, features) in datasets.items():
    print_sorted_correlations(data, count_column, features, title)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Selection

# COMMAND ----------

TF_ENABLE_ONEDNN_OPTS=0

# COMMAND ----------

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.neural_network import MLPRegressor

# COMMAND ----------

input_columns= new_features_daily+weather_params
target_column= ['plantix_count_7days']

# COMMAND ----------

new_features_daily

# COMMAND ----------

def count_high_correlations(data, threshold=0.9):
    # Calculate the correlation matrix
    corr_matrix = data.corr(method='spearman')
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Apply the mask and count correlations above the threshold
    high_corr_count = np.sum(np.abs(corr_matrix.where(mask)) > threshold)
    
    # Get the names of highly correlated feature pairs
    high_corr_pairs = []
    highly_correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                feature1, feature2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))
                highly_correlated_features.add(feature1)
                highly_correlated_features.add(feature2)
    
    return high_corr_count, high_corr_pairs, highly_correlated_features

# Assuming 'data' is your pandas DataFrame
high_corr_count, high_corr_pairs, highly_correlated_features = count_high_correlations(daily_data, threshold=0.9)

print(f"Number of feature pairs correlated more than 90%: {high_corr_count}")
print(f"Number of unique features involved in correlations > 90%: {len(highly_correlated_features)}")
print("\nHighly correlated feature pairs:")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.4f}")

print("\nList of features involved in high correlations:")
for feature in sorted(highly_correlated_features):
    print(f"- {feature}")

# COMMAND ----------

daily_copy= daily_data.copy()

# COMMAND ----------

daily_copy.drop(highly_correlated_features, axis=1, inplace=True)
daily_copy.columns

# COMMAND ----------

copy_columns= ['Leaf wetness probability_mean_3day_avg',
       'Leaf wetness probability_mean_7day_avg',
       'Leaf wetness probability_mean_14day_avg',
       'Leaf wetness probability_mean_30day_avg',
       'Relative Humidity_max_3day_avg', 'Relative Humidity_max_7day_avg',
       'Relative Humidity_max_14day_avg', 'Relative Humidity_max_30day_avg',
       'Relative Humidity_min_3day_avg', 'Soil Moisture_mean_3day_avg',
       'Soil Moisture_mean_7day_avg', 'Soil Temperature_mean_3day_avg',
       'Sunshine Duration_mean_7day_avg', 'Sunshine Duration_mean_14day_avg',
       'Sunshine Duration_mean_30day_avg', 'Temperature_min_3day_avg',
       'Temperature_min_7day_avg', 'Temperature_min_14day_avg',
       'Temperature_min_30day_avg', 'Precipitation Total_mean_3day_avg',
       'Precipitation Total_mean_7day_avg',
       'Temperature_max_Relative Humidity_max_Interaction',
       'High_Temperature_max_High_Relative Humidity_max', 'month',
       'season_Winter', 'season_Spring']
target_column= ['plantix_count_7days']

# COMMAND ----------

bacterial_panicle_blight_daily[copy_columns]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random forest regressor

# COMMAND ----------

def train_rf_model(data, input_columns, target_column, n_splits=4, n_bins=5):
    X = data[input_columns]
    y = data[target_column].values.ravel()

    # Create bins for stratification
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test, y_train_binned, _ = train_test_split(X, y, y_binned, test_size=0.2, random_state=40, stratify=y_binned)

    rf_model = RandomForestRegressor(n_estimators=40, random_state=1, max_depth=10)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_r2_scores = []
    cv_mse_scores = []

    for train_index, val_index in skf.split(X_train, y_train_binned):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        rf_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = rf_model.predict(X_val_fold)

        cv_r2_scores.append(r2_score(y_val_fold, y_pred_fold))
        cv_mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

    # Calculate average metrics
    r2 = np.mean(cv_r2_scores)
    mse = np.mean(cv_mse_scores)
    rmse = np.sqrt(mse)

    # Fit the model on the entire training set
    rf_model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return r2, rmse, feature_importance, cv_r2_scores

# COMMAND ----------

def forward_stepwise_selection(data, all_features, target_column, num_top_combinations=5):
    selected_features = []
    remaining_features = all_features.copy()
   
    selection_record = []
   
    print("Starting Forward Stepwise Selection")
    print("-----------------------------------")
   
    while remaining_features:
        best_new_feature = None
        best_new_r2 = -np.inf
        best_new_rmse = np.inf
        best_new_cv_r2_scores = None
       
        for feature in remaining_features:
            current_features = selected_features + [feature]
            r2, rmse, _, cv_r2_scores = train_rf_model(data, current_features, target_column)
           
            if r2 > best_new_r2:
                best_new_feature = feature
                best_new_r2 = r2
                best_new_rmse = rmse
                best_new_cv_r2_scores = cv_r2_scores
       
        selected_features.append(best_new_feature)
        remaining_features.remove(best_new_feature)
       
        selection_record.append({
            'step': len(selected_features),
            'added_feature': best_new_feature,
            'r2': best_new_r2,
            'rmse': best_new_rmse,
            'cv_r2_scores': best_new_cv_r2_scores,
            'features': selected_features.copy()
        })
       
        print(f"Step {len(selected_features)}:")
        print(f"Added feature: {best_new_feature}")
        print(f"R-squared: {best_new_r2:.4f}")
        print(f"RMSE: {best_new_rmse:.4f}")
        print(f"Cross-validation R2 scores: {list(best_new_cv_r2_scores)}")
        print("Current set of features:")
        print(", ".join(selected_features))
        print("-----------------------------------")
   
    # Sort the selection record by R-squared score in descending order
    sorted_record = sorted(selection_record, key=lambda x: x['r2'], reverse=True)
   
    # Select the top combinations
    top_combinations = sorted_record[:num_top_combinations]
   
    print("\nForward Stepwise Selection Complete")
    print(f"\nTop {num_top_combinations} Combinations:")
    for i, combination in enumerate(top_combinations, 1):
        print(f"\nRank {i}:")
        print(f"R-squared: {combination['r2']:.4f}")
        print(f"RMSE: {combination['rmse']:.4f}")
        print(f"Cross-validation R2 scores: {list(cv_r2_scores)}")
        print(f"Number of features: {len(combination['features'])}")
        print("Features:")
        print(", ".join(combination['features']))
   
    return top_combinations, pd.DataFrame(selection_record)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Bacterial Panicle Blight RF

# COMMAND ----------

# Perform train-test split first
X = bacterial_panicle_blight_daily[copy_columns]
y = bacterial_panicle_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, final_feature_importance, final_cv_r2_scores = train_rf_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")
print("\nFinal Feature Importance:")
print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'panicle_blight_rf_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (RF) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_panicle_blight_daily[[
    "Sunshine Duration_mean_7day_avg",
    "Temperature_min_30day_avg",
    "Temperature_min_7day_avg",
    "season_Spring",
    "High_Temperature_max_High_Relative Humidity_max",
    "season_Winter",
    "Temperature_min_14day_avg"
]]
y= bacterial_panicle_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = RandomForestRegressor(n_estimators=5, random_state=1, max_depth=10)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Bacterial Blight RF

# COMMAND ----------

# Perform train-test split first
X = bacterial_blight_daily[copy_columns]
y = bacterial_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, final_feature_importance, final_cv_r2_scores = train_rf_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")
print("\nFinal Feature Importance:")
print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'panicle_blight_rf_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (RF) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_blight_daily[[
    'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_30day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'season_Winter', 'month', 'Temperature_min_7day_avg'
]]
y= bacterial_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = RandomForestRegressor(n_estimators=110, random_state=1, max_depth=4, max_leaf_nodes= 23, max_features=8)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values BB rf std')
plt.legend()
plt.grid(True)
plt.show()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_rf_std_actual_pred.png')
plt.savefig(plot_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Stem Rot of Rice RF

# COMMAND ----------

# Perform train-test split first
X = stem_rot_daily[copy_columns]
y = stem_rot_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, final_feature_importance, final_cv_r2_scores = train_rf_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")
print("\nFinal Feature Importance:")
print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'panicle_blight_rf_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (RF) saved to: {plot_file_path}")

# COMMAND ----------

X= stem_rot_daily[[
    'Temperature_min_30day_avg', 'month', 'season_Winter', 'High_Temperature_max_High_Relative Humidity_max', 'season_Spring', 'Relative Humidity_max_30day_avg', 'Relative Humidity_max_14day_avg', 'Precipitation Total_mean_7day_avg', 'Temperature_min_14day_avg', 'Relative Humidity_max_7day_avg', 'Soil Moisture_mean_7day_avg', 'Sunshine Duration_mean_14day_avg'
]]
y= stem_rot_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = RandomForestRegressor(n_estimators=40, random_state=1, max_depth=10, max_leaf_nodes= 6, max_features=4)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values SR rf std')
plt.legend()
plt.grid(True)
plt.show()
plot_file_path = os.path.join(results_dir, 'stem_rot_rf_std_actual_pred.png')
plt.savefig(plot_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Blast of Rice RF

# COMMAND ----------

# Perform train-test split first
X = blast_daily[copy_columns]
y = blast_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, final_feature_importance, final_cv_r2_scores = train_rf_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")
print("\nFinal Feature Importance:")
print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'blast_rf_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (RF) saved to: {plot_file_path}")

# COMMAND ----------

X= blast_daily[[
    'Temperature_min_7day_avg', 'Sunshine Duration_mean_30day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'Sunshine Duration_mean_7day_avg','season_Winter', 'season_Spring', 'Relative Humidity_max_30day_avg', 'Soil Moisture_mean_7day_avg', 'Sunshine Duration_mean_14day_avg', 'Relative Humidity_max_14day_avg'
]]
y= blast_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = RandomForestRegressor(n_estimators=110, random_state=1, max_depth=15, max_leaf_nodes= 27, max_features=10)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values Blast rf std')
plt.legend()
plt.grid(True)
plt.show()
plot_file_path = os.path.join(results_dir, 'blast_rf_std_actual_pred.png')
plt.savefig(plot_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Sheath Blight RF

# COMMAND ----------

# Perform train-test split first
X = sheath_blight_daily[copy_columns]
y = sheath_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, final_feature_importance, final_cv_r2_scores = train_rf_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")
print("\nFinal Feature Importance:")
print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'sheath_blight_rf_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (RF) saved to: {plot_file_path}")

# COMMAND ----------

X= sheath_blight_daily[[
    'Precipitation Total_mean_7day_avg', 'month', 'Precipitation Total_mean_3day_avg', 'Relative Humidity_max_14day_avg', 'Relative Humidity_min_3day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'Leaf wetness probability_mean_30day_avg', 'Sunshine Duration_mean_30day_avg', 'season_Winter', 'Sunshine Duration_mean_14day_avg'
]]
y= sheath_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = RandomForestRegressor(n_estimators=125, random_state=1, max_depth=4, max_leaf_nodes= 9, max_features=9)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)
print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-layered Perceptron

# COMMAND ----------

def train_mlp_model(data, input_columns, target_column, n_splits=4, n_bins=5):
    X = data[input_columns]
    y = data[target_column].values.ravel()

    # Create bins for stratification
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test, y_train_binned, _ = train_test_split(X, y, y_binned, test_size=0.2, random_state=40, stratify=y_binned)

    mlp_model = MLPRegressor(hidden_layer_sizes=(150, 50), max_iter=150, random_state=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_r2_scores = []
    cv_mse_scores = []

    for train_index, val_index in skf.split(X_train, y_train_binned):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        mlp_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = mlp_model.predict(X_val_fold)

        cv_r2_scores.append(r2_score(y_val_fold, y_pred_fold))
        cv_mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

    # Calculate average metrics
    r2 = np.mean(cv_r2_scores)
    mse = np.mean(cv_mse_scores)
    rmse = np.sqrt(mse)

    # Fit the model on the entire training set
    mlp_model.fit(X_train, y_train)

    # Get feature importance
    # feature_importance = pd.DataFrame({
    #     'feature': input_columns,
    #     'importance': mlp_model.feature_importances_
    # }).sort_values('importance', ascending=False)

    return r2, rmse,None, cv_r2_scores

# COMMAND ----------

def forward_stepwise_selection(data, all_features, target_column, num_top_combinations=5):
    selected_features = []
    remaining_features = all_features.copy()
   
    selection_record = []
   
    print("Starting Forward Stepwise Selection")
    print("-----------------------------------")
   
    while remaining_features:
        best_new_feature = None
        best_new_r2 = -np.inf
        best_new_rmse = np.inf
        best_new_cv_r2_scores = None
       
        for feature in remaining_features:
            current_features = selected_features + [feature]
            r2, rmse, _, cv_r2_scores = train_mlp_model(data, current_features, target_column)
           
            if r2 > best_new_r2:
                best_new_feature = feature
                best_new_r2 = r2
                best_new_rmse = rmse
                best_new_cv_r2_scores = cv_r2_scores
       
        selected_features.append(best_new_feature)
        remaining_features.remove(best_new_feature)
       
        selection_record.append({
            'step': len(selected_features),
            'added_feature': best_new_feature,
            'r2': best_new_r2,
            'rmse': best_new_rmse,
            'cv_r2_scores': best_new_cv_r2_scores,
            'features': selected_features.copy()
        })
       
        print(f"Step {len(selected_features)}:")
        print(f"Added feature: {best_new_feature}")
        print(f"R-squared: {best_new_r2:.4f}")
        print(f"RMSE: {best_new_rmse:.4f}")
        print(f"Cross-validation R2 scores: {list(best_new_cv_r2_scores)}")
        print("Current set of features:")
        print(", ".join(selected_features))
        print("-----------------------------------")
   
    # Sort the selection record by R-squared score in descending order
    sorted_record = sorted(selection_record, key=lambda x: x['r2'], reverse=True)
   
    # Select the top combinations
    top_combinations = sorted_record[:num_top_combinations]
   
    print("\nForward Stepwise Selection Complete")
    print(f"\nTop {num_top_combinations} Combinations:")
    for i, combination in enumerate(top_combinations, 1):
        print(f"\nRank {i}:")
        print(f"R-squared: {combination['r2']:.4f}")
        print(f"RMSE: {combination['rmse']:.4f}")
        print(f"Cross-validation R2 scores: {list(cv_r2_scores)}")
        print(f"Number of features: {len(combination['features'])}")
        print("Features:")
        print(", ".join(combination['features']))
   
    return top_combinations, pd.DataFrame(selection_record)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Bacterial Panicle Blight MLP

# COMMAND ----------

# Perform train-test split first
X = bacterial_panicle_blight_daily[copy_columns]
y = bacterial_panicle_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_mlp_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'panicle_blight_mlp_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_panicle_blight_daily[[
    'Precipitation Total_mean_7day_avg', 'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_14day_avg', 'Sunshine Duration_mean_30day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'month', 'Leaf wetness probability_mean_3day_avg', 'Leaf wetness probability_mean_7day_avg', 'Soil Moisture_mean_7day_avg', 'Temperature_min_3day_avg', 'Temperature_min_14day_avg'
]]
y= bacterial_panicle_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = MLPRegressor(hidden_layer_sizes=(150, 60), max_iter=280, batch_size=13, random_state=1, activation='relu', solver='lbfgs', learning_rate='adaptive')
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values BPB mlp std')
plt.legend()
plt.grid(True)
plt.show()
plot_file_path = os.path.join(results_dir, 'BPB_mlp_std_actual_pred.png')
plt.savefig(plot_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Bacterial Blight MLP

# COMMAND ----------

# Perform train-test split first
X = bacterial_blight_daily[copy_columns]
y = bacterial_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_mlp_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_mlp_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_blight_daily[[
    'Precipitation Total_mean_7day_avg', 'Relative Humidity_min_3day_avg', 'Sunshine Duration_mean_7day_avg', 'Leaf wetness probability_mean_3day_avg', 'Precipitation Total_mean_3day_avg', 'Temperature_min_30day_avg', 'season_Winter', 'Temperature_min_14day_avg', 'month', 'Sunshine Duration_mean_14day_avg', 'season_Spring', 'Leaf wetness probability_mean_30day_avg', 'Temperature_max_Relative Humidity_max_Interaction'
]]
y= bacterial_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = MLPRegressor(hidden_layer_sizes=(350, 60), max_iter=180, batch_size=8, random_state=1, activation='relu', solver='lbfgs', learning_rate='adaptive')
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Blast of Rice mlp

# COMMAND ----------

# Perform train-test split first
X = blast_daily[copy_columns]
y = blast_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_mlp_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_mlp_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= blast_daily[[
    'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_3day_avg', 'Precipitation Total_mean_7day_avg', 'Soil Moisture_mean_3day_avg', 'Soil Moisture_mean_7day_avg', 'season_Spring', 'season_Winter', 'High_Temperature_max_High_Relative Humidity_max'
]]
y= blast_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = MLPRegressor(hidden_layer_sizes=(160, 70), max_iter=250, batch_size=8, random_state=1, activation='relu', solver='lbfgs', learning_rate='adaptive')
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Stem Rot mlp

# COMMAND ----------

# Perform train-test split first
X = stem_rot_daily[copy_columns]
y = stem_rot_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_mlp_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_mlp_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= stem_rot_daily[[
    'Precipitation Total_mean_7day_avg', 'Sunshine Duration_mean_7day_avg', 'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_3day_avg', 'Temperature_min_14day_avg', 'Leaf wetness probability_mean_7day_avg', 'month', 'Temperature_min_30day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'Temperature_min_3day_avg', 'Soil Temperature_mean_3day_avg', 'Soil Moisture_mean_7day_avg', 'Soil Moisture_mean_3day_avg'
]]
y= stem_rot_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = MLPRegressor(hidden_layer_sizes=(170, 30), max_iter=50, batch_size=8, random_state=1, activation='relu', solver='lbfgs', learning_rate='adaptive')
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Gradient Boosting

# COMMAND ----------

def train_gb_model(data, input_columns, target_column, n_splits=4, n_bins=5):
    X = data[input_columns]
    y = data[target_column].values.ravel()

    # Create bins for stratification
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test, y_train_binned, _ = train_test_split(X, y, y_binned, test_size=0.2, random_state=40, stratify=y_binned)

    gb_model = GradientBoostingRegressor(n_estimators=30, random_state=1, max_features=10)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_r2_scores = []
    cv_mse_scores = []

    for train_index, val_index in skf.split(X_train, y_train_binned):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        gb_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = gb_model.predict(X_val_fold)

        cv_r2_scores.append(r2_score(y_val_fold, y_pred_fold))
        cv_mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

    # Calculate average metrics
    r2 = np.mean(cv_r2_scores)
    mse = np.mean(cv_mse_scores)
    rmse = np.sqrt(mse)

    # Fit the model on the entire training set
    gb_model.fit(X_train, y_train)

    # Get feature importance
    # feature_importance = pd.DataFrame({
    #     'feature': input_columns,
    #     'importance': mlp_model.feature_importances_
    # }).sort_values('importance', ascending=False)

    return r2, rmse,None, cv_r2_scores

# COMMAND ----------

def forward_stepwise_selection(data, all_features, target_column, num_top_combinations=5):
    selected_features = []
    remaining_features = all_features.copy()
   
    selection_record = []
   
    print("Starting Forward Stepwise Selection")
    print("-----------------------------------")
   
    while remaining_features:
        best_new_feature = None
        best_new_r2 = -np.inf
        best_new_rmse = np.inf
        best_new_cv_r2_scores = None
       
        for feature in remaining_features:
            current_features = selected_features + [feature]
            r2, rmse, _, cv_r2_scores = train_gb_model(data, current_features, target_column)
           
            if r2 > best_new_r2:
                best_new_feature = feature
                best_new_r2 = r2
                best_new_rmse = rmse
                best_new_cv_r2_scores = cv_r2_scores
       
        selected_features.append(best_new_feature)
        remaining_features.remove(best_new_feature)
       
        selection_record.append({
            'step': len(selected_features),
            'added_feature': best_new_feature,
            'r2': best_new_r2,
            'rmse': best_new_rmse,
            'cv_r2_scores': best_new_cv_r2_scores,
            'features': selected_features.copy()
        })
       
        print(f"Step {len(selected_features)}:")
        print(f"Added feature: {best_new_feature}")
        print(f"R-squared: {best_new_r2:.4f}")
        print(f"RMSE: {best_new_rmse:.4f}")
        print(f"Cross-validation R2 scores: {list(best_new_cv_r2_scores)}")
        print("Current set of features:")
        print(", ".join(selected_features))
        print("-----------------------------------")
   
    # Sort the selection record by R-squared score in descending order
    sorted_record = sorted(selection_record, key=lambda x: x['r2'], reverse=True)
   
    # Select the top combinations
    top_combinations = sorted_record[:num_top_combinations]
   
    print("\nForward Stepwise Selection Complete")
    print(f"\nTop {num_top_combinations} Combinations:")
    for i, combination in enumerate(top_combinations, 1):
        print(f"\nRank {i}:")
        print(f"R-squared: {combination['r2']:.4f}")
        print(f"RMSE: {combination['rmse']:.4f}")
        print(f"Cross-validation R2 scores: {list(cv_r2_scores)}")
        print(f"Number of features: {len(combination['features'])}")
        print("Features:")
        print(", ".join(combination['features']))
   
    return top_combinations, pd.DataFrame(selection_record)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Bacterial Panicle Blight gb

# COMMAND ----------

# Perform train-test split first
X = bacterial_panicle_blight_daily[copy_columns]
y = bacterial_panicle_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_gb_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_mlp_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_panicle_blight_daily[[
    'Soil Moisture_mean_7day_avg', 'Sunshine Duration_mean_14day_avg', 'month', 'Precipitation Total_mean_7day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'Temperature_max_Relative Humidity_max_Interaction', 'Relative Humidity_max_7day_avg', 'Temperature_min_14day_avg'
]]
y= bacterial_panicle_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = GradientBoostingRegressor(n_estimators=30, random_state=1, max_features=10,max_depth=15)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Bacterial Blight gb

# COMMAND ----------

# Perform train-test split first
X = bacterial_blight_daily[copy_columns]
y = bacterial_blight_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_gb_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_gb_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= bacterial_blight_daily[[
    'Temperature_min_30day_avg', 'month', 'season_Winter', 'season_Spring', 'Sunshine Duration_mean_7day_avg', 'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_30day_avg'
]]
y= bacterial_blight_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = GradientBoostingRegressor(n_estimators=18, random_state=1, max_features=10,max_depth=3)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Blast of Rice gb

# COMMAND ----------

# Perform train-test split first
X = blast_daily[copy_columns]
y = blast_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_gb_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_gb_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= blast_daily[[
    'Temperature_min_30day_avg', 'month', 'season_Winter', 'season_Spring', 'Sunshine Duration_mean_7day_avg', 'Precipitation Total_mean_3day_avg', 'Leaf wetness probability_mean_30day_avg'
]]
y= blast_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = GradientBoostingRegressor(n_estimators=18, random_state=1, max_features=10,max_depth=3)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Stem Rot gb

# COMMAND ----------

# Perform train-test split first
X = stem_rot_daily[copy_columns]
y = stem_rot_daily[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create a new dataframe with only the training data
train_data = pd.concat([X_train, y_train], axis=1)

top_combinations, selection_record = forward_stepwise_selection(train_data, list(X_train.columns), 'plantix_count_7days', num_top_combinations=5)

# Train the final model with the best combination of features
best_features = top_combinations[0]['features']
final_r2, final_rmse, _, final_cv_r2_scores = train_gb_model(
    train_data, best_features, target_column
)
print("\nFinal Model Performance (using best combination):")
print(f"R-squared: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"Cross-validation R2 scores: {list(final_cv_r2_scores)}")

# Remove or comment out the feature importance print
# print("\nFinal Feature Importance:")
# print(final_feature_importance)

# Plotting the performance metrics
plt.figure(figsize=(12, 6))
plt.plot(selection_record['step'], selection_record['r2'], marker='o')
plt.title('R-squared vs Number of Features (RF)')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.grid(True)
plt.tight_layout()
plot_file_path = os.path.join(results_dir, 'bacterial_blight_gb_daily_stepwise.png')
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memory

logging.info(f"R-squared vs Number of Features plot (MLP) saved to: {plot_file_path}")

# COMMAND ----------

X= stem_rot_daily[[
    'Temperature_min_30day_avg', 'month', 'Relative Humidity_max_14day_avg', 'High_Temperature_max_High_Relative Humidity_max', 'season_Winter', 'Relative Humidity_max_30day_avg', 'season_Spring'
]]
y= stem_rot_daily['plantix_count_7days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train the final model
final_model = GradientBoostingRegressor(n_estimators=29, random_state=1, max_features=10,max_depth=5)
final_model.fit(X_train, y_train)

test_score = final_model.score(X_test, y_test)
print(f"Model Score (R-squared on test set): {test_score:.4f}")
train_score = final_model.score(X_train, y_train)
print(f"Model Score (R-squared on training set): {train_score:.4f}")

# Make predictions on the test set
y_pred = np.maximum(final_model.predict(X_test), 0)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
y_pred_train= np.maximum(final_model.predict(X_train), 0)
mse = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) on train set: {mse:.4f}")

print(y_pred)

# COMMAND ----------

