# Pest Presence Prediction Model

## Overview

This project involves the development of a machine learning model designed to predict pest presence in the form of scan counts within a company's app.This is a regression model utilizing several regressors best worked for non-linear data. The prediction is based on a variety of weather conditions and temporal factors. The model focuses on five specific diseases affecting rice crops in Telangana state.

## Data Overview

The main_data is weekly but we had daily weather parameters data so we took weekly moving averages of weather parameters and then aggregated the data datewise. The whole modeling is done for aggregated data only.
The dataset used for training and evaluating the model includes the following features:

- **Date**: The specific day the data was collected.
- **State**: Geographic location, focused on Telangana.
- **District**: Subdivision within the state.
- **Crop**: Type of crop, specifically rice.
- **Disease Name**: Specific disease affecting the crop.
- **Threat Group**: Classification of pest threat.
- **Leaf Wetness**: Measurement of moisture on leaf surfaces.
- **Temperature Max/Min**: Daily maximum and minimum temperatures.
- **Relative Humidity Max/Min**: Daily maximum and minimum relative humidity levels.
- **Precipitation**: Amount of rainfall.
- **Sunshine Duration (hrs)**: Total hours of sunlight.
- **Soil Moisture**: Level of moisture in the soil.
- **Soil Temperature**: Temperature of the soil.

## Features

- Multiple machine learning models were explored such as Gradient Boosting, Multilayer Perceptron, Random Forest Regressor.
- The model was tailored for each disease to enhance prediction accuracy.
- Currently, the model is implemented for five diseases related to rice crops in Telangana.

## Usage

To utilize the model, ensure that the dataset is prepared with the required features. The model can be integrated into the company's app to provide real-time predictions of pest presence based on current weather conditions and dates.
