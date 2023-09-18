
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the data
data_normal = pd.read_csv('2023-08-30_15-25-26-030_25d.csv')

# Adjust the format of 'TimeStamp' and convert to datetime
data_normal['TimeStamp'] = pd.to_datetime(data_normal['TimeStamp'], format='%Y-%m-%d_%H-%M-%S-%f')

# Smooth the data
data_normal['CpuTemp_smooth'] = savgol_filter(data_normal['CpuTemp'], 51, 3)

# Classify the phases
def classify_phase(row, start_time):
    elapsed_time = (row['TimeStamp'] - start_time).total_seconds()
    
    if elapsed_time < 180:  # First 3 minutes
        return 'stabilize'
    elif 180 <= elapsed_time < 480:  # Next 5 minutes
        return 'workload'
    else:  # Last 5 minutes
        return 'cooldown'

start_time_normal = data_normal['TimeStamp'].iloc[0]
data_normal['Phase'] = data_normal.apply(classify_phase, start_time=start_time_normal, axis=1)

# Convert TimeStamp to seconds for regression
data_normal['Time_seconds'] = (data_normal['TimeStamp'] - data_normal['TimeStamp'].iloc[0]).dt.total_seconds()

# Linear regression for cooldown phase of normal data
X_normal = data_normal[data_normal['Phase'] == 'cooldown']['Time_seconds'].values.reshape(-1, 1)
y_normal = data_normal[data_normal['Phase'] == 'cooldown']['CpuTemp_smooth'].values
model_normal = LinearRegression().fit(X_normal, y_normal)
y_pred_normal = model_normal.predict(X_normal)

slope_normal = model_normal.coef_[0]
intercept_normal = model_normal.intercept_
r2_normal = r2_score(y_normal, y_pred_normal)

regression_values = {
    'Normal Data': {
        'Slope': slope_normal,
        'Intercept': intercept_normal,
        'R^2 Value': r2_normal
    }
}

# Function to plot data with regression and display regression values
def plot_data_with_regression_and_values(data, title, regression_values):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot smoothed data
    ax.plot(data['TimeStamp'], data['CpuTemp_smooth'], label='Smoothed CPU Temp')
    
    # Highlight the phases with different background colors
    stabilize_end = data[data['Phase'] == 'stabilize']['TimeStamp'].iloc[-1]
    workload_end = data[data['Phase'] == 'workload']['TimeStamp'].iloc[-1]
    
    ax.axvspan(data['TimeStamp'].min(), stabilize_end, alpha=0.2, color='green', label='Stabilize')
    ax.axvspan(stabilize_end, workload_end, alpha=0.2, color='yellow', label='Workload')
    ax.axvspan(workload_end, data['TimeStamp'].max(), alpha=0.2, color='red', label='Cooldown')
    
    # Linear regression for cooldown phase
    cooldown_data = data[data['Phase'] == 'cooldown']
    X = cooldown_data['Time_seconds'].values.reshape(-1, 1)
    y = cooldown_data['CpuTemp_smooth'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    ax.plot(cooldown_data['TimeStamp'], y_pred, color='blue', linestyle='--', label='Regression Line')
    
    # Display the regression values
    text = f"Slope: {regression_values['Slope']:.4f}\nIntercept: {regression_values['Intercept']:.4f}\nR^2 Value: {regression_values['R^2 Value']:.4f}"
    ax.text(0.7, 0.2, text, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))
    
    ax.set_title(title)
    ax.set_xlabel('TimeStamp')
    ax.set_ylabel('CPU Temp')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Plot the smoothed data with regression and values for both datasets
plot_data_with_regression_and_values(data_normal, 'Normal Data with Linear Regression for Cooldown Phase', regression_values['Normal Data'])

