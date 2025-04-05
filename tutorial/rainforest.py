import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter

def fair_value_model(timestamps, prices, future_horizon=100000):
    """
    Stock fair value and future prediction model optimized for noisy data.
    
    Parameters:
    timestamps (array): Timestamps from the trading data
    prices (array): Mid prices from the trading data
    future_horizon (int): How far into the future to predict
    
    Returns:
    dict: Dictionary containing models and processed data
    """
    # Create a DataFrame for processing
    df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    
    # Step 1: Denoise the data using multiple techniques
    # Simple moving average
    df['sma'] = df['price'].rolling(window=15, min_periods=1).mean()
    
    # Exponential moving average (gives more weight to recent prices)
    df['ema'] = df['price'].ewm(span=20, adjust=False).mean()
    
    # Apply Savitzky-Golay filter (preserves trends better than moving averages)
    window_length = min(21, len(df) // 5 * 2 + 1)  # Must be odd
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    polyorder = min(3, window_length - 1)
    df['smoothed'] = savgol_filter(df['price'].values, window_length, polyorder)
    
    # Step 2: Calculate fair value using robust regression (less affected by outliers)
    X = np.array(df['timestamp']).reshape(-1, 1)
    y = df['smoothed']  # Use smoothed prices for more stable fair value
    
    # Huber regression handles outliers better than standard linear regression
    fair_model = HuberRegressor(epsilon=1.35)
    fair_model.fit(X, y)
    
    # Standard linear regression for comparison
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    
    # Calculate fair values using both models
    fair_values_huber = fair_model.predict(X)
    fair_values_linear = linear_model.predict(X)
    
    # Print model details
    print(f"Huber fair value model - Slope: {fair_model.coef_[0]:.8f}, Intercept: {fair_model.intercept_:.6f}")
    print(f"Linear fair value model - Slope: {linear_model.coef_[0]:.8f}, Intercept: {linear_model.intercept_:.6f}")
    
    # Step 3: Create future prediction model using polynomial regression
    future_timestamps = np.linspace(max(df['timestamp']), max(df['timestamp']) + future_horizon, 100)
    X_future = np.array(future_timestamps).reshape(-1, 1)
    
    # Create and train polynomial regression model (degree 3 as shown in the graph)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(X, y)
    
    # Predict future values
    future_values = poly_model.predict(X_future)
    
    # Step 4: Create a function to predict price at any timestamp
    def predict_price(timestamp):
        """Predict the price of stock at a given timestamp"""
        if isinstance(timestamp, (list, tuple, np.ndarray)):
            timestamps_array = np.array(timestamp).reshape(-1, 1)
        else:
            timestamps_array = np.array([[timestamp]])
            
        if np.max(timestamps_array) <= max(df['timestamp']):
            # Use fair value model for historical timestamps
            return fair_model.predict(timestamps_array)
        else:
            # Use polynomial model for future timestamps
            return poly_model.predict(timestamps_array)
    
    # Step 5: Visualize the results
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.scatter(df['timestamp'], df['price'], color='blue', alpha=0.5, label='Mid Prices')
    
    # Plot fair value line
    plt.plot(df['timestamp'], fair_values_linear, color='red', linewidth=2, label='Fair Value (Linear Regression)')
    
    # Plot future predictions
    plt.plot(future_timestamps, future_values, color='green', linewidth=2, label='Future Prediction (Polynomial Regression)')
    
    plt.title('Stock Analysis')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'fair_model': fair_model,
        'linear_model': linear_model,
        'poly_model': poly_model,
        'fair_values': fair_values_huber,
        'future_timestamps': future_timestamps,
        'future_values': future_values,
        'predict_price': predict_price,
        'processed_data': df
    }

# Example of how to use with actual data
def main():
    # Load your actual data here
    # For example: data = pd.read_csv('rainforest_resin_data.csv')
    # timestamps = data['timestamp'].values
    # prices = data['mid_price'].values
    df = pd.read_csv('/Users/shravanb/Desktop/Programming/IMC-prosperity-3/tutorial/tutorial_data.csv', sep=';', header=0, index_col=0)
    rainforest_df = df[df['product'] == 'RAINFOREST_RESIN']
    rainforest_timestamps = rainforest_df['timestamp'].values
    rainforest_prices = rainforest_df['mid_price'].values

    kelp_df = df[df['product'] == 'KELP']
    kelp_timestamps = kelp_df['timestamp'].values
    kelp_prices = kelp_df['mid_price'].values

    # For demonstration, generate synthetic data similar to the image
    # Replace this with your actual data
    #timestamps = np.linspace(0, 200000, 1000)
    #base_price = 10000  # Fair value appears to be around 10k
    #noise = np.random.normal(0, 1000, 1000)  # Substantial noise as seen in the image
    #prices = base_price + noise
    
    # Run the model
    model_results = fair_value_model(kelp_timestamps, kelp_prices) # Change between rainforest and kelp
    
    # Example predictions
    current_time = 200000
    future_time = 250000
    far_future_time = 300000
    
    print(f"\nPredictions:")
    print(f"Current fair value (t={current_time}): {model_results['predict_price'](current_time)[0]:.2f}")
    print(f"Future value (t={future_time}): {model_results['predict_price'](future_time)[0]:.2f}")
    print(f"Far future value (t={far_future_time}): {model_results['predict_price'](far_future_time)[0]:.2f}")
    
    return model_results

if __name__ == "__main__":
    results = main()



# Change the names of the model 