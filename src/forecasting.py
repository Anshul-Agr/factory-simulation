import numpy as np
from typing import List
class DemandGenerator:
    def generate_demand(self, horizon, base_demand, model='seasonal_noise'):
        time = np.arange(horizon)
        forecast = np.full(horizon, base_demand, dtype=float)
        actuals = forecast.copy()
        if model == 'seasonal_noise':
            seasonal_component = 0.3 * base_demand * np.sin(2 * np.pi * time / 365 * 12) 
            random_noise = 0.15 * base_demand * np.random.randn(horizon)
            actuals += seasonal_component + random_noise
        elif model == 'random_walk':
            random_steps = 0.05 * base_demand * np.random.randn(horizon)
            noise_component = np.cumsum(random_steps)
            actuals += noise_component
        elif model == 'seasonal_spikes':
            seasonal_component = 0.2 * base_demand * np.sin(2 * np.pi * time / 365 * 12)
            spikes = np.random.choice([0, 4 * base_demand], size=horizon, p=[0.97, 0.03]) 
            random_noise = 0.05 * base_demand * np.random.randn(horizon)
            actuals += seasonal_component + spikes + random_noise
        elif model == 'forecast_bias':
            bias = 0.20 * base_demand  
            random_noise = 0.1 * base_demand * np.random.randn(horizon)
            actuals += bias + random_noise
        actuals = np.maximum(0, actuals)
        return forecast.tolist(), actuals.tolist()
class Forecaster:
    def simple_exponential_smoothing(self, history: List[float], alpha: float, initial_forecast: float = None) -> List[float]:
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")
        forecasts = []
        if initial_forecast is None:
            last_forecast = history[0] if history else 0
        else:
            last_forecast = initial_forecast
        forecasts.append(last_forecast)
        for i in range(len(history) - 1):
            actual_t = history[i]
            forecast_t_plus_1 = alpha * actual_t + (1 - alpha) * last_forecast
            forecasts.append(forecast_t_plus_1)
            last_forecast = forecast_t_plus_1
        return forecasts