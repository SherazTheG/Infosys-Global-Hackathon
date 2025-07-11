import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from prophet import Prophet
import matplotlib.pyplot as plt
import json
import warnings
import folium
from folium.plugins import MarkerCluster
warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TimeSeriesTrafficModel:
    """Time-series forecasting model for traffic speed prediction on major roads"""
    
    def __init__(self, data_file_path: str = None):
        """
        Initialize the time-series forecasting model
        
        Args:
            data_file_path: Path to the traffic data CSV file
        """
        self.data_file_path = data_file_path
        self.models = {}  # Store trained models for each location
        self.location_coordinates = {}  # Store coordinates for each location
        
        # Define major roads in Chennai with their coordinates
        self.major_roads = {
            'OMR': {
                'coordinates': [(12.8956, 80.2267), (12.7500, 80.2500)],  # Old Mahabalipuram Road
                'name': 'Old Mahabalipuram Road'
            },
            'ECR': {
                'coordinates': [(12.8270, 80.2420), (12.7000, 80.3000)],  # East Coast Road
                'name': 'East Coast Road'
            },
            'GST_Road': {
                'coordinates': [(12.9165, 80.1315), (12.8500, 80.1000)],  # GST Road
                'name': 'GST Road'
            },
            'Anna_Salai': {
                'coordinates': [(13.0569, 80.2412), (13.0200, 80.2600)],  # Anna Salai/Mount Road
                'name': 'Anna Salai'
            },
            'Poonamallee_High_Road': {
                'coordinates': [(13.0698, 80.1947), (13.1000, 80.1500)],  # Poonamallee High Road
                'name': 'Poonamallee High Road'
            },
            'Velachery_Road': {
                'coordinates': [(12.9816, 80.2209), (12.9500, 80.2000)],  # Velachery Road
                'name': 'Velachery Road'
            },
            'Sardar_Patel_Road': {
                'coordinates': [(13.0382, 80.1595), (13.0200, 80.1800)],  # Sardar Patel Road
                'name': 'Sardar Patel Road'
            },
            'Arcot_Road': {
                'coordinates': [(13.0524, 80.2123), (13.0300, 80.1900)],  # Arcot Road
                'name': 'Arcot Road'
            }
        }
        
        # Traffic speed thresholds
        self.slow_speed_threshold = 20  # km/h - below this is considered slow
        self.very_slow_speed_threshold = 10  # km/h - below this is critical
        
    def generate_traffic_data(self, location_id: str, days: int = 30) -> pd.DataFrame:
        """Generate realistic traffic data for a specific location (enhanced from your existing code)"""
        np.random.seed(42)  # For reproducible results
        
        # Create datetime range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="H")
        
        rows = []
        base_speed = np.random.uniform(30, 50)  # Base speed for this location
        
        for ts in dates:
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Simulate speed dips during peak hours (enhanced logic)
            if hour in [8, 9, 17, 18, 19]:  # Peak hours
                speed_factor = 0.5  # Significant reduction
            elif hour in [10, 11, 16, 20]:  # Semi-peak hours
                speed_factor = 0.7
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5, 6]:  # Night hours
                speed_factor = 1.3  # Faster at night
            else:
                speed_factor = 1.0
            
            # Weekend effects
            if day_of_week in [5, 6]:  # Saturday, Sunday
                if hour in [10, 11, 12, 13, 14, 15]:  # Weekend shopping hours
                    speed_factor *= 0.8
                else:
                    speed_factor *= 1.2
            
            # Add random variation
            speed = base_speed * speed_factor + np.random.uniform(-5, 5)
            speed = max(5, speed)  # Ensure positive speed
            
            # Add occasional random slowdowns (accidents, etc.)
            if np.random.random() < 0.02:  # 2% chance of random slowdown
                speed *= 0.3
            
            rows.append([ts, location_id, round(speed, 2)])
        
        df = pd.DataFrame(rows, columns=["timestamp", "location_id", "avg_speed"])
        return df
    
    def load_or_generate_data(self) -> pd.DataFrame:
        """Load traffic data from file or generate mock data"""
        if self.data_file_path:
            try:
                df = pd.read_csv(self.data_file_path)
                logger.info(f"Loaded traffic data from {self.data_file_path}")
                return df
            except Exception as e:
                logger.error(f"Error loading data file: {e}")
        
        # Generate mock data for all major roads
        all_data = []
        for location_id in self.major_roads.keys():
            location_data = self.generate_traffic_data(location_id)
            all_data.append(location_data)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save generated data
        output_file = "generated_traffic_data.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Generated and saved traffic data to {output_file}")
        
        return combined_df
    
    def prepare_data_for_prophet(self, df: pd.DataFrame, location_id: str) -> pd.DataFrame:
        """Prepare data for Prophet model (based on your existing code)"""
        # Filter data for specific location
        location_data = df[df["location_id"] == location_id].copy()
        
        # Rename columns for Prophet
        location_data = location_data.rename(columns={"timestamp": "ds", "avg_speed": "y"})
        
        # Ensure timestamp is datetime
        location_data['ds'] = pd.to_datetime(location_data['ds'])
        
        # Sort by date
        location_data = location_data.sort_values('ds').reset_index(drop=True)
        
        return location_data[['ds', 'y']]
    
    def train_model(self, location_id: str) -> bool:
        """Train Prophet model for a specific location"""
        try:
            # Load data
            df = self.load_or_generate_data()
            
            # Prepare data for this location
            prophet_data = self.prepare_data_for_prophet(df, location_id)
            
            if prophet_data.empty:
                logger.warning(f"No data available for location {location_id}")
                return False
            
            # Initialize and configure Prophet model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=0.95,
                seasonality_mode='multiplicative'
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='hourly', period=24, fourier_order=8)
            
            # Train the model
            model.fit(prophet_data)
            
            # Store the trained model
            self.models[location_id] = model
            
            logger.info(f"Successfully trained model for {location_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {location_id}: {e}")
            return False
    
    def predict_speed(self, location_id: str, hours_ahead: int = 6) -> Dict:
        """Predict traffic speed for a specific location"""
        if location_id not in self.models:
            logger.error(f"No trained model found for {location_id}")
            return {}
        
        try:
            # Create future dataframe
            future = pd.date_range(
                start=datetime.now(),
                periods=hours_ahead,
                freq='H'
            )
            future_df = pd.DataFrame({'ds': future})
            
            # Make predictions
            forecast = self.models[location_id].predict(future_df)
            
            # Extract predictions
            predictions = []
            for idx, row in forecast.iterrows():
                predictions.append({
                    'timestamp': row['ds'],
                    'predicted_speed': row['yhat'],
                    'lower_bound': row['yhat_lower'],
                    'upper_bound': row['yhat_upper']
                })
            
            return {
                'location_id': location_id,
                'predictions': predictions,
                'model_info': {
                    'training_date': datetime.now(),
                    'forecast_horizon': hours_ahead
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting speed for {location_id}: {e}")
            return {}
    
    def detect_slow_traffic_alerts(self, location_id: str, hours_ahead: int = 6) -> List[Dict]:
        """Detect and flag slow traffic predictions"""
        alerts = []
        
        # Get predictions
        prediction_data = self.predict_speed(location_id, hours_ahead)
        
        if not prediction_data:
            return alerts
        
        # Get location coordinates
        road_info = self.major_roads.get(location_id, {})
        coordinates = road_info.get('coordinates', [(13.0827, 80.2707)])  # Default to Chennai center
        road_name = road_info.get('name', location_id)
        
        # Check each prediction
        for prediction in prediction_data['predictions']:
            predicted_speed = prediction['predicted_speed']
            
            # Determine alert level
            if predicted_speed <= self.very_slow_speed_threshold:
                alert_level = 'critical'
                confidence = 0.9
            elif predicted_speed <= self.slow_speed_threshold:
                alert_level = 'warning'
                confidence = 0.7
            else:
                continue  # No alert needed
            
            # Use midpoint of road coordinates
            lat = sum(coord[0] for coord in coordinates) / len(coordinates)
            lon = sum(coord[1] for coord in coordinates) / len(coordinates)
            
            alert = {
                'timestamp': prediction['timestamp'],
                'latitude': lat,
                'longitude': lon,
                'location_id': location_id,
                'road_name': road_name,
                'predicted_speed': predicted_speed,
                'alert_level': alert_level,
                'confidence': confidence,
                'description': f"Predicted slow traffic on {road_name}: {predicted_speed:.1f} km/h",
                'source': 'timeseries_forecasting'
            }
            
            alerts.append(alert)
        
        return alerts
    
    def train_all_models(self) -> Dict[str, bool]:
        """Train models for all major roads"""
        results = {}
        
        logger.info("Training models for all major roads...")
        
        for location_id in self.major_roads.keys():
            logger.info(f"Training model for {location_id}...")
            results[location_id] = self.train_model(location_id)
        
        successful_trainings = sum(results.values())
        logger.info(f"Successfully trained {successful_trainings}/{len(results)} models")
        
        return results
    
    def get_all_traffic_alerts(self, hours_ahead: int = 6) -> List[Dict]:
        """Get traffic alerts for all locations"""
        all_alerts = []
        
        for location_id in self.models.keys():
            alerts = self.detect_slow_traffic_alerts(location_id, hours_ahead)
            all_alerts.extend(alerts)
        
        logger.info(f"Generated {len(all_alerts)} traffic alerts across all locations")
        return all_alerts
    
    def visualize_predictions(self, location_id: str, hours_ahead: int = 24):
        """Visualize speed predictions for a location"""
        if location_id not in self.models:
            logger.error(f"No trained model found for {location_id}")
            return
        
        try:
            # Get historical data
            df = self.load_or_generate_data()
            prophet_data = self.prepare_data_for_prophet(df, location_id)
            
            # Create future dataframe
            future = self.models[location_id].make_future_dataframe(periods=hours_ahead, freq='H')
            
            # Make predictions
            forecast = self.models[location_id].predict(future)
            
            # Plot
            fig = self.models[location_id].plot(forecast)
            plt.title(f'Traffic Speed Forecast for {location_id}')
            plt.ylabel('Speed (km/h)')
            plt.xlabel('Time')
            
            # Add threshold lines
            plt.axhline(y=self.slow_speed_threshold, color='orange', linestyle='--', 
                       label=f'Slow Traffic Threshold ({self.slow_speed_threshold} km/h)')
            plt.axhline(y=self.very_slow_speed_threshold, color='red', linestyle='--', 
                       label=f'Critical Threshold ({self.very_slow_speed_threshold} km/h)')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'traffic_forecast_{location_id}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualization saved as traffic_forecast_{location_id}.png")
            
        except Exception as e:
            logger.error(f"Error creating visualization for {location_id}: {e}")

    def generate_alert_map(self, alerts: List[Dict], output_file: str = "traffic_alert_map.html"):
        """Generate a Leaflet map using Folium with all traffic alerts"""
        if not alerts:
            logger.warning("No alerts to visualize on the map.")
            return

        # Center map on Chennai
        m = folium.Map(location=[13.0827, 80.2707], zoom_start=11)
        marker_cluster = MarkerCluster().add_to(m)

        for alert in alerts:
            lat = alert['latitude']
            lon = alert['longitude']
            road = alert['road_name']
            speed = alert['predicted_speed']
            level = alert['alert_level'].capitalize()
            desc = alert['description']
            color = 'red' if alert['alert_level'] == 'critical' else 'orange'

            popup_html = f"""
            <b>Road:</b> {road}<br>
            <b>Speed:</b> {speed:.1f} km/h<br>
            <b>Alert:</b> {level}<br>
            <b>Description:</b> {desc}
            """

            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)

        m.save(output_file)
        logger.info(f"Traffic alert map saved to {output_file}")

if __name__ == "__main__":
    # Initialize the model
    model = TimeSeriesTrafficModel()
    
    # Train models for all locations
    training_results = model.train_all_models()
    
    print("\n=== Model Training Results ===")
    for location, success in training_results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{location}: {status}")
    
    # Generate traffic alerts
    alerts = model.get_all_traffic_alerts(hours_ahead=6)
    
    print(f"\n=== Traffic Alerts (Next 6 hours) ===")
    print(f"Total alerts: {len(alerts)}")
    
    for alert in alerts:
        print(f"\nAlert for {alert['road_name']}:")
        print(f"  Location: ({alert['latitude']:.4f}, {alert['longitude']:.4f})")
        print(f"  Predicted Speed: {alert['predicted_speed']:.2f} km/h")
        print(f"  Alert Level: {alert['alert_level'].upper()}")
        print(f"  Description: {alert['description']}")
    
    # Optional: Save alerts to a JSON file for API usage
    with open("traffic_alerts.json", "w") as f:
        json.dump(alerts, f, default=str, indent=4)
    
    print("\nAlerts saved to 'traffic_alerts.json'")
    model.generate_alert_map(alerts, output_file="traffic_alert_map.html")
    print("Map saved to 'traffic_alert_map.html'")
