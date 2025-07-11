import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
from folium.plugins import MarkerCluster
import folium
import json
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeliveryAnomalyDetector:
    def __init__(self):
        """Initialize the anomaly detection model with multiple detection algorithms"""
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Chennai delivery route coordinates (mock data)
        self.delivery_routes = {
            'D001': {
                'route_name': 'Nungambakkam to T.Nagar',
                'coordinates': [(13.0569, 80.2412), (13.0418, 80.2341)],
                'avg_delivery_time': 25  # minutes
            },
            'D002': {
                'route_name': 'Anna Nagar to Adyar',
                'coordinates': [(13.0850, 80.2101), (13.0067, 80.2206)],
                'avg_delivery_time': 35  # minutes
            },
            'D003': {
                'route_name': 'Velachery to OMR',
                'coordinates': [(12.9816, 80.2209), (12.8956, 80.2267)],
                'avg_delivery_time': 30  # minutes
            }
        }
        
        self.is_trained = False
        
    def load_delivery_data(self, csv_path: str) -> pd.DataFrame:
        """Load delivery delay data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} delivery records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self.generate_mock_data()
    
    def generate_mock_data(self) -> pd.DataFrame:
        """Generate mock delivery data for testing"""
        logger.info("Generating mock delivery data")
        rows = []
        delivery_ids = ["D001", "D002", "D003"]
        
        for delivery_id in delivery_ids:
            for ts in pd.date_range("2025-07-01 10:00", "2025-07-07 23:00", freq="H"):
                # Normal delay pattern
                delay = np.random.normal(loc=5, scale=2)
                
                # Add anomalies (5% chance)
                if np.random.rand() < 0.05:
                    delay += np.random.uniform(15, 45)  # Significant delay spike
                
                # Add time-based patterns
                hour = ts.hour
                if 8 <= hour <= 10 or 18 <= hour <= 20:  # Rush hours
                    delay += np.random.uniform(2, 8)
                
                rows.append([ts, delivery_id, max(0, round(delay, 2))])
        
        df = pd.DataFrame(rows, columns=["timestamp", "delivery_id", "delay_minutes"])
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(8, 10)) | (df['hour'].between(18, 20))).astype(int)
        
        # Delivery-specific features
        for delivery_id in df['delivery_id'].unique():
            mask = df['delivery_id'] == delivery_id
            df.loc[mask, 'avg_route_time'] = self.delivery_routes.get(delivery_id, {}).get('avg_delivery_time', 25)
        
        # Rolling statistics (last 24 hours)
        df = df.sort_values(['delivery_id', 'timestamp'])
        df['rolling_mean_24h'] = df.groupby('delivery_id')['delay_minutes'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        df['rolling_std_24h'] = df.groupby('delivery_id')['delay_minutes'].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )
        df['rolling_std_24h'] = df['rolling_std_24h'].fillna(0)
        
        # Deviation from expected
        df['delay_deviation'] = df['delay_minutes'] - df['avg_route_time']
        df['delay_z_score'] = (df['delay_minutes'] - df['rolling_mean_24h']) / (df['rolling_std_24h'] + 1e-8)
        
        return df
    
    def detect_anomalies_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest"""
        feature_cols = [
            'delay_minutes', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'avg_route_time', 'rolling_mean_24h', 'rolling_std_24h', 'delay_deviation', 'delay_z_score'
        ]
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Detect anomalies
        anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        df['is_anomaly_iso'] = (anomaly_labels == -1).astype(int)
        df['anomaly_score'] = anomaly_scores
        
        return df
    
    def detect_anomalies_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using statistical methods"""
        df = df.copy()
        
        # Z-score based detection
        df['is_anomaly_zscore'] = (np.abs(df['delay_z_score']) > 2.5).astype(int)
        
        # IQR based detection
        for delivery_id in df['delivery_id'].unique():
            mask = df['delivery_id'] == delivery_id
            delays = df.loc[mask, 'delay_minutes']
            
            Q1 = delays.quantile(0.25)
            Q3 = delays.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df.loc[mask, 'is_anomaly_iqr'] = (
                (delays < lower_bound) | (delays > upper_bound)
            ).astype(int)
        
        return df
    
    def detect_anomalies_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using DBSCAN clustering"""
        feature_cols = ['delay_minutes', 'hour', 'delay_deviation']
        
        for delivery_id in df['delivery_id'].unique():
            mask = df['delivery_id'] == delivery_id
            X = df.loc[mask, feature_cols].fillna(0)
            
            if len(X) > 10:  # Need minimum samples for clustering
                X_scaled = StandardScaler().fit_transform(X)
                cluster_labels = self.dbscan.fit_predict(X_scaled)
                
                # Points labeled as -1 are anomalies
                df.loc[mask, 'is_anomaly_dbscan'] = (cluster_labels == -1).astype(int)
            else:
                df.loc[mask, 'is_anomaly_dbscan'] = 0
        
        return df
    
    def combine_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple anomaly detection methods"""
        anomaly_cols = ['is_anomaly_iso', 'is_anomaly_zscore', 'is_anomaly_iqr', 'is_anomaly_dbscan']
        
        # Fill missing values
        for col in anomaly_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        # Ensemble approach: majority voting
        df['anomaly_vote_count'] = df[anomaly_cols].sum(axis=1)
        df['is_anomaly_final'] = (df['anomaly_vote_count'] >= 2).astype(int)  # At least 2 methods agree
        
        # Confidence score
        df['anomaly_confidence'] = df['anomaly_vote_count'] / len(anomaly_cols)
        
        return df
    
    def flag_anomalous_routes(self, df: pd.DataFrame) -> List[Dict]:
        """Flag routes with anomalous deliveries"""
        anomalies = df[df['is_anomaly_final'] == 1].copy()
        
        flagged_routes = []
        
        for _, row in anomalies.iterrows():
            delivery_id = row['delivery_id']
            route_info = self.delivery_routes.get(delivery_id, {})
            
            # Get route coordinates
            coordinates = route_info.get('coordinates', [(13.0827, 80.2707)])  # Default to Chennai center
            
            for i, coord in enumerate(coordinates):
                flagged_routes.append({
                    'timestamp': row['timestamp'],
                    'delivery_id': delivery_id,
                    'route_name': route_info.get('route_name', f'Route {delivery_id}'),
                    'latitude': coord[0],
                    'longitude': coord[1],
                    'coordinate_type': 'start' if i == 0 else 'end',
                    'delay_minutes': row['delay_minutes'],
                    'anomaly_confidence': row['anomaly_confidence'],
                    'anomaly_score': row.get('anomaly_score', 0),
                    'delay_deviation': row['delay_deviation'],
                    'description': f"Anomalous delivery delay detected: {row['delay_minutes']:.1f} minutes"
                })
        
        return flagged_routes
    
    def analyze_delivery_delays(self, csv_path: str = None) -> List[Dict]:
        """Main method to analyze delivery delays and detect anomalies"""
        # Load data
        if csv_path:
            df = self.load_delivery_data(csv_path)
        else:
            df = self.generate_mock_data()
        
        if df.empty:
            logger.error("No data available for analysis")
            return []
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Apply different anomaly detection methods
        df = self.detect_anomalies_isolation_forest(df)
        df = self.detect_anomalies_statistical(df)
        df = self.detect_anomalies_clustering(df)
        
        # Combine results
        df = self.combine_anomaly_scores(df)
        
        # Flag anomalous routes
        flagged_routes = self.flag_anomalous_routes(df)
        
        # Log results
        total_anomalies = len(df[df['is_anomaly_final'] == 1])
        logger.info(f"Detected {total_anomalies} anomalous deliveries out of {len(df)} total deliveries")
        logger.info(f"Flagged {len(flagged_routes)} route coordinates")
        
        return flagged_routes
    
    def get_anomaly_summary(self, csv_path: str = None) -> Dict:
        """Get summary statistics of anomalies"""
        if csv_path:
            df = self.load_delivery_data(csv_path)
        else:
            df = self.generate_mock_data()
        
        df = self.engineer_features(df)
        df = self.detect_anomalies_isolation_forest(df)
        df = self.detect_anomalies_statistical(df)
        df = self.detect_anomalies_clustering(df)
        df = self.combine_anomaly_scores(df)
        
        summary = {
            'total_deliveries': len(df),
            'total_anomalies': len(df[df['is_anomaly_final'] == 1]),
            'anomaly_rate': len(df[df['is_anomaly_final'] == 1]) / len(df) * 100,
            'avg_anomaly_delay': df[df['is_anomaly_final'] == 1]['delay_minutes'].mean(),
            'max_anomaly_delay': df[df['is_anomaly_final'] == 1]['delay_minutes'].max(),
            'anomalies_by_route': df[df['is_anomaly_final'] == 1].groupby('delivery_id').size().to_dict(),
            'anomalies_by_hour': df[df['is_anomaly_final'] == 1].groupby('hour').size().to_dict()
        }
        
        return summary
    
    def generate_anomaly_map(self, flagged_routes: List[Dict], output_file: str = "anomaly_alert_map.html"):
        """Generate Leaflet map for delivery anomaly coordinates"""
        if not flagged_routes:
            logger.warning("No anomalies to plot on map.")
            return

        # Base map centered on Chennai
        m = folium.Map(location=[13.0827, 80.2707], zoom_start=11)
        marker_cluster = MarkerCluster().add_to(m)

        for alert in flagged_routes:
            lat = alert['latitude']
            lon = alert['longitude']
            desc = alert['description']
            delay = alert['delay_minutes']
            conf = alert['anomaly_confidence']
            route = alert['route_name']
            coord_type = alert['coordinate_type']

            popup_html = f"""
            <b>Route:</b> {route}<br>
            <b>Type:</b> {coord_type}<br>
            <b>Delay:</b> {delay:.1f} min<br>
            <b>Confidence:</b> {conf:.2f}<br>
            <b>Description:</b> {desc}
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="red",
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)

        m.save(output_file)
        logger.info(f"Anomaly map saved to {output_file}")



# Example usage and testing
if __name__ == "__main__":
    # Initialize the anomaly detector
    detector = DeliveryAnomalyDetector()
    
    # Analyze delivery delays (using mock data if no CSV provided)
    print("\n=== Delivery Anomaly Detection Results ===")
    
    # You can specify your CSV path here
    csv_path = r"C:\Users\Sheraz\Documents\pythontest\TrackSmart\delays.csv"
    
    try:
        flagged_routes = detector.analyze_delivery_delays(csv_path)
    except:
        print("Using mock data (CSV not found)")
        flagged_routes = detector.analyze_delivery_delays()
    
    # Display results
    for route in flagged_routes:
        print(f"\n🚨 Anomaly Alert:")
        print(f"  Route: {route['route_name']}")
        print(f"  Delivery ID: {route['delivery_id']}")
        print(f"  Location: {route['latitude']:.4f}, {route['longitude']:.4f}")
        print(f"  Delay: {route['delay_minutes']:.1f} minutes")
        print(f"  Confidence: {route['anomaly_confidence']:.2f}")
        print(f"  Timestamp: {route['timestamp']}")
        print(f"  Description: {route['description']}")
    
    # Get summary statistics
    print(f"\n=== Summary Statistics ===")
    try:
        summary = detector.get_anomaly_summary(csv_path)
    except:
        summary = detector.get_anomaly_summary()
    
    print(f"Total Deliveries: {summary['total_deliveries']}")
    print(f"Total Anomalies: {summary['total_anomalies']}")
    print(f"Anomaly Rate: {summary['anomaly_rate']:.2f}%")
    print(f"Average Anomaly Delay: {summary['avg_anomaly_delay']:.1f} minutes")
    print(f"Max Anomaly Delay: {summary['max_anomaly_delay']:.1f} minutes")
    print(f"Anomalies by Route: {summary['anomalies_by_route']}")
    print(f"Anomalies by Hour: {summary['anomalies_by_hour']}")
    
    print(f"\n✅ Anomaly detection complete. Found {len(flagged_routes)} flagged route coordinates.")
    with open("anomaly_alerts.json", "w") as f:
        json.dump(flagged_routes, f, default=str, indent=4)

    print("\nFlagged route anomalies saved to 'anomaly_alerts.json'")
    detector.generate_anomaly_map(flagged_routes, output_file="anomaly_alert_map.html")
    print("🗺️ Anomaly map saved to 'anomaly_alert_map.html'")
