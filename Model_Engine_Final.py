import re
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- Twitter NLP Disruption Detection --------------------- #

class TwitterNLPModel:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.disruption_labels = ["protest", "accident", "road closure", "traffic jam", "construction", "emergency", "normal"]
        self.mock_locations = {
            'nungambakkam': (13.0569, 80.2412),
            'anna nagar': (13.0850, 80.2101),
            't nagar': (13.0418, 80.2341),
            'adyar': (13.0067, 80.2206),
            'velachery': (12.9816, 80.2209),
            'tambaram': (12.9249, 80.1000),
            'chrompet': (12.9516, 80.1462),
            'omr': (12.8956, 80.2267),
            'ecr': (12.8270, 80.2420),
            'mount road': (13.0569, 80.2412),
            'velachery junction': (12.9800, 80.2210)
        }

    def preprocess_tweet(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"@\w+", '', text)
        return ' '.join(text.split())

    def classify_event(self, text: str) -> Dict:
        try:
            result = self.classifier(text, self.disruption_labels)
            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0],
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"label": "normal", "confidence": 0.0}

    def extract_location(self, text: str) -> Optional[Tuple[str, Tuple[float, float]]]:
        text = text.lower()
        for loc in self.mock_locations:
            if loc in text:
                return loc.title(), self.mock_locations[loc]
        return None

    def get_mock_tweets(self) -> List[Dict]:
        return [
            {'id': '1', 'text': 'Heavy traffic jam at Nungambakkam area due to accident.', 'created_at': datetime.now()},
            {'id': '2', 'text': 'Protest happening near Anna Nagar causing road closures', 'created_at': datetime.now() - timedelta(minutes=15)},
            {'id': '3', 'text': 'Construction work on OMR causing major delays.', 'created_at': datetime.now() - timedelta(minutes=30)},
            {'id': '4', 'text': 'Emergency vehicles blocking Mount Road. Traffic moving slowly', 'created_at': datetime.now() - timedelta(minutes=45)},
            {'id': '5', 'text': 'Water logging at Velachery junction after rain. Roads flooded', 'created_at': datetime.now() - timedelta(minutes=60)}
        ]

    def analyze_tweets(self) -> List[Dict]:
        tweets = self.get_mock_tweets()
        alerts = []

        for tweet in tweets:
            clean_text = self.preprocess_tweet(tweet['text'])
            classification = self.classify_event(clean_text)
            if classification['label'] != 'normal' and classification['confidence'] > 0.5:
                loc_result = self.extract_location(clean_text)
                location_name, coords = loc_result if loc_result else ("Chennai", (13.0827, 80.2707))
                alerts.append({
                    'timestamp': tweet['created_at'],
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'disruption_type': classification['label'],
                    'confidence': classification['confidence'],
                    'location_name': location_name,
                    'description': clean_text,
                    'source': 'twitter'
                })
        return alerts

# --------------------- Time-Series Forecasting Stub --------------------- #

class TimeSeriesTrafficForecaster:
    def __init__(self):
        self.hotspots = {
            "nungambakkam - t.nagar": (13.0418, 80.2341),
            "anna nagar - adyar": (13.045, 80.215),
            "velachery - omr": (12.935, 80.224)
        }

    def analyze_traffic(self) -> List[Dict]:
        # Simulated forecasts
        alerts = []
        for road, coords in self.hotspots.items():
            # Simulate low speed condition
            predicted_speed = np.random.uniform(4, 9)  # km/h
            if predicted_speed < 10:
                alerts.append({
                    "timestamp": datetime.now(),
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "disruption_type": "low traffic speed",
                    "forecast_speed": predicted_speed,
                    "road_segment": road,
                    "description": f"Predicted slow traffic on {road} ({predicted_speed:.1f} km/h)",
                    "source": "forecast"
                })
        return alerts

# --------------------- Anomaly Detection Model --------------------- #

class DeliveryAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.delivery_routes = {
            'D001': {'route_name': 'Nungambakkam to T.Nagar', 'coordinates': [(13.0569, 80.2412), (13.0418, 80.2341)], 'avg_delivery_time': 25},
            'D002': {'route_name': 'Anna Nagar to Adyar', 'coordinates': [(13.0850, 80.2101), (13.0067, 80.2206)], 'avg_delivery_time': 35},
            'D003': {'route_name': 'Velachery to OMR', 'coordinates': [(12.9816, 80.2209), (12.8956, 80.2267)], 'avg_delivery_time': 30}
        }

    def generate_mock_data(self) -> pd.DataFrame:
        rows = []
        delivery_ids = ["D001", "D002", "D003"]
        for delivery_id in delivery_ids:
            for ts in pd.date_range("2025-07-01 10:00", "2025-07-01 23:00", freq="H"):
                delay = np.random.normal(loc=5, scale=2)
                if np.random.rand() < 0.1:
                    delay += np.random.uniform(15, 45)
                rows.append([ts, delivery_id, max(0, round(delay, 2))])
        return pd.DataFrame(rows, columns=["timestamp", "delivery_id", "delay_minutes"])

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df['timestamp'].dt.hour
        df['delay_deviation'] = df.apply(lambda row: row['delay_minutes'] - self.delivery_routes[row['delivery_id']]['avg_delivery_time'], axis=1)
        return df

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[['delay_minutes', 'hour', 'delay_deviation']]
        X_scaled = self.scaler.fit_transform(X)
        df['is_anomaly'] = self.isolation_forest.fit_predict(X_scaled)
        df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)
        return df[df['is_anomaly'] == 1]

    def analyze_delivery_delays(self) -> List[Dict]:
        df = self.generate_mock_data()
        df = self.engineer_features(df)
        anomalies = self.detect_anomalies(df)

        alerts = []
        for _, row in anomalies.iterrows():
            route_info = self.delivery_routes[row['delivery_id']]
            for i, coord in enumerate(route_info['coordinates']):
                alerts.append({
                    "timestamp": row['timestamp'],
                    "latitude": coord[0],
                    "longitude": coord[1],
                    "disruption_type": "delivery delay anomaly",
                    "delay_minutes": row['delay_minutes'],
                    "route_name": route_info['route_name'],
                    "coordinate_type": "start" if i == 0 else "end",
                    "description": f"Delay anomaly: {row['delay_minutes']} min",
                    "source": "anomaly"
                })
        return alerts

# --------------------- Main ML Engine --------------------- #

def run_combined_ml_engine() -> List[Dict]:
    all_alerts = []

    # Twitter NLP
    try:
        nlp_model = TwitterNLPModel()
        all_alerts += nlp_model.analyze_tweets()
    except Exception as e:
        logger.error(f"NLP model error: {e}")

    # Time-Series Forecasting
    try:
        traffic_model = TimeSeriesTrafficForecaster()
        all_alerts += traffic_model.analyze_traffic()
    except Exception as e:
        logger.error(f"Forecast model error: {e}")

    # Anomaly Detection
    try:
        anomaly_model = DeliveryAnomalyDetector()
        all_alerts += anomaly_model.analyze_delivery_delays()
    except Exception as e:
        logger.error(f"Anomaly model error: {e}")

    # Save to JSON
    with open("combined_disruption_alerts.json", "w") as f:
        json.dump(all_alerts, f, indent=2, default=str)

    print(f"\n✅ Combined alerts saved to 'combined_disruption_alerts.json'")
    return all_alerts

# --------------------- Run It --------------------- #

if __name__ == "__main__":
    alerts = run_combined_ml_engine()
    for a in alerts[:5]:
        print(f"\n🚨 {a['disruption_type'].title()} @ ({a['latitude']}, {a['longitude']})\n  → {a['description']}")
