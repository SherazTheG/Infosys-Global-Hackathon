import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
import folium
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TwitterNLPModel:
    def __init__(self):
        # Initialize zero-shot classification model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.disruption_labels = ["protest", "accident", "road closure", "traffic jam", "construction", "emergency", "normal"]

        # Predefined Chennai locations
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
            {
                'id': '1',
                'text': 'Heavy traffic jam at Nungambakkam area due to accident. Avoid the route!',
                'created_at': datetime.now()
            },
            {
                'id': '2',
                'text': 'Protest happening near Anna Nagar causing road closures #ChennaiTraffic',
                'created_at': datetime.now() - timedelta(minutes=15)
            },
            {
                'id': '3',
                'text': 'Construction work on OMR causing major delays. Take alternate route',
                'created_at': datetime.now() - timedelta(minutes=30)
            },
            {
                'id': '4',
                'text': 'Emergency vehicles blocking Mount Road. Traffic moving slowly',
                'created_at': datetime.now() - timedelta(minutes=45)
            },
            {
                'id': '5',
                'text': 'Water logging at Velachery junction after heavy rain. Roads flooded',
                'created_at': datetime.now() - timedelta(minutes=60)
            }
        ]

    def analyze_tweets(self) -> List[Dict]:
        tweets = self.get_mock_tweets()
        alerts = []

        for tweet in tweets:
            clean_text = self.preprocess_tweet(tweet['text'])
            classification = self.classify_event(clean_text)
            if classification['label'] != 'normal' and classification['confidence'] > 0.5:
                loc_result = self.extract_location(clean_text)
                if loc_result:
                    location_name, coords = loc_result
                else:
                    location_name = "Chennai"
                    coords = (13.0827, 80.2707)

                alerts.append({
                    'timestamp': tweet['created_at'],
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'disruption_type': classification['label'],
                    'confidence': classification['confidence'],
                    'location_name': location_name,
                    'description': clean_text,
                    'tweet_id': tweet['id'],
                })

        return alerts

    def render_map(self, alerts: List[Dict], filename: str = "traffic_map.html"):
        base_map = folium.Map(location=[13.0827, 80.2707], zoom_start=11)

        for alert in alerts:
            folium.Marker(
                location=[alert['latitude'], alert['longitude']],
                popup=folium.Popup(
                    f"<b>{alert['disruption_type'].title()}</b><br>"
                    f"<b>Location:</b> {alert['location_name']}<br>"
                    f"<b>Confidence:</b> {alert['confidence']:.2f}<br>"
                    f"<b>Description:</b> {alert['description']}",
                    max_width=300
                ),
                tooltip=alert['disruption_type'].title(),
                icon=folium.Icon(color="red" if alert['confidence'] > 0.7 else "orange")
            ).add_to(base_map)

        base_map.save(filename)
        logger.info(f"Map saved to {filename}")

def save_alerts_to_json(alerts: List[Dict], filename: str = "twitter_alerts_output.json"):
    with open(filename, "w") as f:
        json.dump(alerts, f, indent=2, default=str)

if __name__ == "__main__":
    model = TwitterNLPModel()
    alerts = model.analyze_tweets()

    print("\n=== Twitter NLP Model Alerts ===")
    for alert in alerts:
        print(f"\nAlert:")
        print(f"  Location: {alert['location_name']} ({alert['latitude']:.4f}, {alert['longitude']:.4f})")
        print(f"  Type: {alert['disruption_type']}")
        print(f"  Confidence: {alert['confidence']:.2f}")
        print(f"  Description: {alert['description']}")
        print(f"  Timestamp: {alert['timestamp']}")

    # Render the map
    model.render_map(alerts, filename="twitter_alerts_demo_map.html")

    save_alerts_to_json(alerts)

    # ✅ Optional: return alerts if running in a larger application
    # Just print to console for now, since __main__ doesn’t return
    print("\n✅ Alerts saved to 'twitter_alerts_output.json'")
    
