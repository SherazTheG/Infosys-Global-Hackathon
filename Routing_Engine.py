import json
import osmnx as ox
import networkx as nx
import folium
from geopy.distance import geodesic
from folium.plugins import MarkerCluster
from scipy.spatial import cKDTree
import numpy as np


class AStarRoutePlanner:
    def __init__(self, json_paths: dict, buffer_radius_m: int = 150):
        """
        Initialize with disruption sources
        Args:
            json_paths (dict): Paths to JSON files for different alert types
            buffer_radius_m (int): Buffer radius around alerts to penalize roads
        """
        self.obstacles = []
        self.buffer_radius = buffer_radius_m
        self.json_paths = json_paths
        self._load_obstacles()

    def _load_obstacles(self):
        """Load all obstacles from JSON files"""
        for path in self.json_paths.values():
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    lat = item.get("latitude")
                    lon = item.get("longitude")
                    if lat and lon:
                        self.obstacles.append((lat, lon))


    def _add_weights_with_penalties(self, G):
        """Optimized: Add edge weights + penalty if near obstacles using KDTree"""
        obstacle_coords = np.radians(self.obstacles)  # Convert to radians
        tree = cKDTree(obstacle_coords)

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))

        for u, v, data in G.edges(data=True):
            point_u = (G.nodes[u]['y'], G.nodes[u]['x'])
            point_v = (G.nodes[v]['y'], G.nodes[v]['x'])
            midpoint = ((point_u[0] + point_v[0]) / 2, (point_u[1] + point_v[1]) / 2)
            midpoint_rad = np.radians(midpoint)

            # Query KDTree for neighbors within radius
            idxs = tree.query_ball_point(midpoint_rad, r=self.buffer_radius / 6371000.0)  # radius in radians

            weight = data.get('length', 1)
            if idxs:
                weight *= 10  # penalty

            data['weight'] = weight


    def find_optimal_route(self, start: tuple, end: tuple, map_output: str = "a_star_route_map.html"):
        """
        Find the optimal route using A* algorithm and save a map
        Args:
            start (tuple): (lat, lon) of warehouse
            end (tuple): (lat, lon) of delivery location
            map_output (str): Output HTML file
        Returns:
            list of (lat, lon) tuples representing path
        """
        # Load street network (10km buffer)
        G = ox.graph_from_point(start, dist=10000, network_type='drive')

        # Get nearest nodes
        orig_node = ox.distance.nearest_nodes(G, X=start[1], Y=start[0])
        dest_node = ox.distance.nearest_nodes(G, X=end[1], Y=end[0])

        # Apply penalties
        self._add_weights_with_penalties(G)

        # Find path with A*
        try:
            route = nx.astar_path(G, orig_node, dest_node, weight='weight')
        except nx.NetworkXNoPath:
            print("❌ No path found avoiding obstacles!")
            return []

        # Get coordinates
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

        # Create map
        m = folium.Map(location=start, zoom_start=13)
        folium.Marker(location=start, icon=folium.Icon(color='blue'), tooltip="Start").add_to(m)
        folium.Marker(location=end, icon=folium.Icon(color='red'), tooltip="Destination").add_to(m)

        # Plot obstacles
        marker_cluster = MarkerCluster().add_to(m)
        for lat, lon in self.obstacles:
            folium.CircleMarker(location=(lat, lon), radius=5, color='orange', fill=True, fill_opacity=0.6).add_to(marker_cluster)

        # Draw route
        folium.PolyLine(route_coords, color='green', weight=5, tooltip="Optimal Path").add_to(m)

        m.save(map_output)
        print(f"✅ Route map saved as {map_output}")
        return route_coords


if __name__ == "__main__":
    # Paths to alert JSONs
    json_sources = {
        "tweets": r"C:\Users\Sheraz\Documents\pythontest\TrackSmart\models_data\twitter_alerts_output.json",
        "anomalies": r"C:\Users\Sheraz\Documents\pythontest\TrackSmart\models_data\anomaly_alerts.json",
        "traffic": r"C:\Users\Sheraz\Documents\pythontest\TrackSmart\models_data\traffic_alerts.json"
    }

    planner = AStarRoutePlanner(json_paths=json_sources)

    warehouse = (12.9816, 80.2209)
    delivery = (13.0601, 80.2492)

    route = planner.find_optimal_route(warehouse, delivery)
