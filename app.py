import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
import osmnx as ox
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime, timedelta
from streamlit_folium import folium_static

# Set page config
st.set_page_config(page_title="Russian Military Presence Visualization", layout="wide")

# Title
st.title("Russian Military Presence in Sumsk Oblast and Kiev Zone")

# Load the data
@st.cache_data
def load_data():
    return gpd.read_file("fixed_convoy_data.geojson")

gdf = load_data()

# Define bounding boxes
sumsk_bbox_coords = [[50.0, 33.0], [52.0, 35.5]]
kiev_bbox_coords = [50.4, 51.553167, 29.267549, 32.161026]

# Filter for points within the Sumsk Oblast and Kiev bounding boxes
sumsk_gdf = gdf.cx[sumsk_bbox_coords[0][1]:sumsk_bbox_coords[1][1], 
                   sumsk_bbox_coords[0][0]:sumsk_bbox_coords[1][0]]
kiev_gdf = gdf.cx[kiev_bbox_coords[2]:kiev_bbox_coords[3], 
                  kiev_bbox_coords[0]:kiev_bbox_coords[1]]

# Combine the two GeoDataFrames
combined_gdf = pd.concat([sumsk_gdf, kiev_gdf])

# Create a map centered between Sumsk and Kiev
center_lat = (sumsk_bbox_coords[0][0] + kiev_bbox_coords[1]) / 2
center_lon = (sumsk_bbox_coords[0][1] + kiev_bbox_coords[3]) / 2

# Function to get road network
@st.cache_data
def get_road_network(bbox_coords):
    if len(bbox_coords) == 2:
        north, south, east, west = bbox_coords[1][0], bbox_coords[0][0], bbox_coords[1][1], bbox_coords[0][1]
    else:
        south, north, west, east = bbox_coords

    return ox.graph_from_bbox(
        north, south, east, west,
        network_type='drive', simplify=True, 
        custom_filter='["highway"~"motorway|trunk|primary|secondary"]'
    )

# Get road networks and convert to GeoDataFrames
sumsk_roads = get_road_network(sumsk_bbox_coords)
kiev_roads = get_road_network(kiev_bbox_coords)
sumsk_roads_gdf = ox.graph_to_gdfs(sumsk_roads, nodes=False)
kiev_roads_gdf = ox.graph_to_gdfs(kiev_roads, nodes=False)

# Function to style roads
def road_style_function(feature):
    highway = feature['properties'].get('highway', '')
    if highway == 'motorway':
        return {'color': 'red', 'weight': 3}
    elif highway == 'trunk':
        return {'color': 'orange', 'weight': 2.5}
    elif highway == 'primary':
        return {'color': 'yellow', 'weight': 2}
    else:  # secondary
        return {'color': 'white', 'weight': 1.5}

# Cluster analysis
coords = combined_gdf.geometry.apply(lambda point: (point.y, point.x)).tolist()
db = DBSCAN(eps=0.1, min_samples=3).fit(coords)
combined_gdf['cluster'] = db.labels_

# Time-based visualization
combined_gdf['timestamp'] = pd.to_datetime(combined_gdf['verifiedDate'])
combined_gdf = combined_gdf.sort_values('timestamp')

def create_geojson_features(df):
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row.geometry.x, row.geometry.y]
            },
            'properties': {
                'time': row['timestamp'].strftime('%Y-%m-%d'),
                'popup': f"Date: {row['timestamp'].strftime('%Y-%m-%d')}<br>Description: {row['description']}"
            }
        }
        features.append(feature)
    return features

geojson_data = create_geojson_features(combined_gdf)

# Streamlit app
st.sidebar.header("Map Options")
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
show_clusters = st.sidebar.checkbox("Show Clusters", value=True)
show_timeline = st.sidebar.checkbox("Show Timeline", value=True)

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

# Add roads to the map
folium.GeoJson(sumsk_roads_gdf, style_function=road_style_function).add_to(m)
folium.GeoJson(kiev_roads_gdf, style_function=road_style_function).add_to(m)

# Add bounding boxes
folium.Rectangle(bounds=sumsk_bbox_coords, color="purple", weight=2, fill=False, popup="Sumsk Oblast").add_to(m)
folium.Rectangle(bounds=[[kiev_bbox_coords[0], kiev_bbox_coords[2]], [kiev_bbox_coords[1], kiev_bbox_coords[3]]],
                 color="green", weight=2, fill=False, popup="Kiev Zone").add_to(m)

# Heatmap layer
if show_heatmap:
    heat_data = [[row.geometry.y, row.geometry.x] for idx, row in combined_gdf.iterrows()]
    HeatMap(heat_data, name="Heatmap").add_to(m)

# Cluster visualization
if show_clusters:
    for cluster_id in set(db.labels_):
        if cluster_id != -1:  # -1 represents noise points
            cluster_points = combined_gdf[combined_gdf['cluster'] == cluster_id]
            center = cluster_points.geometry.centroid.iloc[0]
            folium.CircleMarker(
                location=[center.y, center.x],
                radius=20,
                popup=f'Cluster {cluster_id}: {len(cluster_points)} points',
                color='yellow',
                fill=True,
                fill_color='yellow'
            ).add_to(m)

# Time-based visualization
if show_timeline:
    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': geojson_data},
        period='P1D',
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True,
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Add a title and legend to the map
title_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: 180px; 
                border:2px solid grey; z-index:9999; font-size:14px; background-color:white;">
        &nbsp;<b>Legend</b><br>
        &nbsp;<i class="fa fa-square" style="color:purple"></i> Sumsk Oblast<br>
        &nbsp;<i class="fa fa-square" style="color:green"></i> Kiev Zone<br>
        &nbsp;<i class="fa fa-road" style="color:red"></i> Motorway<br>
        &nbsp;<i class="fa fa-road" style="color:orange"></i> Trunk Road<br>
        &nbsp;<i class="fa fa-road" style="color:yellow"></i> Primary Road<br>
        &nbsp;<i class="fa fa-road" style="color:white"></i> Secondary Road<br>
        &nbsp;<i class="fa fa-circle" style="color:yellow"></i> Activity Cluster
    </div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Display the map
folium_static(m, width=1200, height=800)

# Display additional information
st.sidebar.markdown("## Data Summary")
st.sidebar.write(f"Total data points: {len(combined_gdf)}")
st.sidebar.write(f"Date range: {combined_gdf['timestamp'].min().date()} to {combined_gdf['timestamp'].max().date()}")
st.sidebar.write(f"Number of clusters: {len(set(db.labels_)) - 1}")  # Subtract 1 to exclude noise points

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(combined_gdf)