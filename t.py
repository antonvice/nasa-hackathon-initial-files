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
import requests
import json
import os
import rasterio
from rasterio.features import bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from scipy.interpolate import RegularGridInterpolator
from rasterio.merge import merge

import pystac_client
from planetary_computer import sign_inplace
import matplotlib.pyplot as plt
import io
import base64
from scipy.ndimage import sobel

# Set page config
st.set_page_config(page_title="Military Presence and Terrain Analysis", layout="wide")

# Title
st.title("Russian Military Presence and Terrain Analysis in Sumsk Oblast and Kiev Zone")

# Load the data
@st.cache_data
def load_data():
    return gpd.read_file("fixed_convoy_data.geojson")

gdf = load_data()

# Define bounding boxes
sumsk_bbox_coords = [[50.0, 33.0], [52.0, 35.5]]
kiev_bbox_coords = [50.4, 51.553167, 29.267549, 32.161026]

# Combine bounding boxes for terrain analysis
combined_bbox = [
    min(sumsk_bbox_coords[0][0], kiev_bbox_coords[0]),
    min(sumsk_bbox_coords[0][1], kiev_bbox_coords[2]),
    max(sumsk_bbox_coords[1][0], kiev_bbox_coords[1]),
    max(sumsk_bbox_coords[1][1], kiev_bbox_coords[3])
]

# Filter for points within the Sumsk Oblast and Kiev bounding boxes
sumsk_gdf = gdf.cx[sumsk_bbox_coords[0][1]:sumsk_bbox_coords[1][1], 
                   sumsk_bbox_coords[0][0]:sumsk_bbox_coords[1][0]]
kiev_gdf = gdf.cx[kiev_bbox_coords[2]:kiev_bbox_coords[3], 
                  kiev_bbox_coords[0]:kiev_bbox_coords[1]]

# Combine the two GeoDataFrames
combined_gdf = pd.concat([sumsk_gdf, kiev_gdf])

# Define bounding boxes
sumsk_bbox_coords = [33.0, 50.0, 35.5, 52.0]  # [min_lon, min_lat, max_lon, max_lat]
kiev_bbox_coords = [29.267549, 50.4, 32.161026, 51.553167]  # [min_lon, min_lat, max_lon, max_lat]

# Function to fetch and process terrain data

def save_terrain_data(elevation_data, transform, crs, filename):
    terrain_data = {
        'elevation_data': elevation_data.tolist(),
        'transform': transform.to_gdal(),
        'crs': crs.to_string()
    }
    with open(filename, 'w') as f:
        json.dump(terrain_data, f)

def load_terrain_data(filename):
    with open(filename, 'r') as f:
        terrain_data = json.load(f)
    elevation_data = np.array(terrain_data['elevation_data'])
    transform = rasterio.Affine.from_gdal(*terrain_data['transform'])
    crs = rasterio.crs.CRS.from_string(terrain_data['crs'])
    return elevation_data, transform, crs

# Modify the get_terrain_data function
@st.cache_data

def get_terrain_data_for_region(bbox, region_name):
    filename = f"terrain_data_{region_name}.json"
    if os.path.exists(filename):
        return load_terrain_data(filename)
    
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=sign_inplace
        )

        search = catalog.search(
            collections=["nasadem"],
            bbox=bbox,
            limit=100
        )
        items = list(search.items())

        if not items:
            st.warning(f"No NASADEM data found for {region_name}: {bbox}")
            return None, None

        src_files_to_mosaic = []
        for item in items:
            elevation_asset = item.assets["elevation"]
            src = rasterio.open(elevation_asset.href)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)
        elevation_data = mosaic[0]

        # Get the CRS of the merged raster
        crs = src_files_to_mosaic[0].crs

        # Create affine transform
        transform = rasterio.Affine(out_trans[0], out_trans[1], out_trans[2],
                                    out_trans[3], out_trans[4], out_trans[5])

        save_terrain_data(elevation_data, transform, crs, filename)
        
        return elevation_data, transform, crs

    except Exception as e:
        st.error(f"Failed to fetch terrain data for {region_name}: {str(e)}")
        import traceback
        st.write("Traceback:", traceback.format_exc())
        return None, None, None


# Function to visualize terrain data
def visualize_terrain(elevation_data, slope, aspect, roughness, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    axs[0, 0].imshow(elevation_data, cmap='terrain')
    axs[0, 0].set_title('Elevation')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(slope, cmap='viridis')
    axs[0, 1].set_title('Slope')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(aspect, cmap='hsv')
    axs[1, 0].set_title('Aspect')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(roughness, cmap='plasma')
    axs[1, 1].set_title('Roughness')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

# Function to analyze terrain suitability
def analyze_terrain_suitability(slope, roughness):
    max_suitable_slope = 15  # degrees
    max_suitable_roughness = 50  # arbitrary unit, adjust as needed
    
    slope_suitability = slope <= max_suitable_slope
    roughness_suitability = roughness <= max_suitable_roughness
    
    overall_suitability = slope_suitability & roughness_suitability
    
    return overall_suitability

# Function to create a color-coded image of terrain suitability
def create_suitability_image(suitability, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(suitability, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Suitability')
    plt.title(f'Terrain Suitability for Military Movement - {title}')
    plt.axis('off')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return f'data:image/png;base64,{img_str}'

def flatten_list(x):
    if isinstance(x, list):
        return ';'.join(map(str, x))
    return str(x)

def find_nearby_roads(point_buffer, roads):
    nearby_roads = roads[roads.intersects(point_buffer)]
    if 'highway' in nearby_roads.columns:
        road_types = nearby_roads['highway'].apply(flatten_list).unique()
        return ';'.join(road_types)
    return 'Unknown'
import numpy as np
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

def extract_terrain_features(point, elevation_data, transform):
    try:
        x, y = point.x, point.y
        
        # Create a grid of coordinates
        rows, cols = elevation_data.shape
        x_coords = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
        y_coords = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
        
        # Debug information
        st.write(f"Point coordinates: x={x}, y={y}")
        st.write(f"Elevation data shape: {elevation_data.shape}")
        st.write(f"X coordinate range: {x_coords.min()} to {x_coords.max()}")
        st.write(f"Y coordinate range: {y_coords.min()} to {y_coords.max()}")
        
        # Check if the point is within the terrain data bounds
        if x < x_coords.min() or x > x_coords.max() or y < y_coords.min() or y > y_coords.max():
            st.warning(f"Point ({x}, {y}) is outside the terrain data bounds.")
            return 0, 0  # Return default values for out-of-bounds points
        
        # Create interpolators
        regular_interpolator = RegularGridInterpolator((y_coords, x_coords), elevation_data, 
                                                       bounds_error=False, fill_value=None)
        nearest_interpolator = NearestNDInterpolator(list(zip(y_coords.ravel(), x_coords.ravel())), 
                                                     elevation_data.ravel())
        
        # Define the buffer size (1 km)
        buffer_size = 0.009  # approximately 1 km in degrees
        
        # Create a grid of points within the buffer
        x_buffer = np.linspace(x - buffer_size, x + buffer_size, 20)
        y_buffer = np.linspace(y - buffer_size, y + buffer_size, 20)
        xx, yy = np.meshgrid(y_buffer, x_buffer)  # Note the order: y first, then x
        points = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Interpolate elevation values, using nearest neighbor for out-of-bounds points
        try:
            terrain_subset = regular_interpolator(points)
        except ValueError:
            st.warning("Regular interpolation failed. Using nearest neighbor interpolation.")
            terrain_subset = nearest_interpolator(points)
        
        terrain_subset = terrain_subset.reshape(20, 20)
        
        # Calculate slope
        dy, dx = np.gradient(terrain_subset)
        slope = np.mean(np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy))))
        
        # Calculate roughness
        roughness = np.mean(np.max(terrain_subset) - np.min(terrain_subset))
        
        # Calculate suitability
        suitability = np.mean((slope <= 15) & (roughness <= 50))
        
        return slope, suitability
    except Exception as e:
        st.error(f"Error in extract_terrain_features: {str(e)}")
        return 0, 0  # Return default values in case of error



# Function to determine which terrain data to use for a given point
def get_terrain_for_point(point, sumsk_terrain, sumsk_transform, kiev_terrain, kiev_transform):
    x, y = point.x, point.y
    
    if sumsk_terrain is not None and point_in_bounds(x, y, sumsk_transform, sumsk_terrain.shape):
        return sumsk_terrain, sumsk_transform
    elif kiev_terrain is not None and point_in_bounds(x, y, kiev_transform, kiev_terrain.shape):
        return kiev_terrain, kiev_transform
    else:
        return None, None

def point_in_bounds(x, y, transform, shape):
    rows, cols = shape
    x_min, x_max = transform[2], transform[2] + transform[0] * cols
    y_min, y_max = transform[5] + transform[4] * rows, transform[5]
    return x_min <= x <= x_max and y_min <= y <= y_max

def create_comprehensive_dataset(combined_gdf, roads_gdf, terrain_data, transform, crs, structures_gdf):
    comprehensive_gdf = combined_gdf.copy()
    comprehensive_gdf['buffer'] = comprehensive_gdf.geometry.buffer(0.009)
    comprehensive_gdf['nearby_roads'] = comprehensive_gdf['buffer'].apply(lambda x: find_nearby_roads(x, roads_gdf))
    
    terrain_features = []
    for _, row in comprehensive_gdf.iterrows():
        slope, suitability = extract_terrain_features(row.geometry, terrain_data, transform, crs)
        terrain_features.append((slope, suitability))
    
    comprehensive_gdf['slope'], comprehensive_gdf['terrain_suitability'] = zip(*terrain_features)
    
    # Function to find nearby structures
    def find_nearby_structures(point_buffer, structures):
        nearby_structures = structures[structures.intersects(point_buffer)]
        if 'military' in nearby_structures.columns:
            structure_types = nearby_structures['military'].apply(flatten_list).unique()
            return ';'.join(structure_types)
        return 'Unknown'
    
    # Add nearby structures
    comprehensive_gdf['nearby_structures'] = comprehensive_gdf['buffer'].apply(lambda x: find_nearby_structures(x, structures_gdf))
    
    # Add action type (if 'description' column is available)
    if 'description' in comprehensive_gdf.columns:
        comprehensive_gdf['action_type'] = comprehensive_gdf['description'].apply(
            lambda x: 'movement' if 'move' in str(x).lower() else ('attack' if 'attack' in str(x).lower() else 'other')
        )
    
    # Drop unnecessary columns
    comprehensive_gdf = comprehensive_gdf.drop(columns=['buffer'])
    
    return comprehensive_gdf




@st.cache_data
def get_road_network(bbox_coords, cache_file):
    if os.path.exists(cache_file):
        try:
            return ox.load_graphml(cache_file)
        except Exception as e:
            st.warning(f"Error loading cached road network: {str(e)}")
            # If loading fails, we'll fetch the data again
    
    try:
        # Try to get the road network using OSMnx
        road_network = ox.graph_from_bbox(
            bbox_coords[3], bbox_coords[1], bbox_coords[2], bbox_coords[0],
            network_type='drive', simplify=True,
            custom_filter='["highway"~"motorway|trunk|primary|secondary"]'
        )
        # Save the road network to cache
        ox.save_graphml(road_network, cache_file)
        return road_network
    except Exception as e:
        st.warning(f"Error fetching road network from OSM: {str(e)}")
        st.info("Falling back to simplified road network...")
        
        # Fallback: Create a simplified road network
        bbox = box(bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
        simplified_roads = gpd.GeoDataFrame(
            geometry=[bbox.boundary],
            crs="EPSG:4326"
        )
        simplified_roads['highway'] = 'primary'
        return simplified_roads


# Define bounding boxes
or_sumsk_bbox_coords = [[50.0, 33.0], [52.0, 35.5]]
or_kiev_bbox_coords = [50.4, 51.553167, 29.267549, 32.161026]

sumsk_roads = get_road_network(or_sumsk_bbox_coords, 'sumsk_roads.graphml')
kiev_roads = get_road_network(or_kiev_bbox_coords, 'kiev_roads.graphml')

# Convert to GeoDataFrame if it's not already
if isinstance(sumsk_roads, gpd.GeoDataFrame):
    sumsk_roads_gdf = sumsk_roads
else:
    sumsk_roads_gdf = ox.graph_to_gdfs(sumsk_roads, nodes=False)

if isinstance(kiev_roads, gpd.GeoDataFrame):
    kiev_roads_gdf = kiev_roads
else:
    kiev_roads_gdf = ox.graph_to_gdfs(kiev_roads, nodes=False)

# Function to fetch military infrastructure data
@st.cache_data
def fetch_military_data(bbox, filename):
    # Check if the file already exists
    if os.path.exists(filename):
        return load_data(filename)  # Load data if it exists

    # If not, fetch data from Overpass API
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["military"]{bbox};
      way["military"]{bbox};
      relation["military"]{bbox};
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    save_data(data['elements'], filename)  # Save data to JSON
    return data['elements']

# Fetch and save military infrastructure data
sumsk_bbox = f"({or_sumsk_bbox_coords[0][0]},{or_sumsk_bbox_coords[0][1]},{or_sumsk_bbox_coords[1][0]},{or_sumsk_bbox_coords[1][1]})"
kiev_bbox = f"({or_kiev_bbox_coords[0]},{or_kiev_bbox_coords[2]},{or_kiev_bbox_coords[1]},{or_kiev_bbox_coords[3]})"
# Function to save data to disk
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to load data from disk
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)
if not os.path.exists('sumsk_military.json'):
    sumsk_military = fetch_military_data(sumsk_bbox, 'sumsk_military.json')
else:
    sumsk_military = load_data('sumsk_military.json')

if not os.path.exists('kiev_military.json'):
    kiev_military = fetch_military_data(kiev_bbox, 'kiev_military.json')
else:
    kiev_military = load_data('kiev_military.json')
# Convert military data to GeoDataFrame
def military_to_gdf(data):
    features = []
    for element in data:
        if element['type'] == 'node':
            geom = gpd.points_from_xy([element['lon']], [element['lat']])
        elif 'center' in element:
            geom = gpd.points_from_xy([element['center']['lon']], [element['center']['lat']])
        else:
            continue
        features.append({
            'geometry': geom[0],
            'properties': element.get('tags', {})
        })
    return gpd.GeoDataFrame(features)

sumsk_military_gdf = military_to_gdf(sumsk_military)
kiev_military_gdf = military_to_gdf(kiev_military)

# Combine road networks
roads_gdf = pd.concat([sumsk_roads_gdf, kiev_roads_gdf])

# Combine all data into a unified DataFrame
unified_gdf = pd.concat([combined_gdf, sumsk_military_gdf, kiev_military_gdf])


sumsk_terrain, sumsk_transform, sumsk_crs = get_terrain_data_for_region(sumsk_bbox_coords, "Sumsk")
kiev_terrain, kiev_transform, kiev_crs = get_terrain_data_for_region(kiev_bbox_coords, "Kiev")

# Merge terrain data if needed
if sumsk_terrain is not None and kiev_terrain is not None:
    # You might need to implement a function to merge these datasets
    terrain_data, transform, crs = merge(sumsk_terrain, sumsk_transform, sumsk_crs,
                                                      kiev_terrain, kiev_transform, kiev_crs)
else:
    terrain_data, transform, crs = sumsk_terrain or kiev_terrain, sumsk_transform or kiev_transform, sumsk_crs or kiev_crs

if terrain_data is not None:
    comprehensive_dataset = create_comprehensive_dataset(combined_gdf, roads_gdf, terrain_data, transform, crs, structures_gdf)
else:
    st.error("Failed to fetch terrain data for both regions.")

structures_gdf = pd.concat([sumsk_military_gdf, kiev_military_gdf])

comprehensive_dataset = create_comprehensive_dataset(combined_gdf, roads_gdf, terrain_data, transform, structures_gdf)

# Display the first few rows of the comprehensive dataset
st.subheader("Comprehensive Dataset for Military Action Prediction")
st.write(comprehensive_dataset.head())

# Display column names for debugging
st.subheader("Available Columns in the Dataset")
st.write(comprehensive_dataset.columns.tolist())

# Save the dataset to a CSV file
comprehensive_dataset.to_csv("military_action_prediction_dataset.csv", index=False)
st.success("Comprehensive dataset has been saved to 'military_action_prediction_dataset.csv'")

# Additional features for ML model (only add if columns are available)
if 'nearby_roads' in comprehensive_dataset.columns:
    comprehensive_dataset['road_density'] = comprehensive_dataset['nearby_roads'].apply(lambda x: len(str(x).split(';')))
if 'nearby_structures' in comprehensive_dataset.columns:
    comprehensive_dataset['structure_density'] = comprehensive_dataset['nearby_structures'].apply(lambda x: len(str(x).split(';')))

# Only add time-based features if 'timestamp' column is available
if 'timestamp' in comprehensive_dataset.columns:
    comprehensive_dataset['time_of_day'] = pd.to_datetime(comprehensive_dataset['timestamp']).dt.hour
    comprehensive_dataset['day_of_week'] = pd.to_datetime(comprehensive_dataset['timestamp']).dt.dayofweek

# Display summary statistics
st.subheader("Dataset Summary Statistics")
st.write(comprehensive_dataset.describe())

# Display correlation matrix for numeric columns only
st.subheader("Correlation Matrix")
numeric_columns = comprehensive_dataset.select_dtypes(include=[np.number]).columns
correlation_matrix = comprehensive_dataset[numeric_columns].corr()
st.write(correlation_matrix)

# Suggestions for ML model features
st.subheader("Suggested Features for ML Model")
st.write("Based on the available columns in your dataset, consider using:")
for col in comprehensive_dataset.columns:
    st.write(f"- {col}")

st.write("""
Additional features to consider (if data is available):
- Distance to nearest major city
- Weather conditions
- Historical conflict data for the area
- Population density of the area
""")

# Tips for building the ML model
st.subheader("Tips for Building the ML Model")
st.write("""
1. Use categorical encoding for text columns (e.g., one-hot encoding or label encoding).
2. Consider using a Random Forest or Gradient Boosting model for this type of prediction task.
3. Split your data into training and testing sets (e.g., 80% training, 20% testing).
4. Use cross-validation to ensure your model generalizes well.
5. Pay attention to class imbalance if certain action types are rare.
6. Consider using SMOTE or other resampling techniques if class imbalance is severe.
7. Evaluate your model using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
8. Use feature importance analysis to understand which factors most strongly influence military actions.
9. Consider ensemble methods or stacking multiple models for improved performance.
10. Regularly update and retrain your model as new data becomes available.
""")