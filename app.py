import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
import osmnx as ox
from sklearn.cluster import DBSCAN
import numpy as np
from streamlit_folium import st_folium
import requests
import json
import os
import rasterio
import matplotlib.pyplot as plt
import io
import base64
import pystac_client
from planetary_computer import sign_inplace
from shapely.geometry import box
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set page config
st.set_page_config(page_title="Military Presence and Terrain Analysis", layout="wide")

# Load the data
@st.cache_data
def load_convoy_data():
    return gpd.read_file("fixed_convoy_data.geojson")

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


# Function to visualize terrain data
def visualize_terrain(elevation_data, slope, roughness, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    axs[0, 0].imshow(elevation_data, cmap='terrain')
    axs[0, 0].set_title('Elevation')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(slope, cmap='viridis')
    axs[0, 1].set_title('Slope')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(roughness, cmap='plasma')
    axs[1, 0].set_title('Roughness')
    axs[1, 0].axis('off')
    
    axs[1, 1].axis('off')  # Hide the unused subplot
    
    plt.tight_layout()
    return fig

# Function to analyze terrain suitability
def analyze_terrain_suitability(slope, roughness):
    max_suitable_slope = 15  # degrees
    max_suitable_roughness = np.percentile(roughness, 75)  # Using 75th percentile as threshold
    
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

# Function to fetch military infrastructure data
@st.cache_data
def fetch_infrastructure_data(bbox, filename):
    if os.path.exists(filename):
        return load_data(filename)

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["military"]{bbox};
      way["military"]{bbox};
      relation["military"]{bbox};
      node["railway"="station"]{bbox};
      way["railway"="station"]{bbox};
      node["amenity"="hospital"]{bbox};
      way["amenity"="hospital"]{bbox};
      node["aeroway"="aerodrome"]{bbox};
      way["aeroway"="aerodrome"]{bbox};
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    save_data(data['elements'], filename)
    return data['elements']

# Function to save data to disk
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to load data from disk
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def infrastructure_to_gdf(data):
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



@st.cache_data
def get_terrain_features(_points, sd_sumsk_bbox_coords, sd_kiev_bbox_coords):
    features = np.zeros((len(_points), 2))
    for area, bbox in [("Sumy Oblast", sd_sumsk_bbox_coords), ("Kyiv Oblast", sd_kiev_bbox_coords)]:
        terrain_file = f"terrain_analysis_{area.replace(' ', '_').lower()}.json"
        if os.path.exists(terrain_file):
            with open(terrain_file, 'r') as f:
                terrain_data = json.load(f)
            slope = np.array(terrain_data['slope'])
            roughness = np.array(terrain_data['roughness'])
        else:
            elevation_data, slope, roughness, terrain_suitability, transform, crs = get_and_analyze_terrain(bbox, area)
            if slope is not None and roughness is not None:
                # Save terrain analysis data
                terrain_data = {
                    'slope': slope.tolist(),
                    'roughness': roughness.tolist(),
                    'terrain_suitability': terrain_suitability.tolist()
                }
                with open(terrain_file, 'w') as f:
                    json.dump(terrain_data, f)
                
                # Save terrain analysis images
                plt.figure(figsize=(10, 10))
                plt.imshow(terrain_suitability, cmap='RdYlGn', interpolation='nearest')
                plt.colorbar(label='Suitability')
                plt.title(f'Terrain Suitability for Military Movement - {area}')
                plt.axis('off')
                plt.savefig(f"terrain_suitability_{area.replace(' ', '_').lower()}.png")
                plt.close()
            else:
                continue

        area_box = box(*bbox)
        mask = _points.intersects(area_box)
        if mask.any():
            # Interpolate terrain features to match the points
            x = np.linspace(bbox[0], bbox[2], slope.shape[1])
            y = np.linspace(bbox[1], bbox[3], slope.shape[0])
            from scipy.interpolate import RegularGridInterpolator
            slope_interp = RegularGridInterpolator((y, x), slope)
            roughness_interp = RegularGridInterpolator((y, x), roughness)
            
            points_in_area = _points[mask]
            interp_coords = np.array([(p.y, p.x) for p in points_in_area])
            features[mask] = np.column_stack((slope_interp(interp_coords), roughness_interp(interp_coords)))

    return features

# Function to fetch and process terrain data
@st.cache_data
def get_terrain_data(bbox, region_name):
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
            limit=1  # Reduce the number of items to process
        )
        items = list(search.items())

        if not items:
            st.warning(f"No NASADEM data found for {region_name}: {bbox}")
            return None, None, None

        elevation_asset = items[0].assets["elevation"]
        with rasterio.open(elevation_asset.href) as src:
            # Read the data at a lower resolution
            elevation_data = src.read(1, out_shape=(src.height // 8, src.width // 8))
            transform = src.transform * src.transform.scale(
                (src.width / elevation_data.shape[1]),
                (src.height / elevation_data.shape[0])
            )
            crs = src.crs

        save_terrain_data(elevation_data, transform, crs, filename)
        
        return elevation_data, transform, crs

    except Exception as e:
        st.error(f"Failed to fetch terrain data for {region_name}: {str(e)}")
        return None, None, None

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

# Function to extract terrain features
@st.cache_data
def extract_terrain_features(elevation_data):
    # Calculate slope
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    
    # Calculate roughness as local standard deviation
    from scipy.ndimage import uniform_filter
    
    roughness = np.zeros_like(elevation_data)
    elevation_mean = uniform_filter(elevation_data, size=5)
    elevation_sqr_mean = uniform_filter(elevation_data**2, size=5)
    roughness = np.sqrt(elevation_sqr_mean - elevation_mean**2)
    
    return slope, roughness

def get_map_key(center_lat, center_lon, sumsk_bbox_coords, kiev_bbox_coords):
    """Generate a unique key for map caching."""
    return f"{center_lat}_{center_lon}_{sumsk_bbox_coords}_{kiev_bbox_coords}"

@st.cache_data
def create_map_data(center_lat, center_lon, sumsk_bbox_coords, kiev_bbox_coords):
    """Create and return the base map data."""
    return {
        "center": [center_lat, center_lon],
        "sumsk_bbox": sumsk_bbox_coords,
        "kiev_bbox": kiev_bbox_coords,
    }

def create_and_display_map(map_data, sumsk_roads_gdf, kiev_roads_gdf, combined_gdf, db, 
                           geojson_data, sumsk_infrastructure_gdf, kiev_infrastructure_gdf,
                           show_heatmap, show_clusters, show_timeline, show_military):
    logger.info("Creating map...")
    try:
        center_lat, center_lon = map_data["center"]
        sumsk_bbox_coords = map_data["sumsk_bbox"]
        kiev_bbox_coords = map_data["kiev_bbox"]

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        
        # Add roads to the map
        logger.info("Adding roads to the map...")
        folium.GeoJson(sumsk_roads_gdf, style_function=road_style_function).add_to(m)
        folium.GeoJson(kiev_roads_gdf, style_function=road_style_function).add_to(m)

        # Add bounding boxes
        logger.info("Adding bounding boxes...")
        folium.Rectangle(bounds=sumsk_bbox_coords, color="purple", weight=2, fill=False, popup="Sumsk Oblast").add_to(m)
        folium.Rectangle(bounds=[[kiev_bbox_coords[0], kiev_bbox_coords[2]], [kiev_bbox_coords[1], kiev_bbox_coords[3]]],
                         color="green", weight=2, fill=False, popup="Kiev Zone").add_to(m)

        # Heatmap layer
        if show_heatmap:
            logger.info("Adding heatmap...")
            heat_data = [[row.geometry.y, row.geometry.x] for idx, row in combined_gdf.iterrows()]
            HeatMap(heat_data, name="Heatmap").add_to(m)

        # Cluster visualization
        if show_clusters:
            logger.info("Adding cluster visualization...")
            for cluster_id in set(db.labels_):
                if cluster_id != -1:  # -1 represents noise points
                    cluster_points = combined_gdf[combined_gdf['cluster'] == cluster_id]
                    # Project to a suitable UTM zone
                    cluster_points_proj = cluster_points.to_crs(cluster_points.estimate_utm_crs())
                    center = cluster_points_proj.geometry.centroid
                    # Convert the center point back to EPSG:4326
                    center = gpd.GeoSeries(center, crs=cluster_points_proj.crs).to_crs("EPSG:4326").iloc[0]
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
            logger.info("Adding time-based visualization...")
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


        # Infrastructure visualization
        if show_military:
            logger.info("Adding infrastructure visualization...")
            infrastructure_cluster = MarkerCluster(name="Infrastructure").add_to(m)

            def get_icon(properties):
                if 'military' in properties:
                    return folium.Icon(color='black', icon='info-sign')
                elif 'railway' in properties and properties['railway'] == 'station':
                    return folium.Icon(color='blue', icon='train')
                elif 'amenity' in properties and properties['amenity'] == 'hospital':
                    return folium.Icon(color='red', icon='plus')
                elif 'aeroway' in properties and properties['aeroway'] == 'aerodrome':
                    return folium.Icon(color='purple', icon='plane')
                else:
                    return folium.Icon(color='gray', icon='question-sign')

            for gdf in [sumsk_infrastructure_gdf, kiev_infrastructure_gdf]:
                for _, row in gdf.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=f"Type: {', '.join(row['properties'].keys())}",
                        icon=get_icon(row['properties'])
                    ).add_to(infrastructure_cluster)

        # Add layer control
        layer_control = folium.LayerControl()
        layer_control.add_to(m)

        # Add a title and legend to the map
        title_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: 220px; 
                        border:2px solid grey; z-index:9999; font-size:14px; background-color:white;">
                &nbsp;<b>Legend</b><br>
                &nbsp;<i class="fa fa-square" style="color:purple"></i> Sumsk Oblast<br>
                &nbsp;<i class="fa fa-square" style="color:green"></i> Kiev Zone<br>
                &nbsp;<i class="fa fa-road" style="color:red"></i> Motorway<br>
                &nbsp;<i class="fa fa-road" style="color:orange"></i> Trunk Road<br>
                &nbsp;<i class="fa fa-road" style="color:yellow"></i> Primary Road<br>
                &nbsp;<i class="fa fa-road" style="color:white"></i> Secondary Road<br>
                &nbsp;<i class="fa fa-circle" style="color:yellow"></i> Activity Cluster<br>
                &nbsp;<i class="fa fa-info-sign" style="color:black"></i> Military Infrastructure<br>
                &nbsp;<i class="fa fa-map" style="color:green"></i> Terrain Suitability
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        map_filename = "military_presence_map.html"
        m.save(map_filename)
        logger.info(f"Map saved to {map_filename}")

        return m, map_filename
    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        return None, None



@st.cache_data
def get_and_analyze_terrain(bbox, region_name):
    elevation_data, transform, crs = get_terrain_data(bbox, region_name)
    
    if elevation_data is not None:
        slope, roughness = extract_terrain_features(elevation_data)
        terrain_suitability = analyze_terrain_suitability(slope, roughness)
        return elevation_data, slope, roughness, terrain_suitability, transform, crs
    else:
        return None, None, None, None, None, None
    

# Function to create comprehensive dataset

def create_comprehensive_dataset(combined_gdf, roads_gdf, structures_gdf, sd_sumsk_bbox_coords, sd_kiev_bbox_coords):
    logger.info("Creating comprehensive dataset...")
    
    # Project to a suitable UTM zone
    utm_crs = combined_gdf.estimate_utm_crs()
    comprehensive_gdf = combined_gdf.to_crs(utm_crs)
    roads_gdf = roads_gdf.to_crs(utm_crs)
    structures_gdf = structures_gdf.to_crs(utm_crs)
    
    comprehensive_gdf['buffer'] = comprehensive_gdf.geometry.buffer(1000)  # 1000 meters buffer
    
    # Vectorized nearby roads calculation
    logger.info("Calculating nearby roads...")
    spatial_index = roads_gdf.sindex
    nearby_roads = comprehensive_gdf['buffer'].apply(lambda x: list(spatial_index.intersection(x.bounds)))
    
    logger.debug(f"Sample of nearby_roads: {nearby_roads.head()}")
    logger.debug(f"roads_gdf columns: {roads_gdf.columns}")
    logger.debug(f"Sample of roads_gdf: {roads_gdf.head()}")

    def process_nearby_roads(x):
        if len(x) > 0:
            try:
                return ';'.join(roads_gdf.iloc[x]['highway'].astype(str).unique())
            except Exception as e:
                logger.error(f"Error processing nearby roads: {e}")
                logger.debug(f"Problematic data: {x}")
                return ''
        return ''

    comprehensive_gdf['nearby_roads'] = nearby_roads.apply(process_nearby_roads)
    
    # Vectorized terrain feature extraction
    logger.info("Extracting terrain features...")
    terrain_features = get_terrain_features(comprehensive_gdf.geometry, sd_sumsk_bbox_coords, sd_kiev_bbox_coords)
    comprehensive_gdf['slope'] = terrain_features[:, 0]
    comprehensive_gdf['terrain_suitability'] = terrain_features[:, 1]
    
    # Vectorized nearby structures calculation
    logger.info("Calculating nearby structures...")
    structures_index = structures_gdf.sindex
    nearby_structures = comprehensive_gdf['buffer'].apply(lambda x: list(structures_index.intersection(x.bounds)))
    
    logger.debug(f"Sample of nearby_structures: {nearby_structures.head()}")
    logger.debug(f"structures_gdf columns: {structures_gdf.columns}")
    logger.debug(f"Sample of structures_gdf: {structures_gdf.head()}")

    def process_nearby_structures(x):
        if len(x) > 0:
            try:
                return ';'.join(structures_gdf.iloc[x]['military'].astype(str).unique())
            except Exception as e:
                logger.error(f"Error processing nearby structures: {e}")
                logger.debug(f"Problematic data: {x}")
                return ''
        return ''

    comprehensive_gdf['nearby_structures'] = nearby_structures.apply(process_nearby_structures)
    
    comprehensive_gdf['action_type'] = comprehensive_gdf['description'].str.lower().apply(
        lambda x: 'movement' if 'move' in str(x) else ('attack' if 'attack' in str(x) else 'other')
    )
    
    comprehensive_gdf = comprehensive_gdf.drop(columns=['buffer'])
    
    # Project back to EPSG:4326
    comprehensive_gdf = comprehensive_gdf.to_crs("EPSG:4326")
    
    logger.info("Comprehensive dataset creation completed.")
    logger.debug(f"Final comprehensive_gdf columns: {comprehensive_gdf.columns}")
    logger.debug(f"Sample of final comprehensive_gdf: {comprehensive_gdf.head()}")
    
    return comprehensive_gdf



def main():
    st.title("Russian Military Presence and Terrain Analysis in Sumsk Oblast and Kiev Zone")

    # Load data
    gdf = load_convoy_data()

    # Define bounding boxes
    sumsk_bbox_coords = [[50.0, 33.0], [52.0, 35.5]]
    kiev_bbox_coords = [50.4, 51.553167, 29.267549, 32.161026]
    sd_sumsk_bbox_coords = [33.0, 50.0, 35.5, 52.0]
    sd_kiev_bbox_coords = [29.267549, 50.4, 32.161026, 51.553167]

    # Filter and combine data
    sumsk_gdf = gdf.cx[sumsk_bbox_coords[0][1]:sumsk_bbox_coords[1][1], 
                       sumsk_bbox_coords[0][0]:sumsk_bbox_coords[1][0]]
    kiev_gdf = gdf.cx[kiev_bbox_coords[2]:kiev_bbox_coords[3], 
                      kiev_bbox_coords[0]:kiev_bbox_coords[1]]
    combined_gdf = pd.concat([sumsk_gdf, kiev_gdf])
    combined_gdf = combined_gdf.set_crs("EPSG:4326", allow_override=True)

    # Create map centered between Sumsk and Kiev
    center_lat = (sumsk_bbox_coords[0][0] + kiev_bbox_coords[1]) / 2
    center_lon = (sumsk_bbox_coords[0][1] + kiev_bbox_coords[3]) / 2

    # Get road networks
    sumsk_roads = get_road_network(sumsk_bbox_coords, 'sumsk_roads.graphml')
    kiev_roads = get_road_network(kiev_bbox_coords, 'kiev_roads.graphml')

    # Convert to GeoDataFrame if needed
    sumsk_roads_gdf = sumsk_roads if isinstance(sumsk_roads, gpd.GeoDataFrame) else ox.graph_to_gdfs(sumsk_roads, nodes=False)
    kiev_roads_gdf = kiev_roads if isinstance(kiev_roads, gpd.GeoDataFrame) else ox.graph_to_gdfs(kiev_roads, nodes=False)

    # Cluster analysis
    coords = combined_gdf.geometry.apply(lambda point: (point.y, point.x)).tolist()
    db = DBSCAN(eps=0.1, min_samples=3).fit(coords)
    combined_gdf['cluster'] = db.labels_

    # Time-based visualization
    combined_gdf['timestamp'] = pd.to_datetime(combined_gdf['verifiedDate'])
    combined_gdf = combined_gdf.sort_values('timestamp')
    geojson_data = create_geojson_features(combined_gdf)

    # Fetch infrastructure data
    sumsk_bbox = f"({sumsk_bbox_coords[0][0]},{sumsk_bbox_coords[0][1]},{sumsk_bbox_coords[1][0]},{sumsk_bbox_coords[1][1]})"
    kiev_bbox = f"({kiev_bbox_coords[0]},{kiev_bbox_coords[2]},{kiev_bbox_coords[1]},{kiev_bbox_coords[3]})"
    sumsk_infrastructure = fetch_infrastructure_data(sumsk_bbox, 'sumsk_infrastructure.json')
    kiev_infrastructure = fetch_infrastructure_data(kiev_bbox, 'kiev_infrastructure.json')
    sumsk_infrastructure_gdf = infrastructure_to_gdf(sumsk_infrastructure).set_crs("EPSG:4326", allow_override=True)
    kiev_infrastructure_gdf = infrastructure_to_gdf(kiev_infrastructure).set_crs("EPSG:4326", allow_override=True)

    # Create placeholders
    map_placeholder = st.empty()
    analysis_placeholder = st.empty()

    # Sidebar
    st.sidebar.header("Map Options")
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
    show_clusters = st.sidebar.checkbox("Show Clusters", value=True)
    show_timeline = st.sidebar.checkbox("Show Timeline", value=True)
    show_military = st.sidebar.checkbox("Show Infrastructure", value=True)
    show_terrain = st.sidebar.checkbox("Show Terrain Suitability", value=True)

    # Create map centered between Sumsk and Kiev
    center_lat = (sumsk_bbox_coords[0][0] + kiev_bbox_coords[1]) / 2
    center_lon = (sumsk_bbox_coords[0][1] + kiev_bbox_coords[3]) / 2

    # Create map data
    map_data = create_map_data(center_lat, center_lon, sumsk_bbox_coords, kiev_bbox_coords)

    # Generate a unique key for the map
    map_key = get_map_key(center_lat, center_lon, sumsk_bbox_coords, kiev_bbox_coords)


    if 'map' not in st.session_state:
        st.session_state.map, map_filename = create_and_display_map(map_data, sumsk_roads_gdf, kiev_roads_gdf, combined_gdf, db, 
                                                                    geojson_data, sumsk_infrastructure_gdf, kiev_infrastructure_gdf,
                                                                    show_heatmap, show_clusters, show_timeline, show_military)
    
    
    with map_placeholder.container():
        st.subheader("Interactive Map")
        if st.session_state.map is not None:
            components.html(
                f"""
                <div id="map-container"></div>
                <script>
                    function loadMap() {{
                        try {{
                            var mapContent = {open(map_filename, 'r').read()};
                            document.getElementById('map-container').innerHTML = mapContent;
                        }} catch (error) {{
                            console.error('Error loading map:', error);
                            document.getElementById('map-container').innerHTML = 'Error loading map. Please check the console for details.';
                        }}
                    }}
                    loadMap();
                </script>
                """,
                height=800
            )
            st.success(f"Map saved to {map_filename}")
        else:
            st.error("Failed to create the map. Please check the logs for more information.")

    # Perform terrain analysis
    with analysis_placeholder.container():
        perform_terrain_analysis()

    # Display additional information
    display_additional_info(combined_gdf, sumsk_infrastructure_gdf, kiev_infrastructure_gdf, db, sumsk_roads_gdf, kiev_roads_gdf, sd_sumsk_bbox_coords, sd_kiev_bbox_coords)


def perform_terrain_analysis():
    st.header("Terrain Analysis")
    sd_sumsk_bbox_coords = [33.0, 50.0, 35.5, 52.0]
    sd_kiev_bbox_coords = [29.267549, 50.4, 32.161026, 51.553167]
    
    for area, bbox in [("Sumy Oblast", sd_sumsk_bbox_coords), ("Kyiv Oblast", sd_kiev_bbox_coords)]:
        st.subheader(f"Terrain Analysis for {area}")
        
        elevation_data, slope, roughness, terrain_suitability, transform, crs = get_and_analyze_terrain(bbox, f"{area}")

        if elevation_data is not None:
            terrain_fig = visualize_terrain(elevation_data, slope, roughness, area)
            st.pyplot(terrain_fig)
            
            suitability_image = create_suitability_image(terrain_suitability, area)
            if suitability_image:
                st.image(suitability_image, caption=f"Terrain Suitability Map - {area}", use_column_width=True)
            else:
                st.write(f"Terrain suitability image could not be generated for {area}.")

            st.markdown("## Terrain Suitability Analysis")
            st.write(f"""
            The terrain suitability analysis for {area} considers two main factors:
            1. **Slope**: Areas with a slope of 15 degrees or less are considered suitable for military movement.
            2. **Roughness**: Areas with low terrain roughness are considered more suitable.

            This analysis is based on NASADEM data, which provides global elevation information.

            Green areas on the suitability map indicate terrain that is more suitable for military movement, while red areas indicate less suitable terrain.
            """)

            st.markdown("## Interpretation and Limitations")
            st.write("""
            - This analysis provides a high-level overview of terrain suitability for military movement based on slope and roughness.
            - It doesn't account for other factors that might affect military operations, such as land cover, weather conditions, tactical considerations, or temporary obstacles.
            - The resolution of the NASADEM data (approximately 30 meters) may not capture all local variations in terrain.
            - This tool should be used in conjunction with other intelligence and on-the-ground information for comprehensive planning.
            """)
        else:
            st.write(f"Terrain data could not be fetched or processed for {area}.")

def display_additional_info(combined_gdf, sumsk_infrastructure_gdf, kiev_infrastructure_gdf, db, sumsk_roads_gdf, kiev_roads_gdf, sd_sumsk_bbox_coords, sd_kiev_bbox_coords):
    st.sidebar.markdown("## Data Summary")
    st.sidebar.write(f"Total data points: {len(combined_gdf)}")
    st.sidebar.write(f"Date range: {combined_gdf['timestamp'].min().date()} to {combined_gdf['timestamp'].max().date()}")
    st.sidebar.write(f"Number of clusters: {len(set(db.labels_)) - 1}")
    st.sidebar.write(f"Infrastructure points: {len(sumsk_infrastructure_gdf) + len(kiev_infrastructure_gdf)}")

    # Create the comprehensive dataset
        # Create the comprehensive dataset
    structures_gdf = pd.concat([sumsk_infrastructure_gdf, kiev_infrastructure_gdf])
    roads_gdf = pd.concat([sumsk_roads_gdf, kiev_roads_gdf])

    comprehensive_dataset = create_comprehensive_dataset(combined_gdf, roads_gdf, structures_gdf, sd_sumsk_bbox_coords, sd_kiev_bbox_coords)
    
    # Display the comprehensive dataset
    st.subheader("Comprehensive Dataset for Military Action Prediction")
    st.write(comprehensive_dataset.head())

    # Save the dataset to a CSV file
    comprehensive_dataset.to_csv("military_action_prediction_dataset.csv", index=False)
    st.success("Comprehensive dataset has been saved to 'military_action_prediction_dataset.csv'")

    # Additional features for ML model
    comprehensive_dataset['road_density'] = comprehensive_dataset['nearby_roads'].apply(lambda x: len(str(x).split(';')))
    comprehensive_dataset['structure_density'] = comprehensive_dataset['nearby_structures'].apply(lambda x: len(str(x).split(';')))
    if 'timestamp' in comprehensive_dataset.columns:
        comprehensive_dataset['time_of_day'] = pd.to_datetime(comprehensive_dataset['timestamp']).dt.hour
        comprehensive_dataset['day_of_week'] = pd.to_datetime(comprehensive_dataset['timestamp']).dt.dayofweek

    # Display summary statistics and correlation matrix
    st.subheader("Dataset Summary Statistics")
    st.write(comprehensive_dataset.describe())

    st.subheader("Correlation Matrix")
    numeric_columns = comprehensive_dataset.select_dtypes(include=[np.number]).columns
    correlation_matrix = comprehensive_dataset[numeric_columns].corr()
    st.write(correlation_matrix)
if __name__ == "__main__":
    main()