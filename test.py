import folium
import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import Point, Polygon

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
logging.info("Loading GeoJSON files for Kyiv and Sumsk roads and infrastructure.")
kyiv_edges_gdf = gpd.read_file("Kyiv_roads.geojson")
sumsk_edges_gdf = gpd.read_file("Sumsk_roads.geojson")
kyiv_infrastructure_gdf = gpd.read_file('kyiv_infrastructure.geojson')
sumsk_infrastructure_gdf = gpd.read_file('sumsk_infrastructure.geojson')

logging.debug(f"Kyiv infrastructure GeoDataFrame columns: {kyiv_infrastructure_gdf.columns}")
logging.debug(f"Sumsk infrastructure GeoDataFrame columns: {sumsk_infrastructure_gdf.columns}")

# Load ACLED data
logging.info("Loading ACLED data from CSV.")
df = pd.read_csv('acled-ua-war-2022-2024-page-28.csv')
filtered_df = df[(df['event_date'] == '2022-02-24') & (df['event_type'] == 'Battles')]
logging.info(f"Filtered ACLED data to {len(filtered_df)} events on 2022-02-24.")

# Define bounding boxes
kyiv_bbox_coords = [49.178744, 51.553167, 29.267549, 32.161026]  # [min_lat, max_lat, min_lon, max_lon]
sumsk_bbox_coords = [50.0, 52.0, 33.0, 35.5]  # [min_lat, max_lat, min_lon, max_lon]

# Create a map centered between Kyiv and Sumsk oblasts
logging.info("Creating map centered between Kyiv and Sumsk.")
m = folium.Map(location=[(kyiv_bbox_coords[0] + sumsk_bbox_coords[1]) / 2,
                         (kyiv_bbox_coords[2] + sumsk_bbox_coords[3]) / 2], zoom_start=7)

# Add road networks
logging.info("Adding road networks to the map.")
folium.GeoJson(kyiv_edges_gdf, name="Kyiv Roads", style_function=lambda x: {'color': 'gray', 'weight': 1}).add_to(m)
folium.GeoJson(sumsk_edges_gdf, name="Sumsk Roads", style_function=lambda x: {'color': 'gray', 'weight': 1}).add_to(m)

# Function to style infrastructure points
def style_infrastructure(feature):
    amenity = feature.get('amenity')
    landuse = feature.get('landuse')
    military = feature.get('military')
    
    if amenity == 'hospital':
        return {'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.7, 'radius': 5}
    elif amenity == 'school':
        return {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.7, 'radius': 5}
    elif amenity == 'power_plant' or landuse == 'industrial':
        return {'color': 'purple', 'fillColor': 'purple', 'fillOpacity': 0.7, 'radius': 5}
    elif military == 'base' or landuse == 'military':
        return {'color': 'green', 'fillColor': 'green', 'fillOpacity': 0.7, 'radius': 5}
    else:
        return {'color': 'orange', 'fillColor': 'orange', 'fillOpacity': 0.7, 'radius': 5}

# Add infrastructure
logging.info("Adding infrastructure points to the map.")
for gdf, name in [(kyiv_infrastructure_gdf, "Kyiv Infrastructure"), (sumsk_infrastructure_gdf, "Sumsk Infrastructure")]:
    logging.debug(f"Processing {name}")
    for _, row in gdf.iterrows():
        logging.debug(f"Processing row: {row.to_dict()}")
        style = style_infrastructure(row)
        if isinstance(row.geometry, Point):
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=style['radius'],
                color=style['color'],
                fillColor=style['fillColor'],
                fillOpacity=style['fillOpacity'],
                tooltip=f"Name: {row.get('name', 'N/A')}<br>Amenity: {row.get('amenity', 'N/A')}<br>Land Use: {row.get('landuse', 'N/A')}<br>Military: {row.get('military', 'N/A')}",
                popup=f"Name: {row.get('name', 'N/A')}<br>Amenity: {row.get('amenity', 'N/A')}<br>Land Use: {row.get('landuse', 'N/A')}<br>Military: {row.get('military', 'N/A')}",
            ).add_to(m)
        elif isinstance(row.geometry, Polygon):
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    'fillColor': style['fillColor'],
                    'color': style['color'],
                    'weight': 2,
                    'fillOpacity': style['fillOpacity']
                },
                tooltip=f"Name: {row.get('name', 'N/A')}<br>Amenity: {row.get('amenity', 'N/A')}<br>Land Use: {row.get('landuse', 'N/A')}<br>Military: {row.get('military', 'N/A')}",
                popup=f"Name: {row.get('name', 'N/A')}<br>Amenity: {row.get('amenity', 'N/A')}<br>Land Use: {row.get('landuse', 'N/A')}<br>Military: {row.get('military', 'N/A')}",
            ).add_to(m)

# Add ACLED events
logging.info("Adding ACLED events to the map.")
for _, row in filtered_df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.Icon(color='red', icon='info-sign', prefix='fa'),
        tooltip=f"Location: {row['location']}<br>Date: {row['event_date']}<br>Type: {row['event_type']}"
    ).add_to(m)

# Add bounding boxes
logging.info("Adding bounding boxes for Kyiv and Sumsk oblasts.")
folium.Rectangle(bounds=[[kyiv_bbox_coords[0], kyiv_bbox_coords[2]], [kyiv_bbox_coords[1], kyiv_bbox_coords[3]]],
                 color='black', fill=False, weight=2, tooltip="Kyiv Oblast").add_to(m)
folium.Rectangle(bounds=[[sumsk_bbox_coords[0], sumsk_bbox_coords[2]], [sumsk_bbox_coords[1], sumsk_bbox_coords[3]]],
                 color='black', fill=False, weight=2, tooltip="Sumsk Oblast").add_to(m)

# Add labels for oblasts
logging.info("Adding labels for Kyiv and Sumsk oblasts.")
folium.Marker(
    location=[(kyiv_bbox_coords[0] + kyiv_bbox_coords[1]) / 2, (kyiv_bbox_coords[2] + kyiv_bbox_coords[3]) / 2],
    icon=folium.DivIcon(html='<div style="font-size: 24px; font-weight: bold;">Kyiv Oblast</div>')
).add_to(m)

folium.Marker(
    location=[(sumsk_bbox_coords[0] + sumsk_bbox_coords[1]) / 2, (sumsk_bbox_coords[2] + sumsk_bbox_coords[3]) / 2],
    icon=folium.DivIcon(html='<div style="font-size: 24px; font-weight: bold;">Sumsk Oblast</div>')
).add_to(m)

# Add layer control
logging.info("Adding layer control to the map.")
folium.LayerControl().add_to(m)

# Save the map
logging.info("Saving the map to 'map.html'.")
m.save("map.html")

logging.info("Map generation complete.")