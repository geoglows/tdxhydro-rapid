import glob
import os
import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString

stream_files = glob.glob(r"D:\geoglows_v3\geoglows_v3\hydrography\*\streams_*.gpkg")
lake_table_path = os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data', 'lake_table.csv')

def identify_downstream_segments(multi_line_geom: MultiLineString):
    # Extract individual LineStrings from the MultiLineString
    line_segments = list(multi_line_geom.geoms)
    
    # Identify upstream segments
    upstream_segments = []
    downstream_segments = []
    
    for i, segment in enumerate(line_segments):
        is_upstream = True
        for j, other_segment in enumerate(line_segments):
            # Skip comparing the segment with itself
            if i == j:
                continue
            
            # Check if the end of the current segment matches the start of any other segment
            if segment.coords[-1] == other_segment.coords[0]:
                is_upstream = False
                break
        
        # If no downstream segment was found, it's an upstream segment
        if is_upstream:
            upstream_segments.append(segment)
        else:
            downstream_segments.append(segment)
    
    return downstream_segments

def purge_dissolved(gdf: gpd.GeoSeries):
    g = gdf['geometry']
    
    if hasattr(g, 'geoms') and len(g.geoms) >= 3 and gdf['strmOrder'] <= 2:
        if len(g.geoms) == 3:
            return g.geoms[-1]
        geoms = identify_downstream_segments(g)
        multicoords = [list(line.coords) for line in geoms]
        return LineString([item for sublist in multicoords for item in sublist]) 
    
    if isinstance(g, MultiLineString):
        multicoords = [list(line.coords) for line in g.geoms]
        return LineString([item for sublist in multicoords for item in sublist])  # Flatten coords
    
    if isinstance(g, LineString):
        return g # Return the geometry unchanged if no condition is met
    
    raise ValueError(f"Could not resolve geometry: {gdf}")

if __name__ == '__main__':
    stream_files = [f for f in stream_files if 'mapping' not in os.path.basename(f)]
    lake_outlets = set(pd.read_csv(lake_table_path)['outlet'])

    for file in (pbar := tqdm.tqdm(stream_files)):
        vpu = file.split('.')[0].split('_')[-1]
        pbar.set_description(f"Processing {vpu}")

        out_file = os.path.join(os.path.dirname(file), f'streams_mapping_{vpu}.gpkg')
        if os.path.exists(out_file):
            continue

        df: gpd.GeoDataFrame = gpd.read_file(file)
        df = df[~df['LINKNO'].isin(lake_outlets)]
        df.geometry = df.apply(purge_dissolved, axis=1)

        df.to_file(out_file, driver='GPKG')
