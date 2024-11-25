import glob
import json
import logging
import os
import sys

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point

import tdxhydrorapid as rp

gpkg_dir = 'test/gpkgs'
gpq_dir = '/Volumes/EB406_T7_3/geoglows_v3/parquets'
save_dir = 'test/'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects

    returns length in meters
    """
    length = Geod(ellps='WGS84').geometry_length(line)

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.01
    return length

def update_nexus_list(nexus_list: list, gdf: gpd.GeoDataFrame) -> None:
    for river_id in gdf[gdf['LengthGeodesicMeters'] <= 0.01]['LINKNO'].values:
        feat = gdf[gdf['LINKNO'] == river_id]
        upstreams = set(gdf[gdf['DSLINKNO'] == river_id]['LINKNO'])
        if not feat['DSLINKNO'].values != -1 and all([x != -1 for x in upstreams]):
            continue

        stream_row = gdf[gdf['LINKNO'] == river_id]
        point = Point(stream_row['geometry'].values[0].coords[0])
        downstream = gdf[gdf['LINKNO'] == stream_row['DSLINKNO'].values[0]]
        ds_strahler_order = downstream['strmOrder'].values[0]
        upstreams = ",".join(map(str, upstreams))
        
        nexus_list.append((river_id, point.y, point.x, downstream['LINKNO'].values[0], ds_strahler_order, upstreams, point))


if __name__ == '__main__':
    logging.info('Converting TDX-Hydro GPKG to Geoparquet')
    # add globally unique ID numbers
    with open(os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data', 'tdx_header_numbers.json')) as f:
        tdx_header_numbers = json.load(f)

    os.makedirs(gpq_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    nexus_file = os.path.join(save_dir, 'nexus_points.gpkg')
    nexus_list = []
    crs=None
    for gpkg in sorted(glob.glob(os.path.join(gpkg_dir, 'TDX*.gpkg'))):
        region_number = os.path.basename(gpkg).split('_')[-2]
        tdx_header_number = int(tdx_header_numbers[str(region_number)])
        logging.info(gpkg)

        out_file_name = os.path.join(gpq_dir, os.path.basename(gpkg).replace('.gpkg', '.parquet'))
        if os.path.exists(out_file_name):
            if not os.path.exists(nexus_file):
                if 'streamnet' in os.path.basename(gpkg):
                    gdf = gpd.read_file(out_file_name)
                    crs = gdf.crs
                    update_nexus_list(nexus_list, gdf)
            continue
        
        if 'streamnet' in os.path.basename(gpkg):
            gdf['LINKNO'] = gdf['LINKNO'].astype(int) + (tdx_header_number * 10_000_000)
            gdf['DSLINKNO'] = gdf['DSLINKNO'].astype(int)
            gdf.loc[gdf['DSLINKNO'] != -1, 'DSLINKNO'] = gdf['DSLINKNO'] + (tdx_header_number * 10_000_000)
            gdf['strmOrder'] = gdf['strmOrder'].astype(int)
            gdf['LengthGeodesicMeters'] = gdf['geometry'].apply(_calculate_geodesic_length)
            gdf['TDXHydroRegion'] = region_number

            gdf = gdf[[
                'LINKNO',
                'DSLINKNO',
                'strmOrder',
                'Magnitude',
                'USContArea',
                'DSContArea',
                'LengthGeodesicMeters',
                'TDXHydroRegion',
                'geometry'
            ]]

            crs = gdf.crs
            update_nexus_list(nexus_list, gdf)
        else:
            gdf['LINKNO'] = gdf['streamID'].astype(int) + (tdx_header_number * 10_000_000)
            gdf = gdf.drop(columns=['streamID'])

        gdf.to_parquet(out_file_name)

    if nexus_list:
        gpd.GeoDataFrame(nexus_list, columns=['LINKNO', 'Lat', 'Lon', 'DSLINKNO', 'DSStrahlerOrder', 'USLINKNOs', 'geometry'], crs=crs).to_file(nexus_file, index=False)
    else:
        logging.info('No Nexus Points Found')
