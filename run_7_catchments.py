import glob
import logging
import os
import sys

import geopandas as gpd
from natsort import natsorted
import pandas as pd
import tdxhydrorapid as rp

tdx_geoparquet_dir = '/Volumes/T9Hales4TB/TDXHydroGeoParquet'
final_output_dir = '/Volumes/T9Hales4TB/geoglows2/'
tdx_inputs_dir = os.path.join(final_output_dir, 'tdxhydro-inputs')
gpkg_dir = os.path.join(final_output_dir, 'catchments')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.makedirs(gpkg_dir, exist_ok=True)

mdf = pd.read_parquet('/Volumes/T9Hales4TB/geoglows2/geoglows-v2-master-table.parquet')

for tdx in natsorted(glob.glob(os.path.join(tdx_inputs_dir, '*'))):
    print(tdx)
    tdxnumber = os.path.basename(tdx)
    catchments_gpq = os.path.join(tdx_geoparquet_dir, f'TDX_streamreach_basins_{tdxnumber}_01.parquet')
    gdf = gpd.read_parquet(catchments_gpq)
    gdf = (
        rp
        .network
        .dissolve_catchments(tdx, gdf)
    )
    for vpu in natsorted(mdf.loc[mdf['TDXHydroRegion'] == tdxnumber, 'VPUCode'].unique()):
        vpu_numbers = mdf.loc[mdf['VPUCode'] == vpu, 'LINKNO'].unique()
        (
            gdf
            .loc[gdf['LINKNO'].isin(vpu_numbers)]
            .to_file(os.path.join(gpkg_dir, f'catchments_{vpu}.spatialite'), driver='SQLite', spatialite=True)
        )
