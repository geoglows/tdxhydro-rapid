import glob
import logging
import os
import sys

import geopandas as gpd
from natsort import natsorted
import pandas as pd
import tdxhydrorapid as rp

tdx_geoparquet_dir = '/Users/ricky/tdxhydro-postprocessing/test/pqs'
# final_output_dir = '/Volumes/T9Hales4TB/geoglows2/'
tdx_inputs_dir = "/Users/ricky/tdxhydro-postprocessing/test/rapid_inputs"
gpkg_dir = "/Users/ricky/tdxhydro-postprocessing/test/vpus/catchments"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.makedirs(gpkg_dir, exist_ok=True)

mdf = pd.read_parquet('/Users/ricky/tdxhydro-postprocessing/test/vpus/geoglows-v2-master-table.parquet')
for tdx in natsorted(glob.glob(os.path.join(tdx_inputs_dir, '*'))):
    tdxnumber = os.path.basename(tdx)
    logging.info(tdxnumber)
    catchments_gpq = os.path.join(tdx_geoparquet_dir, f'TDX_streamreach_basins_{tdxnumber}_01.parquet')
    gdf = gpd.read_parquet(catchments_gpq)
    gdf, lake_gdf = (
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
        if not lake_gdf.empty:
            (
                lake_gdf
                .loc[lake_gdf['LINKNO'].isin(vpu_numbers)]
                .to_file(os.path.join(gpkg_dir, f'lakes_{vpu}.gpkg'))
            )
