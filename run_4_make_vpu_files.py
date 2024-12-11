import logging
import os
import shutil
import sys
import glob
import traceback

import pandas as pd
import geopandas as gpd

import tdxhydrorapid as rp

tdx_inputs_dir = '/Users/ricky/tdxhydro-postprocessing/test/rapid_inputs'
og_pqs = glob.glob("/Users/ricky/tdxhydro-postprocessing/test/pqs/TDX_streamnet_*_01.parquet")
final_output_dir = '/Users/ricky/tdxhydro-postprocessing/test/vpus'
vpu_inputs_dir = os.path.join(final_output_dir, 'inputs')
gpkg_dir = os.path.join(final_output_dir, 'streams')
vpu_assignment_table = './tdxhydrorapid/network_data/vpu_table.csv'
vpu_boundaries = '/Users/ricky/Downloads/vpu-boundaries.gpkg' or None

MAKE_GPKG = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.makedirs(vpu_inputs_dir, exist_ok=True)
os.makedirs(gpkg_dir, exist_ok=True)

logging.info('Creating Model Master Table')
master_table_path = os.path.join(os.path.dirname(vpu_inputs_dir), 'geoglows-v2-master-table.parquet')
if not os.path.exists(master_table_path):
    rp.inputs.concat_tdxregions(tdx_inputs_dir, vpu_assignment_table, master_table_path, og_pqs)
mdf = pd.read_parquet(master_table_path)
logging.info(f'Total streams: {len(mdf)}')

vpu_bounds_gdf = None
if vpu_boundaries and os.path.exists(vpu_boundaries):
    vpu_bounds_gdf = gpd.read_file(vpu_boundaries)

for vpu in sorted(mdf['VPUCode'].unique()):
    logging.info(vpu)
    vpu_df = mdf.loc[mdf['VPUCode'] == vpu]
    tdx_region = str(vpu_df['TDXHydroRegion'].values[0])

    vpu_dir = os.path.join(vpu_inputs_dir, str(vpu))
    if os.path.exists(vpu_dir) and (not MAKE_GPKG or os.path.exists(os.path.join(gpkg_dir, f'streams_{vpu}.gpkg'))):
        if rp.check_outputs_are_valid(vpu_dir):
            continue
        else:
            shutil.rmtree(vpu_dir)

    os.makedirs(vpu_dir, exist_ok=True)
    try:
        rp.inputs.vpu_files_from_masters(vpu_df,
                                         vpu_dir,
                                         tdxinputs_directory=tdx_inputs_dir,
                                         make_gpkg=MAKE_GPKG,
                                         gpkg_dir=gpkg_dir, 
                                         vpu_boundaries=vpu_bounds_gdf)
    except Exception as e:
        logging.error(vpu)
        logging.error(tdx_region)
        logging.error(traceback.format_exc())
        shutil.rmtree(vpu_dir)
        continue

    if not rp.check_outputs_are_valid(vpu_dir):
        shutil.rmtree(vpu_dir)
        continue
