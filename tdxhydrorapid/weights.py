import logging
import os
import warnings
import json
from typing import Generator, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import xarray as xr
import dask_geopandas as dgpd

logger = logging.getLogger(__name__)

__all__ = [
    'make_thiessen_grid_from_netcdf_sample',
    'make_weight_table_from_thiessen_grid',
    'make_weight_table_from_netcdf',
    'apply_weight_table_simplifications',
    'get_expected_weight_tables'
]


def make_thiessen_grid_from_netcdf_sample(lsm_sample: str, out_dir: str, ) -> None:
    new_file_name = os.path.basename(lsm_sample).replace('.nc', '_thiessen_grid.parquet')
    new_file_name = os.path.join(out_dir, new_file_name)
    if os.path.exists(new_file_name):
        logger.info(f'Thiessen grid already exists: {os.path.basename(new_file_name)}')
        return

    # Extract xs and ys dimensions from the ds
    lsm_ds = xr.open_dataset(lsm_sample)
    x_var = [v for v in lsm_ds.variables if v in ('lon', 'longitude',)][0]
    y_var = [v for v in lsm_ds.variables if v in ('lat', 'latitude',)][0]
    xs = lsm_ds[x_var].values
    ys = lsm_ds[y_var].values
    lsm_ds.close()

    # correct irregular x coordinates
    xs[xs > 180] = xs[xs > 180] - 360

    all_xs = xs.copy()
    all_ys = ys.copy()

    # create thiessen polygons around the 2d array centers and convert to a geodataframe
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    # Create Thiessen polygon based on the point feature
    # the order of polygons in the voronoi diagram is **guaranteed not** the same as the order of the input points
    logging.info('\tCreating Thiessen polygons')
    regions = shapely.ops.voronoi_diagram(
        shapely.geometry.MultiPoint(
            [shapely.geometry.Point(x, y) for x, y in zip(x_grid, y_grid)]
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.info('\tadding metadata to voronoi polygons gdf')
        # create a geodataframe from the voronoi polygons
        tg_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs=4326)
        tg_gdf['lon'] = tg_gdf.geometry.apply(lambda x: x.centroid.x).astype(float)
        tg_gdf['lat'] = tg_gdf.geometry.apply(lambda y: y.centroid.y).astype(float)
        tg_gdf['lon_index'] = tg_gdf['lon'].apply(lambda x: np.argmin(np.abs(all_xs - x)))
        tg_gdf['lat_index'] = tg_gdf['lat'].apply(lambda y: np.argmin(np.abs(all_ys - y)))

    # Remove invalid geometries
    tg_gdf = tg_gdf[tg_gdf['lon'].between(-180, 180) & tg_gdf['lat'].between(-90, 90)]

    # save the thiessen grid to disc
    logging.info('\tSaving Thiessen grid to disc')
    tg_gdf.to_parquet(new_file_name)
    return

def overlay(left: dgpd.GeoDataFrame, right: dgpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Performs a spatial intersection overlay of two GeoDataFrames using Dask.
    Reproject geometries to a Cylindrical Equal Area projection.
    """
    return gpd.GeoDataFrame(
        left.sjoin(right.assign(right_geometry=right.geometry))
        .assign(geometry=lambda x: x.geometry.intersection(x.right_geometry).to_crs({'proj':'cea'}))
        .drop(columns="right_geometry")
        .compute()).sort_index() # Sort index is needed (used when calcualting area_sqm)

def make_weight_table_from_thiessen_grid(tg_parquet: str,
                                         out_dir: str,
                                         basins_gdf: gpd.GeoDataFrame,
                                         id_field: str = 'LINKNO') -> None:
    # load the thiessen grid
    logger.info('\tloading thiessen grid')
    tg_gdf = gpd.read_parquet(tg_parquet)

    weight_file_name = get_weight_table_name_from_grid(tg_gdf).replace('.csv', '_full.csv')
    out_name = os.path.join(out_dir, weight_file_name)
    if os.path.exists(os.path.join(out_dir, out_name)):
        logger.info(f'Weight table already exists: {os.path.basename(out_name)}')
        return
    logger.info(f'Creating weight table: {os.path.basename(out_name)}')

    # filter the thiessen grid to only include points within the basins bounding box
    logger.info('\tfiltering thiessen grid by bounding box')
    basins_bbox = basins_gdf.total_bounds
    tg_gdf = tg_gdf.cx[basins_bbox[0]:basins_bbox[2], basins_bbox[1]:basins_bbox[3]]

    logger.info('\tcalculating intersections and areas')
    tg_ddf: dgpd.GeoDataFrame = dgpd.from_geopandas(tg_gdf, npartitions=1)
    basins_ddf: dgpd.GeoDataFrame = dgpd.from_geopandas(basins_gdf, npartitions=80)
    intersections = overlay(basins_ddf, tg_ddf)
    intersections['area_sqm'] = dgpd.from_geopandas(intersections, npartitions=80).area.compute()

    intersections.loc[intersections[id_field].isna(), id_field] = 0

    logger.info('\twriting weight table csv')
    (
        intersections[[id_field, 'area_sqm', 'lon_index', 'lat_index', 'lon', 'lat']]
        .sort_values([id_field, 'area_sqm'])
        .to_csv(out_name, index=False)
    )


def make_weight_table_from_netcdf(lsm_sample: str,
                                  out_dir: str,
                                  basins_gdf: gpd.GeoDataFrame,
                                  id_field: str = 'LINKNO') -> None:
    out_name = os.path.join(out_dir, 'weight_' + os.path.basename(os.path.splitext(lsm_sample)[0]) + '_full.csv')
    if os.path.exists(os.path.join(out_dir, out_name)):
        logger.info(f'Weight table already exists: {os.path.basename(out_name)}')
        return
    logger.info(f'Creating weight table: {os.path.basename(out_name)}')

    # Extract xs and ys dimensions from the ds
    lsm_ds = xr.open_dataset(lsm_sample)
    x_var = [v for v in lsm_ds.variables if v in ('lon', 'longitude',)][0]
    y_var = [v for v in lsm_ds.variables if v in ('lat', 'latitude',)][0]
    xs = lsm_ds[x_var].values
    ys = lsm_ds[y_var].values
    lsm_ds.close()

    # get the resolution of the ds
    resolution = np.abs(xs[1] - xs[0])

    # correct irregular x coordinates
    xs[xs > 180] = xs[xs > 180] - 360

    all_xs = xs.copy()
    all_ys = ys.copy()

    # buffer the min/max in case any basins are close to the edges
    x_min, y_min, x_max, y_max = basins_gdf.total_bounds
    x_min = x_min - resolution
    x_max = x_max + resolution
    y_min = y_min - resolution
    y_max = y_max + resolution

    # find the indexes of the bounding box
    x_min_idx = np.argmin(np.abs(xs - x_min))
    x_max_idx = np.argmin(np.abs(xs - x_max))
    y_min_idx = np.argmin(np.abs(ys - y_min))
    y_max_idx = np.argmin(np.abs(ys - y_max))

    if x_min_idx > x_max_idx:
        xs = np.concatenate((xs[x_min_idx:], xs[:x_max_idx + 1]))
    else:
        xs = xs[x_min_idx:x_max_idx + 1]
    y_min_idx, y_max_idx = min(y_min_idx, y_max_idx), max(y_min_idx, y_max_idx)
    ys = ys[y_min_idx:y_max_idx + 1]

    # create thiessen polygons around the 2d array centers and convert to a geodataframe
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    # Create Thiessen polygon based on the point feature
    # the order of polygons in the voronoi diagram is **guaranteed not** the same as the order of the input points
    logging.info('\tCreating Thiessen polygons')
    regions = shapely.ops.voronoi_diagram(
        shapely.geometry.MultiPoint(
            [shapely.geometry.Point(x, y) for x, y in zip(x_grid, y_grid)]
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.info('\tadding metadata to voronoi polygons gdf')
        # create a geodataframe from the voronoi polygons
        tg_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs=4326)
        tg_gdf['lon'] = tg_gdf.geometry.apply(lambda x: x.centroid.x).astype(float)
        tg_gdf['lat'] = tg_gdf.geometry.apply(lambda y: y.centroid.y).astype(float)
        tg_gdf['lon_index'] = tg_gdf['lon'].apply(lambda x: np.argmin(np.abs(all_xs - x)))
        tg_gdf['lat_index'] = tg_gdf['lat'].apply(lambda y: np.argmin(np.abs(all_ys - y)))

    tg_ddf: dgpd.GeoDataFrame = dgpd.from_geopandas(tg_gdf, npartitions=1)
    basins_ddf: dgpd.GeoDataFrame = dgpd.from_geopandas(basins_gdf, npartitions=80)
    intersections = overlay(basins_ddf, tg_ddf)
    intersections['area_sqm'] = dgpd.from_geopandas(intersections, npartitions=80).area.compute()

    intersections.loc[intersections[id_field].isna(), id_field] = 0

    logger.info('\tcalculating number of points')
    intersections['npoints'] = intersections.groupby(id_field)[id_field].transform('count')

    logger.info('\twriting weight table csv')
    (
        intersections[[id_field, 'area_sqm', 'lon_index', 'lat_index', 'npoints', 'lon', 'lat']]
        .sort_values([id_field, 'area_sqm'])
        .to_csv(out_name, index=False)
    )
    return

def apply_weight_table_simplifications(save_dir: str,
                                       weight_table_in_path: str,
                                       weight_table_out_path: str,
                                       id_field: str = 'LINKNO') -> None:
    logging.info(f'Processing {weight_table_in_path}')
    wt = pd.read_csv(weight_table_in_path)

    headwater_dissolve_path = os.path.join(save_dir, 'mod_dissolve_headwater.csv')
    if os.path.exists(headwater_dissolve_path):
        o2_to_dissolve = pd.read_csv(headwater_dissolve_path).fillna(-1).astype(int)
        merge_dict = {stream: streams_to_merge[0] for streams_to_merge in o2_to_dissolve.values for stream in streams_to_merge}
        wt[id_field] = wt[id_field].map(merge_dict).fillna(wt[id_field]).astype(int)

    streams_to_prune_path = os.path.join(save_dir, 'mod_prune_streams.csv')
    if os.path.exists(streams_to_prune_path):
        ids_to_prune = pd.read_csv(streams_to_prune_path).astype(int).set_index('LINKTODROP')['LINKNO'].to_dict()
        wt[id_field] = wt[id_field].map(ids_to_prune).fillna(wt[id_field]).astype(int)

    drop_streams_path = os.path.join(save_dir, 'mod_drop_small_trees.csv')
    if os.path.exists(drop_streams_path):
        ids_to_drop = pd.read_csv(drop_streams_path).astype(int)
        wt = wt[~wt[id_field].isin(ids_to_drop.values.flatten())]

    wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()

    lake_streams_path = os.path.join(save_dir, 'mod_dissolve_lakes.json')
    if os.path.exists(lake_streams_path):
        lake_streams_df = pd.read_json(lake_streams_path, orient='index', convert_axes=False, convert_dates=False)
        streams_to_delete = set()
        for outlet, _, lake_streams in lake_streams_df.itertuples():
            outlet = int(outlet)
            if outlet in wt[id_field].values:
                wt.loc[wt[id_field].isin(lake_streams), id_field] = outlet
            else:
                streams_to_delete.update(lake_streams)

        wt = wt[~wt[id_field].isin(streams_to_delete)]
        wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()

    low_flow_streams_path = os.path.join(save_dir, 'mod_drop_low_flow.json')
    if os.path.exists(low_flow_streams_path):
        with open(low_flow_streams_path, 'r') as f:
            low_flow_streams_dict: dict = json.load(f)

        streams_to_delete = set()
        for outlet, low_flow_streams in low_flow_streams_dict.items():
            outlet = int(outlet)
            if outlet in wt[id_field].values:
                wt.loc[wt[id_field].isin(low_flow_streams), id_field] = outlet
            else:
                streams_to_delete.update(low_flow_streams)              

        wt = wt[~wt[id_field].isin(streams_to_delete)]
        wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()
    
    island_streams_path = os.path.join(save_dir, 'mod_dissolve_islands.json')
    if os.path.exists(island_streams_path):
        with open(island_streams_path, 'r') as f:
            island_streams_dict: dict = json.load(f)

        streams_to_delete = set()
        for outlet, island_streams in island_streams_dict.items():
            outlet = int(outlet)
            if outlet == -1 or outlet not in wt[id_field].values:
                streams_to_delete.update(island_streams)
            else:
                wt.loc[wt[id_field].isin(island_streams), id_field] = outlet

        wt = wt[~wt[id_field].isin(streams_to_delete)]

    # merge small streams into larger streams
    merge_streams_path = os.path.join(save_dir, 'mod_merge_short_streams.csv')
    if os.path.exists(merge_streams_path):
        merge_streams = pd.read_csv(merge_streams_path).astype(int)
        for merge_stream in merge_streams.values:
            # check that both ID's exist before editing - some may have been fixed in previous iterations
            if merge_stream[0] not in wt[id_field].values or merge_stream[1] not in wt[id_field].values:
                continue
            wt[id_field] = wt[id_field].replace(merge_stream[1], merge_stream[0])

    # group by matching values in columns except for area_sqm, and sum the areas in grouped rows
    wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()
    wt = wt.sort_values([id_field, 'area_sqm'], ascending=[True, False])
    wt.to_csv(weight_table_out_path, index=False)
    return

def get_weight_table_name_from_grid(grid: Union[str, gpd.GeoDataFrame], warn=False) -> str:
    if isinstance(grid, str):
        gpq = gpd.read_parquet(grid)
    else:
        gpq = grid
    dxs = gpq['lon'].round(12).drop_duplicates().sort_values().diff().dropna().unique()
    if len(dxs) > 1:
        if warn: logger.warning(f'Multiple dx values found in {grid}, using median')
        dx = round(np.median(dys), 12)
    else:
        dx = round(dxs[0], 12)

    dys = gpq['lat'].round(12).drop_duplicates().sort_values().diff().dropna().unique()
    if len(dys) > 1:
        if warn: logger.warning(f'Multiple dy values found in {grid}, using median')
        dy = round(np.median(dys), 12)
    else:
        dy = round(dys[0], 12)
    
    xinit = gpq['lon'].round(12).min()
    yinit = gpq['lat'].round(12).min()
    
    return f"weight_xinit={xinit}_yinit={yinit}_dx={dx}_dy={dy}_.csv"

def get_expected_weight_tables(sample_grids: list[str], warn=False) -> Generator[str, None, None]:
    for grid in sample_grids:
        yield get_weight_table_name_from_grid(grid, warn)

