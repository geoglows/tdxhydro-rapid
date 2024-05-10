import glob
import logging
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry as sg

__all__ = [
    'sort_topologically',
    'create_directed_graphs',
    'find_headwater_branches_to_dissolve',
    'find_branches_to_prune',
    'identify_0_length',
    'correct_0_length_streams',
    'correct_0_length_basins',
    'make_final_streams',
    'dissolve_catchments',
]

logger = logging.getLogger(__name__)


def sort_topologically(digraph_from_headwaters: nx.DiGraph) -> np.array:
    return np.array(list(nx.topological_sort(digraph_from_headwaters))).astype(int)


def create_directed_graphs(df: pd.DataFrame,
                           id_field='LINKNO',
                           ds_id_field='DSLINKNO', ):
    G = nx.DiGraph()

    for node in df[id_field].values:
        G.add_node(node)
    for i, row in df.iterrows():
        if row[ds_id_field] != -1:
            G.add_edge(row[id_field], row[ds_id_field])
    return G


def find_headwater_branches_to_dissolve(sdf: pd.DataFrame or gpd.GeoDataFrame,
                                        G: nx.DiGraph,
                                        min_order_to_keep: int,
                                        stream_order_field: str = 'strmOrder',
                                        id_field: str = 'LINKNO', ) -> pd.DataFrame:
    # select rows where the number of predecessors in the graph is 2+ and 2+ of them and 2+ are min_order_to_keep - 1
    def _is_headwater_branch(row):
        predecessors = list(G.predecessors(row[id_field]))

        # if there are only 0 or 1 predecessors then it is not a headwater branch
        if len(predecessors) < 2:
            return False

        # get the stream order of all predecessors
        pred_orders = [sdf.loc[sdf[id_field] == x, stream_order_field].values[0] for x in predecessors]

        # check if there are 2+ predecessors with order min_order_to_keep - 1
        if len(predecessors) == 2 and not len([x for x in pred_orders if x == (min_order_to_keep - 1)]) == 2:
            return False

        if (len(predecessors) > 2) and \
                (not max(pred_orders) == min_order_to_keep - 1) and \
                (len([x for x in pred_orders if x == (min_order_to_keep - 1)]) >= 2):
            return False
        return True

    # todo parameterize the column names
    # order1 = sdf[sdf['strmOrder'] == (min_order_to_keep - 1)]['LINKNO'].values.flatten()
    candidate_streams = sdf[sdf[stream_order_field] == min_order_to_keep]

    candidate_streams = candidate_streams[candidate_streams.apply(_is_headwater_branch, axis=1)]

    # order2 = order2[order2[us_cols].isin(order1).sum(axis=1) >= 2]
    candidate_streams = candidate_streams[['LINKNO', ]]

    # get a dictionary of all streams upstream of the new headwater streams
    upstream_df = {x: list(nx.ancestors(G, x)) for x in candidate_streams[id_field].values.flatten()}
    upstream_df = pd.DataFrame.from_dict(upstream_df, orient='index').fillna(-1).astype(int)
    upstream_df.index.name = id_field
    upstream_df.columns = [f'USLINKNO{i}' for i in range(1, len(upstream_df.columns) + 1)]
    upstream_df = upstream_df.reset_index()
    return upstream_df


def find_branches_to_prune(sdf: gpd.GeoDataFrame or pd.DataFrame,
                           G: nx.DiGraph,
                           id_field: str = 'LINKNO',
                           ds_id_field: str = 'DSLINKNO', ) -> pd.DataFrame:
    # find all order 1 and 2+ branches
    order1s = sdf.loc[sdf['strmOrder'] == 1]
    # select the order 1s whose downstream is 2+
    order1s = order1s.loc[order1s[ds_id_field].isin(sdf.loc[sdf['strmOrder'] >= 2, id_field].values)]

    sibling_pairs = pd.DataFrame(columns=['LINKNO', 'LINKTODROP'])
    for index, row in order1s.iterrows():
        siblings = list(G.predecessors(row[ds_id_field]))
        siblings = [s for s in siblings if s != row[id_field]]

        # if there is only 1 sibling, we want to merge with that one
        # if there are 2+ siblings, we want need to figure out which one to merge with
        if len(siblings) > 1:
            sibling_stream_orders = sdf[sdf[id_field].isin(siblings)]['strmOrder'].values

            # Case: 1 of the siblings has same order, one has higher order -> pick the highest stream order
            if row['strmOrder'] in sibling_stream_orders and row['strmOrder'] < max(sibling_stream_orders):
                siblings = [
                    s for s in siblings if sdf.loc[sdf[id_field] == s, 'strmOrder'].values[0] > row['strmOrder']
                ]

            # otherwise, there are 2 higher order streams, and we should pick the nearest sibling
            else:
                try:
                    siblings = sdf[sdf[id_field].isin(siblings)]
                    siblings['dist'] = (
                        gpd
                        .GeoSeries(siblings.geometry)
                        .distance(row.geometry.centroid)
                    )
                    siblings = (
                        siblings
                        .sort_values('dist')
                        .iloc[0]
                        .loc[id_field]
                    )
                    siblings = [siblings, ]
                except Exception as e:
                    print(e)
                    print(siblings)
                    print(row)

        # In the case where there is a 3 river confluence, there may be more than 1 order 1 stream that must be merged.
        # Instead of creating a dictionary to store these values (which can only have one unique key), we use a DataFrame directly
        new_row = {'LINKNO': siblings[0], 'LINKTODROP': row[id_field]}
        sibling_pairs.loc[sibling_pairs.shape[0]] = new_row

    return sibling_pairs


def identify_0_length(gdf: gpd.GeoDataFrame,
                      stream_id_col: str,
                      ds_id_col: str,
                      length_col: str, ) -> pd.DataFrame:
    """
    Fix streams that have 0 length.
    General Error Cases:
    1) Feature is coastal w/ no upstream or downstream
        -> Delete the stream and its basin
    2) Feature is bridging a 3-river confluence (Has downstream and upstreams)
        -> Artificially create a basin with 0 area, and force a length on the point of 1 meter
    3) Feature is costal w/ upstreams but no downstream
        -> Force a length on the point of 1 meter
    4) Feature doesn't match any previous case
        -> Raise an error for now

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Stream network
    stream_id_col : string
        Field in stream network that corresponds to the unique id of each stream segment
    ds_id_col : string
        Field in stream network that corresponds to the unique downstream id of each stream segment
    length_col : string
        Field in basins network that corresponds to the unique length of each stream segment
    """
    case1_ids = []
    case2_ids = []
    case3_ids = []
    case4_ids = []

    for rivid in gdf[gdf[length_col] <= 0.01][stream_id_col].values:
        feat = gdf[gdf[stream_id_col] == rivid]

        upstreams = gdf[gdf[ds_id_col] == rivid][stream_id_col].values

        # Case 1
        # if feat[ds_id_col].values == -1 and feat['USLINKNO1'].values == -1 and feat['USLINKNO2'].values == -1:
        if feat[ds_id_col].values == -1 and all([x == -1 for x in upstreams]):
            case1_ids.append(rivid)

        # Case 2
        # elif feat[ds_id_col].values != -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
        elif feat[ds_id_col].values != -1 and all([x != -1 for x in upstreams]):
            case2_ids.append(rivid)

        # Case 3
        # elif feat[ds_id_col].values == -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
        elif feat[ds_id_col].values == -1 and all([x != -1 for x in upstreams]):
            case3_ids.append(rivid)

        # Case 4
        else:
            logging.warning(f"The stream segment {feat[stream_id_col]} has conditions we've not yet considered")
            case4_ids.append(rivid)

    # variable length lists with np.nan to make them the same length
    longest_list = max([len(case1_ids), len(case2_ids), len(case3_ids), len(case4_ids), ])
    case1_ids = case1_ids + [np.nan] * (longest_list - len(case1_ids))
    case2_ids = case2_ids + [np.nan] * (longest_list - len(case2_ids))
    case3_ids = case3_ids + [np.nan] * (longest_list - len(case3_ids))
    case4_ids = case4_ids + [np.nan] * (longest_list - len(case4_ids))

    return pd.DataFrame({
        'case1': case1_ids,
        'case2': case2_ids,
        'case3': case3_ids,
        'case4': case4_ids,
    })


def correct_0_length_streams(sgdf: gpd.GeoDataFrame,
                             zero_length_df: pd.DataFrame,
                             id_field: str, ) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        sgdf:
        zero_length_df:
        id_field:

    Returns:

    """
    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    c1 = zero_length_df['case1'].dropna().astype(int).values
    sgdf = sgdf[~sgdf[id_field].isin(c1)]

    # Case 3 - Coastal w/ upstreams but no downstream - Assign small non-zero length
    # Apply before case 2 to handle some edges cases where zero length basins drain into other zero length basins
    # c3_us_ids = sgdf[sgdf[id_field].isin(zero_length_df['case3'].dropna().values)][
    #     ['USLINKNO1', 'USLINKNO2']].values.flatten()
    c3_us_ids = sgdf[sgdf['DSLINKNO'].isin(zero_length_df['case3'].dropna().values)][id_field].unique().flatten()

    sgdf.loc[sgdf[id_field].isin(c3_us_ids), 'DSLINKNO'] = -1
    sgdf = sgdf[~sgdf['LINKNO'].isin(zero_length_df['case3'].dropna().values)]

    # Case 2 - Allow 3-river confluence - Delete the temporary basin and modify the connectivity properties
    # Sort by DSLINKNO to handle some edges cases where zero length basins drain into other zero length basins
    c2 = sgdf[sgdf['LINKNO'].isin(zero_length_df['case2'].dropna().astype(int).values)]
    c2 = c2.sort_values(by=['DSLINKNO'], ascending=True)
    c2 = c2['LINKNO'].values
    for river_id in c2:
        # ids_to_apply = sgdf.loc[sgdf[id_field] == river_id, ['USLINKNO1', 'USLINKNO2', 'DSLINKNO']]
        dslinkno = sgdf.loc[sgdf[id_field] == river_id, 'DSLINKNO'].values[0]
        upstreams = sgdf.loc[sgdf['DSLINKNO'] == river_id, 'LINKNO'].unique().flatten()
        # if the downstream basin is also a zero length basin, find the basin 1 step further downstream
        if dslinkno in c2:
            dslinkno = sgdf.loc[sgdf[id_field] == dslinkno, 'DSLINKNO'].values[0]
        sgdf.loc[sgdf[id_field].isin(upstreams), 'DSLINKNO'] = dslinkno

    # Remove the rows corresponding to the rivers to be deleted
    sgdf = sgdf[~sgdf['LINKNO'].isin(c2)]

    return sgdf


def correct_0_length_basins(basins_gpq: str,
                            save_dir: str,
                            stream_id_col: str, ) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        basins_gpq: Basins to correct
        save_dir: Directory to save the corrected basins to
        stream_id_col:

    Returns:

    """
    basin_gdf = gpd.read_parquet(basins_gpq)

    zero_fix_csv_path = os.path.join(save_dir, 'mod_basin_zero_centroid.csv')
    if os.path.exists(zero_fix_csv_path):
        box_radius_degrees = 0.015
        basin_zero_centroid = pd.read_csv(zero_fix_csv_path)
        centroid_x = basin_zero_centroid['centroid_x'].values[0]
        centroid_y = basin_zero_centroid['centroid_y'].values[0]
        link_zero_box = gpd.GeoDataFrame({
            'geometry': [sg.box(
                centroid_x - box_radius_degrees,
                centroid_y - box_radius_degrees,
                centroid_x + box_radius_degrees,
                centroid_y + box_radius_degrees
            )],
            stream_id_col: [0, ]
        }, crs=basin_gdf.crs)
        basin_gdf = pd.concat([basin_gdf, link_zero_box])

    zero_length_csv_path = os.path.join(save_dir, 'mod_zero_length_streams.csv')
    if os.path.exists(zero_length_csv_path):
        logger.info('\tRevising basins with 0 length streams')
        zero_length_df = pd.read_csv(zero_length_csv_path)
        # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
        logger.info('\tHandling Case 1 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case1'])]
        # Case 2 - Allow 3-river confluence - basin does not exist (try to delete just in case)
        logger.info('\tHandling Case 2 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case2'])]
        # Case 3 - Coastal w/ upstreams but no downstream - basin exists so delete it
        logger.info('\tHandling Case 3 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case3'])]

    small_tree_csv_path = os.path.join(save_dir, 'mod_drop_small_trees.csv')
    if os.path.exists(small_tree_csv_path):
        logger.info('\tDeleting small trees')
        small_tree_df = pd.read_csv(small_tree_csv_path)
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(small_tree_df.values.flatten())]

    basin_gdf = basin_gdf.reset_index(drop=True)
    return basin_gdf


def make_final_streams(final_inputs_directory: str,
                       final_gis_directory: str,
                       tdxrapid_dir: str, ) -> None:
    print('reading modified geoparquets')
    modified_gpqs = sorted(glob.glob(os.path.join(tdxrapid_dir, '*', '*.geoparquet')))
    mgdf = pd.concat([gpd.read_parquet(gpq) for gpq in modified_gpqs])

    print('merging with master table')
    mgdf = mgdf.merge(pd.read_parquet(os.path.join(final_inputs_directory, 'geoglows-v2-master-table.parquet')),
                      on='LINKNO', how='inner')

    for vpu_code in sorted(mgdf['VPUCode'].unique()):
        print(vpu_code)
        file_path = os.path.join(final_gis_directory, f'vpu_{vpu_code}_streams.gpkg')
        if os.path.exists(file_path):
            continue
        mgdf[mgdf['VPUCode'] == vpu_code].to_file(file_path)
    return


def dissolve_catchments(save_dir: str, gdf: gpd.GeoDataFrame, id_field: str = 'LINKNO'):
    headwater_dissolve_path = os.path.join(save_dir, 'mod_dissolve_headwater.csv')
    if os.path.exists(headwater_dissolve_path):
        o2_to_dissolve = pd.read_csv(headwater_dissolve_path).fillna(-1).astype(int)
        for streams_to_merge in o2_to_dissolve.values:
            gdf.loc[gdf[id_field].isin(streams_to_merge), id_field] = streams_to_merge[0]

    streams_to_prune_path = os.path.join(save_dir, 'mod_prune_streams.csv')
    if os.path.exists(streams_to_prune_path):
        ids_to_prune = pd.read_csv(streams_to_prune_path).astype(int).set_index('LINKTODROP')
        gdf[id_field] = gdf[id_field].replace(ids_to_prune['LINKNO'])

    drop_streams_path = os.path.join(save_dir, 'mod_drop_small_trees.csv')
    if os.path.exists(drop_streams_path):
        ids_to_drop = pd.read_csv(drop_streams_path).astype(int)
        gdf = gdf[~gdf[id_field].isin(ids_to_drop.values.flatten())]

    short_streams_path = os.path.join(save_dir, 'mod_merge_short_streams.csv')
    if os.path.exists(short_streams_path):
        short_streams = pd.read_csv(short_streams_path).astype(int)
        for streams_to_merge in short_streams.values:
            gdf.loc[gdf[id_field].isin(streams_to_merge), id_field] = streams_to_merge[0]

    # dissolve the geometries based on shared value in the id field
    gdf = gdf.dissolve(by=id_field).reset_index()
    return gdf
