import logging
import sys
import glob
import os
import networkx as nx
import tqdm

import geopandas as gpd
import pandas as pd
import dask_geopandas as dgpd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

lakes_parquet = '/Users/ricky/Downloads/GLAKES/all_lakes_filtered.parquet'
gpq_dir = '/Volumes/EB406_T7_3/geoglows_v3/parquets'
save_dir = './tdxhydrorapid/network_data/'

def create_directed_graphs(df: gpd.GeoDataFrame,
                           id_field='LINKNO',
                           ds_id_field='DSLINKNO', ) -> nx.DiGraph:
    G: nx.DiGraph = nx.from_pandas_edgelist(df[df[ds_id_field] != -1], source=id_field, target=ds_id_field, create_using=nx.DiGraph)
    G.add_nodes_from(df[id_field].values)
    return G

if __name__ == "__main__":
    logging.info('Getting Lake CSV')
    lakes_gdf = gpd.read_parquet(lakes_parquet)
    lakes_gdf = lakes_gdf[lakes_gdf['Area_PW'] > 0.4]
    values_list = []
    pqs = glob.glob(os.path.join(gpq_dir, 'TDX_streamnet*.parquet'))
    #pqs = ['/Volumes/EB406_T7_3/geoglows_v3/parquets/TDX_streamnet_7020000010_01.parquet']
    for pq in tqdm.tqdm(pqs):
        gdf = gpd.read_parquet(pq)
        G = create_directed_graphs(gdf)
        strm_order_dict: dict[int, int] = gdf.set_index('LINKNO')['strmOrder'].to_dict()
        
        # Get all streams that intersect with a lake
        bounds = gdf.total_bounds
        lakes_subset = lakes_gdf.cx[ bounds[0]:bounds[2], bounds[1]:bounds[3]]
        dgdf = dgpd.from_geopandas(gdf, npartitions=os.cpu_count()*2)
        intersect = dgpd.sjoin(dgdf, dgpd.from_geopandas(lakes_subset), how='inner', predicate='intersects').compute()

        if intersect.empty:
            continue

        intersect = intersect.drop(columns=['index_right'])
        geom_dict: dict[int, int] = gdf[gdf['LINKNO'].isin(intersect['LINKNO'])].set_index('LINKNO')['geometry'].to_dict()
        lakes_g = create_directed_graphs(intersect)
        intersect = intersect.set_index('LINKNO')

        # Get connected components
        for lake_ids in nx.weakly_connected_components(lakes_g):
            # Get sink
            outlets = [node for node in lake_ids if lakes_g.out_degree(node) == 0]
            if len(outlets) != 1:
                raise RuntimeError(f'Lake has {len(outlets)} outlets')
            
            outlet = outlets[0]
            # Move the outlet, as needed
            predecessors = set(G.predecessors(outlet))
            if len(predecessors) == 2 and all([pred in lake_ids for pred in predecessors]):
                # Outlet can stay in this situation
                pass
            else:
                # Set the upstream predecessor that touches the lake as the outlet
                for pred in predecessors:
                    if pred in lake_ids:
                        outlet = pred
                        break        

            # Get all inlets
            inlets = {node for node in lake_ids if lakes_g.in_degree(node) == 0}
            if outlet in inlets or len(lake_ids - inlets - {outlet}) == 0:
                # Small lake, skip
                continue

            # Add any direct predecessors of inlets that have strmOrder == 1
            new_inlets = set()
            lake_id_dict: dict[int, int] = intersect.loc[intersect.index.isin(lake_ids), 'Lake_id'].to_dict()
            lake_polygon_dict = lakes_subset[lakes_subset['Lake_id'].isin(intersect.loc[intersect.index.isin(lake_ids), 'Lake_id'].unique())].set_index('Lake_id')['geometry'].to_dict()
            for inlet in inlets:
                preds = set(G.predecessors(inlet))
                if not preds:
                    # Inlet is a headwater touching lake
                    pass
                else:
                    intersection = geom_dict[inlet].intersection(lake_polygon_dict[lake_id_dict[inlet]])
                    intersection_length = intersection.length if not intersection.is_empty else 0

                    # Calculate the length outside the polygon
                    outside = geom_dict[inlet].difference(lake_polygon_dict[lake_id_dict[inlet]])
                    outside_length = outside.length if not outside.is_empty else 0
                    if intersection_length > outside_length:
                        # This stream is mostly inside the lake, so lets add preds
                        new_inlets.update(preds)
                        for pred in preds:
                            lake_id_dict[pred] = lake_id_dict[inlet]
                    else:
                        # This stream is mostly outside the lake, so lets maintain it as an inlet
                        new_inlets.add(inlet)

            if len(lake_ids - new_inlets - {outlet}) == 0:
                # Not worth making a lake
                continue

            
            for inlet in new_inlets:
                values_list.append((inlet, outlet, lake_id_dict[inlet]))

    df = pd.DataFrame(values_list, columns=['inlet', 'outlet', 'lake_id'])
    # gdf['type'] = ""
    # gdf.loc[gdf['LINKNO'].isin(df['inlet']), 'type'] = 'inlet'
    # gdf.loc[gdf['LINKNO'].isin(df['outlet']), 'type'] = 'outlet'
    # gdf[gdf['type'] != ''].to_file('inlets_outlets.gpkg')
    out_name = os.path.join(save_dir, 'lakes.csv')
    df.to_csv(out_name, index=False)
