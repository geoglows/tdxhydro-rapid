import dask_geopandas as dgpd

gdf = dgpd.read_parquet('/Users/ricky/tdxhydro-postprocessing/test/rapid_inputs/*/*geoparquet', filesystem='arrow', columns=['LINKNO', 'geometry'])
gdf['geometry'] = gdf['geometry'].simplify(0.001, preserve_topology=False)
gdf.compute().to_file('test_global_streams_simplified_dask.gpkg', driver='GPKG')
