from herramientas.herramientas import rasterize_streams

dem = r"C:\Users\lrr43\mapas-de-inundacion\dems_recortados\recorte.tif"
streamlines_files = [r"C:\Users\lrr43\mapas-de-inundacion\cache\streams_718.geoparquet"]
output = 'rasterized_streams.tif'

rasterize_streams(dem, streamlines_files, output)