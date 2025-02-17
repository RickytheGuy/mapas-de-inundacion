import herramientas.herramientas as h
import os
import glob

from arc import Arc

configs = h.read_config_yaml('config.yml')
minx = configs['minx']
miny = configs['miny']
maxx = configs['maxx']
maxy = configs['maxy']
dem_dir = configs['dem_dir']
output_dem = configs['output_dem']
streamlines_dir = configs['streamlines_dir']
output_streamlines = configs['output_streamlines']
stream_raster = configs['output_streams']
land_use_dir = configs['land_use_dir']
base_max_file = configs['base_max_file']
output_land_use = configs["output_land_use"]
main_input_file = configs['main_input_file']

h.download_fabdem(minx, miny, maxx, maxy, dem_dir)
dems_in_extent = h.get_fabdem_in_extent(dem_dir, minx, miny, maxx, maxy)
h.crop_and_merge(minx, miny, maxx, maxy, dems_in_extent, output_dem)
h.download_geoglows_streams(minx, miny, maxx, maxy, streamlines_dir, output_streamlines)
h.rasterize_streams(output_dem, output_streamlines, stream_raster)
h.clean_stream_raster(stream_raster)
h.download_land_use(minx, miny, maxx, maxy, land_use_dir)

input_land_use = glob.glob(os.path.join(land_use_dir, '*'))

h.crop_and_resize_land_cover(output_dem, input_land_use, output_land_use, vrt=False)
h.get_base_max(stream_raster, base_max_file)
h.create_main_input_file(main_input_file, configs)

# Run the main input file
Arc(main_input_file).run()
