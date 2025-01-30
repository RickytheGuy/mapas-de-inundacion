from herramientas.herramientas import crop_and_merge, get_fabdem_in_extent, get_dems_in_extent, read_config_yaml

configs = read_config_yaml('config.yml')

minx = configs['minx']
miny = configs['miny']
maxx = configs['maxx']
maxy = configs['maxy']

dem_dir = configs['dem_dir']
output_dem = configs['output_dem']

dems_in_extent = get_fabdem_in_extent(dem_dir, minx, miny, maxx, maxy)
# Below is for non-FABDEM DEMs
# dems_in_extent = get_dems_in_extent(dem_dir, minx, miny, maxx, maxy)

crop_and_merge(minx, miny, maxx, maxy, dems_in_extent, output_dem)
# Below is to write as a VRT; it saves a lot of disk space
# crop_and_merge(minx, miny, maxx, maxy, dems_in_extent, output_dem, True)


