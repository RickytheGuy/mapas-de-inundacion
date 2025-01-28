import os

from herramientas.herramientas import crop_and_merge, get_fabdem_in_extent, get_dems_in_extent

minx = -66.5
miny = 18.2
maxx = -66.3
maxy = 18.4

dem_dir = 'dems'
output_dir = 'dems_recortados'
os.makedirs(output_dir, exist_ok=True)

dems_in_extent = get_fabdem_in_extent(dem_dir, minx, miny, maxx, maxy)
# Below is for non-FABDEM DEMs
# dems_in_extent = get_dems_in_extent(dem_dir, minx, miny, maxx, maxy)

output_file = os.path.join(output_dir, "recorte.tif")

crop_and_merge(minx, miny, maxx, maxy, dems_in_extent, output_file)
# Below is to write as a VRT; it saves a lot of disk space
# crop_and_merge(minx, miny, maxx, maxy, dems_in_extent, output_file, True)


