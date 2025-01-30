import os
import glob
from herramientas.herramientas import crop_and_resize_land_cover, read_config_yaml

configs = read_config_yaml('config.yml')
output_dem = configs['output_dem']
land_use_dir = configs['land_use_dir']
input_land_use = glob.glob(os.path.join(land_use_dir, '*'))
output_land_use = configs["output_land_use"]


crop_and_resize_land_cover(output_dem, input_land_use, output_land_use, vrt=False)
