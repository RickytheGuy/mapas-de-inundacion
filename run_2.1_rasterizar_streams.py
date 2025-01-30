import os
import glob

from herramientas.herramientas import rasterize_streams, read_config_yaml

configs = read_config_yaml('config.yml')

output_dem = configs['output_dem']
streamlines_files = configs['output_streamlines']
output_streams = configs['output_streams']

rasterize_streams(output_dem, streamlines_files, output_streams)
