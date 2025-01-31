from herramientas.herramientas import read_config_yaml, get_base_max

configs = read_config_yaml('config.yml')
stream_raster = configs['output_streams']
base_max_file = configs['base_max_file']

get_base_max(stream_raster, base_max_file)
