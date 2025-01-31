from herramientas.herramientas import read_config_yaml, get_return_period

configs = read_config_yaml('config.yml')
stream_raster = configs['output_streams']
flow_file = configs['flow_file']
return_period = configs['return_period']

get_return_period(stream_raster, return_period, flow_file)
