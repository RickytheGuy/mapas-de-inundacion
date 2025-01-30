from herramientas.herramientas import clean_stream_raster, read_config_yaml

configs = read_config_yaml('config.yml')
stream_raster = configs['output_streams']

clean_stream_raster(stream_raster)

