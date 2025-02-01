from herramientas.herramientas import read_config_yaml, get_return_period, get_historical

configs = read_config_yaml('config.yml')
stream_raster = configs['output_streams']
flow_file = configs['flow_file']

if configs['flow_value_source'] == 'return_period':
    return_period = configs['return_period']
    get_return_period(stream_raster, return_period, flow_file)
elif configs['flow_value_source'] == 'historical':
    date_start = configs['date_start']
    date_end = configs['date_end']
    get_historical(stream_raster, flow_file, date_start, date_end)
else:
    print(f"Flow value source {configs['flow_value_source']} not recognized")
