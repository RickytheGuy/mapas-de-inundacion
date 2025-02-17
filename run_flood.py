import herramientas.herramientas as h

from curve2flood import Curve2Flood_MainFunction

configs = h.read_config_yaml('config.yml')
stream_raster = configs['output_streams']
flow_file = configs['flow_file']
return_period = configs['return_period']
date_start = configs['date_start']
date_end = configs['date_end']
main_input_file = configs['main_input_file']

if configs['flow_value_source'] == 'return_period':
    h.get_return_period(stream_raster, return_period, flow_file)
elif configs['flow_value_source'] == 'historical':
    h.get_historical(stream_raster, flow_file, date_start, date_end)
else:
    print(f"Flow value source {configs['flow_value_source']} not recognized")

h.create_main_input_file(main_input_file, configs)

# Run the main input file
Curve2Flood_MainFunction(main_input_file)
