from herramientas.herramientas import read_config_yaml

from curve2flood import Curve2Flood_MainFunction

configs = read_config_yaml('config.yml')
main_input_file = configs['main_input_file']

# Run the main input file
Curve2Flood_MainFunction(main_input_file)
