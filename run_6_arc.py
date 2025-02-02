from herramientas.herramientas import read_config_yaml

from arc import Arc

configs = read_config_yaml('config.yml')
main_input_file = configs['main_input_file']

# Run the main input file
Arc(main_input_file).run()
