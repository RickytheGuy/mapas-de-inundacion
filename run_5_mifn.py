from herramientas.herramientas import read_config_yaml, create_main_input_file

configs = read_config_yaml('config.yml')
main_input_file = configs['main_input_file']

create_main_input_file(main_input_file, configs)

