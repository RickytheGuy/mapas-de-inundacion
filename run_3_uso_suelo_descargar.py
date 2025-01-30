from herramientas.herramientas import download_land_use, read_config_yaml

configs = read_config_yaml('config.yml')
minx = configs['minx']
miny = configs['miny']
maxx = configs['maxx']
maxy = configs['maxy']
land_use_dir = configs['land_use_dir']

download_land_use(minx, miny, maxx, maxy, land_use_dir)
