from herramientas.herramientas import download_fabdem, read_config_yaml

configs = read_config_yaml('config.yml')
minx = configs['minx']
miny = configs['miny']
maxx = configs['maxx']
maxy = configs['maxy']
dem_dir = configs['dem_dir']

download_fabdem(minx, miny, maxx, maxy, dem_dir)


