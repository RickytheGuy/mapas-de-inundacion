from herramientas.herramientas import download_geoglows_streams, read_config_yaml

configs = read_config_yaml('config.yml')
minx = configs['minx']
miny = configs['miny']
maxx = configs['maxx']
maxy = configs['maxy']
streamlines_dir = configs['streamlines_dir']
output_streamlines = configs['output_streamlines']

download_geoglows_streams(minx, miny, maxx, maxy, streamlines_dir, output_streamlines)


