from herramientas.herramientas import crop_and_resize_land_cover

dem = r"C:\Users\lrr43\mapas-de-inundacion\dems_recortados\recorte.tif"
input_land_use = r'C:\Users\lrr43\mapas-de-inundacion\land\N18W069.tif'
output_land_use = r"C:\Users\lrr43\mapas-de-inundacion\land\lu_recorte.tif"

crop_and_resize_land_cover(dem, input_land_use, output_land_use, vrt=False)