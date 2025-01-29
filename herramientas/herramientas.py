import os 
import io
import re
import sys
import glob
import shutil
import logging
import zipfile
from typing import Union
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor

import tqdm
import requests
from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import box

from . import constantes as c

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
gdal.UseExceptions()

def download_fabdem(minx: float, 
                    miny: float, 
                    maxx: float, 
                    maxy: float, 
                    output_dir: str, 
                    cleanup: bool = False) -> None:
    if not output_dir:
        logging.error(f"Please provide an output directory")
        return
    if minx >= maxx or miny >= maxy:
        logging.error("Invalid bounding box")
        return

    # Load tile geojson
    FABDEM_GEOJSON_PATH = os.path.join(c.CACHE_DIR, "FABDEM_v1-2_tiles.geojson")
    if not os.path.exists(FABDEM_GEOJSON_PATH):
        try:
            tiles_df: gpd.GeoDataFrame = gpd.read_file(c.FABDEM_GEOJSON_URL)
        except HTTPError:
            logging.error(f"Error downloading {FABDEM_GEOJSON_PATH=}... Does this file exist?")
            return
        tiles_df.to_file(FABDEM_GEOJSON_PATH)
    else:
        tiles_df = gpd.read_file(FABDEM_GEOJSON_PATH)

    # Find which Fabdem tiles intersect with the bounding box
    bbox = box(minx, miny, maxx, maxy)
    intersecting_tiles = tiles_df[tiles_df.intersects(bbox)]

    # Get filenames
    zip_files: list[str] = intersecting_tiles["zipfile_name"].tolist()
    # tile_files = intersecting_tiles['file_name'] # We cannot use this because filenames do not always match

    # Download Fabdem zip folders and unzip
    max_threads = min(os.cpu_count(), len(zip_files))
    with ThreadPoolExecutor(max_threads) as executor:
        executor.map(_download_fabdem, [(z, pos) for pos, z in enumerate(zip_files)])

    # Move the files to the output directory
    os.makedirs(output_dir, exist_ok=True) 
    output_dir = os.path.abspath(output_dir)
    logging.info(f"Moving files to '{output_dir}'")
    for zip_file in zip_files:
        zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
        tiles_in_extent = get_fabdem_in_extent(zip_folder, minx, miny, maxx, maxy)
        for tile_file in tiles_in_extent:
            shutil.copy(tile_file, os.path.join(output_dir, os.path.basename(tile_file)))
        else:
            logging.warning(f"Could not extract tile coordinates from {tile_file}")

    if cleanup:
        logging.info("Cleaning up")
        for zip_file in zip_files:
            zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
            if os.path.exists(zip_folder):
                shutil.rmtree(zip_folder)
       
def _download_fabdem(zip_file: str, 
                     pbar_pos: int) -> None:
    zip_url = c.FABDEM_BASE_URL + zip_file
    zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
    if not os.path.exists(zip_folder):
        logging.info(f"Downloading {zip_file} to {c.CACHE_DIR}")
        try:
            # Send a GET request to the URL
            response = requests.get(zip_url, stream=True, allow_redirects=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get('Content-Length', 0))

            # Check if the response is a ZIP file
            if 'application/zip'not in response.headers.get('Content-Type', ''):
                logging.warning("The URL does not appear to point to a ZIP file.")

            # Create a BytesIO object to store the downloaded content
            downloaded_data = io.BytesIO()
            with tqdm.tqdm(total=total_size, unit='B', desc="Downloading", position=pbar_pos) as progress_bar:
                # Download in 1 KB chunks; this seems fastest for this dataset
                for chunk in response.iter_content(chunk_size=1024):  
                    downloaded_data.write(chunk)
                    progress_bar.update(len(chunk))

            os.makedirs(zip_folder, exist_ok=True)
            with zipfile.ZipFile(downloaded_data) as zip_file:
                zip_file.extractall(path=zip_folder)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download the file: {e}")
        except zipfile.BadZipFile as e:
            logging.error(f"Failed to extract the file: {e}")
    else:
        logging.debug(f"{zip_folder} already exists")

def crop_and_merge(minx: float, 
                   miny: float, 
                   maxx: float, 
                   maxy: float, 
                   rasters: list[str], 
                   output_file: str, 
                   vrt: bool = False) -> None:
    # We assume all files passed in are in 4326 projection, and within the extent
    if not rasters:
        logging.error("No rasters to process")
        return
    
    # Needed for VRTs to work
    rasters = [os.path.abspath(raster) for raster in rasters]
    
    if vrt:
        if output_file.endswith('.tif'):
            output_file = output_file.replace('.tif', '.vrt')
        options = gdal.BuildVRTOptions(outputBounds=[minx, miny, maxx, maxy])
        gdal.BuildVRT(output_file, rasters, options=options)
    else:
        options = gdal.WarpOptions(creationOptions=['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'PREDICTOR=2'],
                                   format='GTiff', 
                                   outputBounds=[minx, miny, maxx, maxy])
        gdal.Warp(output_file, rasters, options=options)

def get_fabdem_in_extent(dem_dir: str,
                         minx: float, 
                         miny: float, 
                         maxx: float, 
                         maxy: float) -> list[str]:
    output = []
    pattern = re.compile(r'([NS])(\d+)[EW](\d+)')  # Pattern to extract the tile coordinates

    for file in glob.glob(os.path.join(dem_dir, "*.tif")):
        basename = os.path.basename(file)
        match = pattern.search(basename)
        if match:
            ns = 1 if match.group(1) == 'N' else -1
            ew = 1 if match.group(3) == 'E' else -1
            lat = ns * int(match.group(2))
            lon = ew * int(match.group(3))
            if minx <= lon + 1 and maxx >= lon and miny <= lat + 1 and maxy >= lat:
                output.append(file)        
    return output

def get_dems_in_extent(dem_dir: str, 
                       minx: float, 
                       miny: float, 
                       maxx: float, 
                       maxy: float) -> list[str]:
    # We assume that the files are in 4326 projection
    output = []
    for file in glob.glob(os.path.join(dem_dir, "*.tif")):
        ds: gdal.Dataset = gdal.Open(file)
        gt = ds.GetGeoTransform()
        x = gt[0] + gt[1] * ds.RasterXSize
        y = gt[3] + gt[5] * ds.RasterYSize
        if minx <= x and maxx >= gt[0] and miny <= gt[3] and maxy >= y:
            output.append(file)
        ds = None
    return output

def download_geoglows_streams(minx: float,
                              miny: float,
                              maxx: float,
                              maxy: float,
                              output_dir: str = None) -> list[str]:
    # Get VPU boundaries
    VPU_BOUNDARIES_PATH = os.path.join(c.CACHE_DIR, "vpu-boundaries.gpkg")
    if os.path.exists(VPU_BOUNDARIES_PATH):
        vpu_boundaries: gpd.GeoDataFrame = gpd.read_file(VPU_BOUNDARIES_PATH)
    else:
        logging.info("Downloading VPU boundaries... this may take some time")
        vpu_boundaries: gpd.GeoDataFrame = gpd.read_file(c.VPU_BOUNDARIES_URL)
        # Simply geometries to reduce size
        vpu_boundaries['geometry'] = vpu_boundaries['geometry'].simplify(0.001)
        vpu_boundaries = vpu_boundaries.to_crs('EPSG:4326')
        vpu_boundaries.to_file(VPU_BOUNDARIES_PATH)

    bbox = box(minx, miny, maxx, maxy)
    intersecting_vpus = vpu_boundaries[vpu_boundaries.intersects(bbox)]
    if intersecting_vpus.empty:
        logging.error("No VPU boundaries intersect with the bounding box")
        return
    
    # Get the VPU names
    vpus: list[str] = intersecting_vpus['VPU'].tolist()

    # Download streamlines to cache
    max_threads = min(os.cpu_count(), len(vpus))
    with ThreadPoolExecutor(max_threads) as executor:
        file_paths = executor.map(_download_geoglows_streams, vpus, range(len(vpus)), [output_dir] * len(vpus))

    return list(file_paths)
    


def _download_geoglows_streams(vpu: str, 
                               pbar_pos: int,
                               output_dir: str = None) -> str:
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = c.CACHE_DIR

    # Save as a geoparquet to save disk space and read/write times
    vpu_file = os.path.join(output_dir, f'streams_{vpu}.geoparquet')
    if os.path.exists(vpu_file):
        logging.debug(f"{vpu_file} already exists")
        return vpu_file
    
    # If not cached, download it
    vpu_url = f'{c.STREAMLINES_BASE_URL}vpu={vpu}/streams_{vpu}.gpkg'
    logging.info(f"Downloading {vpu} streamlines to {c.CACHE_DIR}")
    gpd.read_file(vpu_url).to_file(vpu_file)
    return vpu_file

def rasterize_streams(streams_files: Union[str, list[str]], 
                     output_file: str, 
                     resolution: float = 0.0001, 
                     minx: float = None, 
                     miny: float = None, 
                     maxx: float = None, 
                     maxy: float = None) -> None:
    # We assume that the streams
    if not streams_files:
        logging.error("No streams file provided")
        return
    if not output_file:
        logging.error("No output file provided")
        return
    
    if isinstance(streams_files, str):
        streams_files = [streams_files]

    # Load the streams
    # streams_df: gpd.GeoDataFrame = gpd.read_file(streams_file)


    # Get the extent of the streams
    if minx is None or miny is None or maxx is None or maxy is None:
        bounds = streams_df.total_bounds
        minx, miny, maxx, maxy = bounds

    # Create a raster with the extent and resolution
    x_size = int((maxx - minx) / resolution)
    y_size = int((maxy - miny) / resolution)
    driver = gdal.GetDriverByName('GTiff')
    raster_ds: gdal.Dataset = driver.Create(output_file, x_size, y_size, 1, gdal.GDT_Byte)
    raster_ds.SetGeoTransform((minx, resolution, 0, maxy, 0, -resolution))

    # Rasterize the streams
    gdal.RasterizeLayer(raster_ds, [1], streams_df.geometry.__geo_interface__, burn_values=[1])

    # Clean up
    raster_ds.FlushCache()
    raster_ds = None
