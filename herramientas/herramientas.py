import os 
import io
import re
import sys
import glob
import shutil
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor

import tqdm
import requests
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
from urllib.error import HTTPError

from . import constantes as c

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

def download_fabdem(minx: float, miny: float, maxx: float, maxy: float, output_dir: str = None, cleanup: bool = False):
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
    zip_files = intersecting_tiles["zipfile_name"]
    # tile_files = intersecting_tiles['file_name'] # We cannot use this because filenames do not always match

    # Download Fabdem zip folders and unzip
    max_threads = min(os.cpu_count(), len(zip_files))
    with ThreadPoolExecutor(max_threads) as executor:
        executor.map(_download_fabdem, zip_files)

    # Move the files to the output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True) 
        output_dir = os.path.abspath(output_dir)
        pattern = re.compile(r'([NS])(\d+)[EW](\d+)')  # Pattern to extract the tile coordinates
        logging.info(f"Moving files to '{output_dir}'")
        for zip_file in zip_files:
            zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
            for tile_file in glob.glob(os.path.join(zip_folder, "*.tif")):
                basename = os.path.basename(tile_file)
                match = pattern.search(basename)
                if match:
                    ns = 1 if match.group(1) == 'N' else -1
                    ew = 1 if match.group(3) == 'E' else -1
                    lat = ns * int(match.group(2))
                    lon = ew * int(match.group(3))
                    if minx <= lon + 1 and maxx >= lon and miny <= lat + 1 and maxy >= lat:
                        shutil.copy(tile_file, os.path.join(output_dir, basename))
                else:
                    logging.warning(f"Could not extract tile coordinates from {tile_file}")

    if cleanup:
        logging.info("Cleaning up")
        for zip_file in zip_files:
            zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
            if os.path.exists(zip_folder):
                shutil.rmtree(zip_folder)
       

def _download_fabdem(zip_file: str):
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
            with tqdm.tqdm(total=total_size, unit='B', desc="Downloading") as progress_bar:
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

def crop_dem(minx: float, miny: float, maxx: float, maxy: float, dem_file: str, output_file: str):
    pass

def get_files_in_extent(rasters: list[str], minx: float, miny: float, maxx: float, maxy: float):
    pass

