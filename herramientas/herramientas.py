import os 
import io
import re
import sys
import glob
import yaml
import shutil
import atexit
import logging
import zipfile
import datetime
import warnings
from typing import Union
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor

import boto3
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm.auto import tqdm
from osgeo import gdal, ogr
from botocore import UNSIGNED
from shapely.geometry import box
from botocore.client import Config

from . import constantes as c

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
gdal.UseExceptions()

S3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, response_checksum_validation='when_required'))
# Silences checksum warning
STORAGE_OPTIONS={"anon": True, 'config_kwargs': {'response_checksum_validation':'when_required'}}
CB_FMT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
FABDEM_PATTERN = re.compile(r'([NS])(\d+)[EW](\d+)')  # Pattern to extract the tile coordinates

USE_PARQUET = gdal.GetDriverByName("Parquet") is not None

def read_config_yaml(config_file: str) -> dict:
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            logging.error(f"Error reading the YAML file: {e}")
            return {}

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
    zip_files: set[str] = set(intersecting_tiles["zipfile_name"])
    # tile_files = intersecting_tiles['file_name'] # We cannot use this because filenames do not always match

    # Download Fabdem zip folders and unzip
    max_threads = min(os.cpu_count(), len(zip_files))
    with ThreadPoolExecutor(max_threads) as executor:
        executor.map(_download_fabdem, zip_files)

    # Move the files to the output directory
    os.makedirs(output_dir, exist_ok=True) 
    output_dir = os.path.abspath(output_dir)
    logging.info(f"Moving files to '{output_dir}'")
    for zip_file in zip_files:
        zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
        tiles_in_extent = get_fabdem_in_extent(zip_folder, minx, miny, maxx, maxy)
        for tile_file in tiles_in_extent:
            shutil.copy(tile_file, os.path.join(output_dir, os.path.basename(tile_file)))

    if cleanup:
        logging.info("Cleaning up")
        for zip_file in zip_files:
            zip_folder = os.path.join(c.CACHE_DIR, zip_file.replace('.zip',''))
            if os.path.exists(zip_folder):
                shutil.rmtree(zip_folder)
       
def _download_fabdem(zip_file: str,) -> None:
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
            with tqdm(total=total_size, unit='B', desc="Downloading", leave=True, position=0) as progress_bar:
                # Download in 1 KB chunks; this seems fastest for this dataset
                for chunk in response.iter_content(chunk_size=1024):  # 64 KB chunks
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

    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with tqdm(total=100, desc="Merging rasters", bar_format=CB_FMT) as pbar:
        _pbar = lambda info, *args: pbar.update(round(info * 100 - pbar.n))
        if vrt:
            if output_file.endswith('.tif'):
                output_file = output_file.replace('.tif', '.vrt')
            options = gdal.BuildVRTOptions(outputBounds=[minx, miny, maxx, maxy],
                                           callback=_pbar)
            gdal.BuildVRT(output_file, rasters, options=options)
        else:
            options = gdal.WarpOptions(creationOptions=['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'PREDICTOR=2'],
                                    format='GTiff', 
                                    outputBounds=[minx, miny, maxx, maxy],
                                    callback=_pbar)
            gdal.Warp(output_file, rasters, options=options)

def get_fabdem_in_extent(dem_dir: str,
                         minx: float, 
                         miny: float, 
                         maxx: float, 
                         maxy: float) -> list[str]:
    output = []

    for file in glob.glob(os.path.join(dem_dir, "*.tif")):
        basename = os.path.basename(file)
        match = FABDEM_PATTERN.search(basename)
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
                              streamlines_dir: str,
                              output_streamlines: str) -> None:
    if not output_streamlines:
        logging.error(f"Please provide an output streamlines file")
        return
    
    output_streamlines = os.path.abspath(output_streamlines)
    
    if minx >= maxx or miny >= maxy:
        logging.error("Invalid bounding box")
        return
    
    if not streamlines_dir:
        logging.error(f"Please provide a streamlines directory")
        return
    
    streamlines_dir = os.path.abspath(streamlines_dir)
    
    # Get VPU boundaries
    if USE_PARQUET:
        VPU_BOUNDARIES_PATH = os.path.join(c.CACHE_DIR, "vpu-boundaries.parquet")
        if os.path.exists(VPU_BOUNDARIES_PATH):
            vpu_boundaries: gpd.GeoDataFrame = gpd.read_parquet(VPU_BOUNDARIES_PATH)
        else:
            logging.info("Downloading VPU boundaries... this may take some time")
            vpu_boundaries: gpd.GeoDataFrame = gpd.read_file(c.VPU_BOUNDARIES_URL)
            # Simply geometries to reduce size
            vpu_boundaries['geometry'] = vpu_boundaries['geometry'].simplify(0.001)
            vpu_boundaries = vpu_boundaries.to_crs('EPSG:4326')
            vpu_boundaries.to_parquet(VPU_BOUNDARIES_PATH)
    else:
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
    vpus: set[str] = set(intersecting_vpus['VPU'])

    # Download streamlines to cache
    max_threads = min(os.cpu_count(), len(vpus))
    with ThreadPoolExecutor(max_threads) as executor:
        file_paths = executor.map(_download_geoglows_streams, vpus, [streamlines_dir] * len(vpus))

    # Now combine the files and crop to extent
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4326').to_crs('EPSG:3857')
    gdf: gpd.GeoDataFrame = pd.concat([gpd.read_file(file, bbox=bbox_gdf) for file in file_paths])
        
    if gdf.empty:
        logging.error("No streamlines found in the bounding box")
        return
    
    gdf = gdf.to_crs('EPSG:4326')
    
    if output_streamlines.lower().endswith(('.parquet', '.geoparquet')):
        if USE_PARQUET:
            gdf.to_parquet(output_streamlines)
        else:
            logging.error("Cannot save as parquet. Please use GeoPackage or Shapefile")
            raise ValueError("Cannot save as parquet. Please use GeoPackage or Shapefile")
    else:
        gdf.to_file(output_streamlines)

def download_from_s3(local_file_name, s3_bucket, s3_object_key):
    meta_data = S3.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    with tqdm(total=total_length,  desc=os.path.basename(local_file_name), unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        with open(local_file_name, 'wb') as f:
            S3.download_file(s3_bucket, s3_object_key, local_file_name, Callback=pbar.update)


def _download_geoglows_streams(vpu: str, 
                               output_dir: str = None) -> str:
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = c.CACHE_DIR

    vpu_file = os.path.join(output_dir, f'streams_{vpu}.gpkg')
    if os.path.exists(vpu_file):
        logging.debug(f"{vpu_file} already exists")
        return vpu_file
    
    # If not cached, download it
    logging.info(f"Downloading VPU {vpu} to {vpu_file}")
    download_from_s3(vpu_file, 'rfs-v2', f'hydrography/vpu={vpu}/streams_{vpu}.gpkg')       

    return vpu_file

def rasterize_streams(dem: str, 
                      streams_files: Union[str, list[str]],  
                      output_file: str,  
                      burn_value: str = 'LINKNO') -> None:
    # We assume that the DEM is in 4326 projection
    if not streams_files:
        logging.error("No streams file provided")
        return
    if not output_file:
        logging.error("No output file provided")
        return
    
    output_file = os.path.abspath(output_file)
    
    if isinstance(streams_files, str):
        streams_files = [streams_files]

    # Load the dem
    ds: gdal.Dataset = gdal.Open(dem)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    # Create output raster
    raster_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(output_file, width, height, 1, gdal.GDT_Int32)
    raster_ds.SetGeoTransform(gt)
    raster_ds.SetProjection(proj)

    # Rasterize the streams
    for streams_file in streams_files:
        stream_ds: gdal.Dataset = ogr.Open(streams_file)
        layer = stream_ds.GetLayer()
        with tqdm(total=100, desc=os.path.basename(streams_file), bar_format=CB_FMT) as pbar:
            gdal.RasterizeLayer(raster_ds, 
                                [1], 
                                layer, 
                                options=[f"ATTRIBUTE={burn_value}"], 
                                callback=lambda info, *args: pbar.update(round(info * 100 - pbar.n, 2)))

    # Clean up
    raster_ds.FlushCache()
    raster_ds = None

def ordinal(n: int) -> str:
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)] if not 11 <= (n % 100) <= 13 else 'th'
    return f"{n}{suffix}"
    
def clean_stream_raster(stream_raster: str, num_passes: int = 2) -> None:
    """
    This function comes from Mike Follum's ARC at https://github.com/MikeFHS/automated-rating-curve
    """
    assert num_passes > 0, "num_passes must be greater than 0"
    
    # Get stream raster
    stream_ds: gdal.Dataset = gdal.Open(stream_raster, gdal.GA_Update)
    array: np.ndarray = stream_ds.ReadAsArray().astype(np.int64)
    
    # Create an array that is slightly larger than the STRM Raster Array
    array = np.pad(array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    row_indices, col_indices = array.nonzero()
    num_nonzero = len(row_indices)
    
    passes = []
    pbar = tqdm(total=num_nonzero * num_passes * 2, unit='cells', leave=True, position=0)
    for _ in range(num_passes):
        # First pass is just to get rid of single cells hanging out not doing anything
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        n=0
        for x in range(num_nonzero):
            pbar.update(1)
            if x >= p_count * p_percent:
                p_count = p_count + 1
            r = row_indices[x]
            c = col_indices[x]
            if array[r,c] <= 0:
                continue

            # Left and Right cells are zeros
            if array[r,c + 1] == 0 and array[r, c - 1] == 0:
                # The bottom cells are all zeros as well, but there is a cell directly above that is legit
                if (array[r+1,c-1]+array[r+1,c]+array[r+1,c+1])==0 and array[r-1,c]>0:
                    array[r,c] = 0
                    n=n+1
                # The top cells are all zeros as well, but there is a cell directly below that is legit
                elif (array[r-1,c-1]+array[r-1,c]+array[r-1,c+1])==0 and array[r+1,c]>0:
                    array[r,c] = 0
                    n=n+1
            # top and bottom cells are zeros
            if array[r,c]>0 and array[r+1,c]==0 and array[r-1,c]==0:
                # All cells on the right are zero, but there is a cell to the left that is legit
                if (array[r+1,c+1]+array[r,c+1]+array[r-1,c+1])==0 and array[r,c-1]>0:
                    array[r,c] = 0
                    n=n+1
                elif (array[r+1,c-1]+array[r,c-1]+array[r-1,c-1])==0 and array[r,c+1]>0:
                    array[r,c] = 0
                    n=n+1
        
        passes.append(n)
        
        # This pass is to remove all the redundant cells
        n = 0
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        for x in range(num_nonzero):
            pbar.update(1)
            if x >= p_count * p_percent:
                p_count = p_count + 1
            r = row_indices[x]
            c = col_indices[x]
            value = array[r,c]
            if value<=0:
                continue

            if array[r+1,c] == value and (array[r+1, c+1] == value or array[r+1, c-1] == value):
                if array[r+1,c-1:c+2].max() == 0:
                    array[r+ 1 , c] = 0
                    n = n + 1
            elif array[r-1,c] == value and (array[r-1, c+1] == value or array[r-1, c-1] == value):
                if array[r-1,c-1:c+2].max() == 0:
                    array[r- 1 , c] = 0
                    n = n + 1
            elif array[r,c+1] == value and (array[r+1, c+1] == value or array[r-1, c+1] == value):
                if array[r-1:r+1,c+2].max() == 0:
                    array[r, c + 1] = 0
                    n = n + 1
            elif array[r,c-1] == value and (array[r+1, c-1] == value or array[r-1, c-1] == value):
                if array[r-1:r+1,c-2].max() == 0:
                        array[r, c - 1] = 0
                        n = n + 1

        passes.append(n)
    
    # Write the cleaned array to the raster
    stream_ds.WriteArray(array[1:-1, 1:-1])

    pbar.close()
    for _pass in passes:
        logging.info(f"{ordinal(passes.index(_pass) + 1)} pass removed {_pass} cells") 
    
def download_land_use(minx: float,
                      miny: float,
                      maxx: float,
                      maxy: float,
                      output_dir: str,) -> list[str]:
    """
    Inspired by Mike Follum
    """
    if not output_dir:
        logging.error(f"Please provide an output directory")
        return
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if minx >= maxx or miny >= maxy:
        logging.error("Invalid bounding box")
        return
    
    # Get the tiles that intersect with the bounding box
    esa_tiles_file = os.path.join(c.CACHE_DIR, "esa_worldcover_tiles.gpkg")
    if os.path.exists(esa_tiles_file):
        esa_df: gpd.GeoDataFrame = gpd.read_file(esa_tiles_file)
    else:
        logging.info("Downloading ESA Worldcover tiles...")
        esa_df: gpd.GeoDataFrame = gpd.read_file(f"{c.ESA_BASE_URL}/esa_worldcover_grid.geojson")
        esa_df.to_file(esa_tiles_file)

    bbox = box(minx, miny, maxx, maxy)
    intersecting_tiles = esa_df[esa_df.intersects(bbox)]

    if intersecting_tiles.empty:
        logging.error("No ESA Worldcover tiles intersect with the bounding box")
        return
    
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = c.CACHE_DIR

    tiles = set(intersecting_tiles['ll_tile'])
    # Download the tiles
    max_threads = min(os.cpu_count(), len(tiles))
    with ThreadPoolExecutor(max_threads) as executor:
        file_paths = executor.map(_download_esa_tile, tiles, [output_dir] * len(tiles))

    return list(file_paths)

def _download_esa_tile(tile: str, 
                      output_dir: str = None) -> str:
    output_dir = os.path.abspath(output_dir)

    # Save as a geoparquet to save disk space and read/write times
    tile_file = os.path.join(output_dir, f'{tile}.tif')
    if os.path.exists(tile_file):
        logging.debug(f"{tile_file} already exists")
        return tile_file
    
    # If not cached, download it
    tile_url = f"{c.ESA_BASE_URL}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    logging.info(f"Downloading {tile} to {output_dir}")
    
    # Use gdal to create a local copy with compression (literally saves GB of space)
    with tqdm(total=100, desc=tile, bar_format=CB_FMT) as pbar:
        options = gdal.TranslateOptions(format='GTiff', 
                                        creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2'],
                                        callback=lambda info, *args: pbar.update(round(info * 100 - pbar.n, 2)))

        gdal.Translate(tile_file, tile_url, options=options)

def crop_and_resize_land_cover(dem: str,
                               input_land_use: Union[str, list[str]],
                               output_land_use: str,
                               vrt: bool = False) -> None:
    if not input_land_use:
        logging.error("No land use files provided")
        return
    
    output_land_use = os.path.abspath(output_land_use)

    if isinstance(input_land_use, str):
        input_land_use = [input_land_use]

    input_land_use = [os.path.abspath(land) for land in input_land_use]

    # Get DEM information
    ds: gdal.Dataset = gdal.Open(dem)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    # Create output raster
    with tqdm(total=100, desc="Merging land use", bar_format=CB_FMT) as pbar:
        _pbar = lambda info, *args: pbar.update(round(info * 100 - pbar.n))
        if vrt:
            if output_land_use.endswith('.tif'):
                output_land_use = output_land_use.replace('.tif', '.vrt')
            options = gdal.BuildVRTOptions(outputBounds=[gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]], 
                                        srcNodata=0, 
                                        xRes=abs(gt[1]), 
                                        yRes=abs(gt[5]),
                                        outputSRS=proj,
                                        callback=_pbar)
            gdal.BuildVRT(output_land_use, input_land_use, options=options)
        else:
            options = gdal.WarpOptions(creationOptions=['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'PREDICTOR=2'],
                                    format='GTiff',
                                    outputBounds=[gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]],
                                    dstSRS=proj,
                                    width=width,
                                    height=height,
                                    dstNodata=0,
                                    callback=_pbar)
            gdal.Warp(output_land_use, input_land_use, options=options)

def get_linknos(stream_raster: str,) -> np.ndarray:
    ds: gdal.Dataset = gdal.Open(stream_raster)
    array = ds.ReadAsArray()
    linknos = np.unique(array)
    if linknos[0] == 0:
        linknos = linknos[1:]
    return linknos

def get_base_max(stream_raster: str,
                 base_max_file: str,) -> None:
    linknos = get_linknos(stream_raster)
    base_max_file = os.path.abspath(base_max_file)

    # Open the return period zarr
    # Select return period 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_zarr(c.RETURN_PERIODS_ZARR_URL, storage_options=STORAGE_OPTIONS) as ds:
            # Filter to only include existing values
            existing = set(ds['river_id'].values)
            linknos = [r for r in linknos if r in existing]

            df = (
                ds.sel(river_id=linknos, return_period=2)
                .to_dataframe()
                .drop(columns='return_period')
            )

    df['gumbel'] = df['logpearson3'].fillna(df['gumbel'])
    df['max_simulated'] *= 1.5
    df = df.drop(columns=['logpearson3'])

    # Save csv
    (
        df.rename(columns={'gumbel':'baseflow', 'max_simulated':'max'})
        .round(2)
        .to_csv(base_max_file)
    )
    logging.info(f"Base and max values saved to {base_max_file}")

def get_return_period(stream_raster: str,
                      rp: int, 
                      flow_file: str,) -> None:
    linknos = get_linknos(stream_raster)
    flow_file = os.path.abspath(flow_file)

    logging.info("Opening return periods Zarr file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(c.RETURN_PERIODS_ZARR_URL, storage_options=STORAGE_OPTIONS)
        # Filter linknos to only include existing values
        existing = set(ds['river_id'].values)
        linknos = [r for r in linknos if r in existing]
    try:
        df = ds.sel(river_id=linknos, return_period=rp).to_dataframe()
    except KeyError:
        logging.error(f"Return period {rp} not found in the dataset. Available return periods are {', '.join(ds['return_period'].values.astype(str))}")
        return
    
    df = (
        df.reset_index()
        .drop(columns='return_period')
        .round(2)
    )
    df['logpearson3'] = df['logpearson3'].fillna(df[f'gumbel'])
    (
        df.drop(columns=['gumbel'])
        .rename(columns={'logpearson3': f"rp{rp}"})
        .to_csv(flow_file, index=False)
    )
    logging.info(f"Return period values saved to {flow_file}")

def get_historical(stream_raster: str,
                   flow_file: str,
                   date_start: str,
                   date_end: str = None,) -> None:
    linknos = get_linknos(stream_raster)
    flow_file = os.path.abspath(flow_file)

    if date_end and date_end != date_start:
        if date_end < date_start:
            logging.error("End date is before start date")
            return
        selection = np.arange(date_start, date_end, dtype='datetime64[h]')
    else:
        selection = np.arange(date_start, date_start + datetime.timedelta(days=1), dtype='datetime64[h]')
    
    logging.info("Opening the Zarr file")
    ds = xr.open_zarr(c.DAILY_ZARR_URL, storage_options=STORAGE_OPTIONS)
    existing = set(ds['river_id'].values)
    linknos = [linkno for linkno in linknos if linkno in existing]
    (  
        ds.sel(river_id=linknos)
        .sel(time=selection, method='nearest')
        ['Q']
        .max(dim='time')
        .to_dataframe()
        .reset_index()
        .rename(columns={'Q':f"{date_start}{'--' + str(date_end) if date_end else ''}", 'river_id':'linkno'})
        .round(2)
        .to_csv(flow_file, index=False)
    )
    logging.info(f"Historical values saved to {flow_file}")

def create_main_input_file(out_path: str, configs: dict) -> None:
    with open(out_path, 'w') as f:
        f.write("# Main input file for ARC and Curve2Flood\n\n")

        f.write("\n# Input files - Required\n")
        f.write(f"DEM_File\t{os.path.abspath(configs['output_dem'])}\n")
        f.write(f"Stream_File\t{os.path.abspath(configs['output_streams'])}\n")
        f.write(f"LU_Raster_SameRes\t{os.path.abspath(configs['output_land_use'])}\n")
        f.write(f"LU_Manning_n\t{os.path.abspath(configs['land_use_text_file'])}\n")
        f.write(f"Flow_File\t{os.path.abspath(configs['base_max_file'])}\n")
        f.write(f"COMID_Flow_File\t{os.path.abspath(configs['flow_file'])}\n")

        f.write("\n# Input files - Optional\n")
        if configs['output_streamlines'] and configs.get('create_shp_file', False): 
            streams = os.path.abspath(configs['output_streamlines'])  
            if streams.endswith(('.parquet', '.geoparquet')):
                # Convert to gpkg
                streams = streams.replace('.parquet', '.gpkg')
                streams = streams.replace('.geoparquet', '.gpkg')
                if not os.path.exists(streams):
                    gpd.read_parquet(configs['output_streamlines']).to_file(streams, driver='GPKG')
            f.write(f"StrmShp_File\t{streams}\n")

        f.write("\n# Output files - Required\n")
        f.write(f"Print_VDT_Database\t{os.path.abspath(configs['vdt_file'])}\n")

        f.write("\n# Output files - Optional\n")
        if configs['flood_map']: f.write(f"OutFLD\t{os.path.abspath(configs['flood_map'])}\n")
        if configs['curve_file']:   f.write(f"Print_Curve_File\t{os.path.abspath(configs['curve_file'])}\n")
        if configs['meta_file']:    f.write(f"Meta_File\t{os.path.abspath(configs['meta_file'])}\n")
        if configs['cross_section_file']: f.write(f"XS_Out_File\t{os.path.abspath(configs['cross_section_file'])}\n")

        f.write("\n# Parameters - Required\n")
        f.write(f"Flow_File_ID\triver_id\n")
        f.write(f"Flow_File_BF\tbaseflow\n")
        f.write(f"Flow_File_QMax\tmax\n")
        f.write(f"Spatial_Units\tdeg\n")

        f.write("\n# Parameters - Optional\n")
        if configs['cross_section_distance']:   f.write(f"X_Section_Dist\t{configs['cross_section_distance']}\n")
        if configs['degree_manipulation']:      f.write(f"Degree_Manip\t{configs['degree_manipulation']}\n")
        if configs['degree_interval']:          f.write(f"Degree_Interval\t{configs['degree_interval']}\n")
        if configs['low_spot_range']:           f.write(f"Low_Spot_Range\t{configs['low_spot_range']}\n")
        if configs['direction_distance']:       f.write(f"Gen_Dir_Dist\t{configs['direction_distance']}\n")
        if configs['slope_distance']:           f.write(f"Gen_Slope_Dist\t{configs['slope_distance']}\n")
        if configs['land_use_water_value']:           f.write(f"LC_Water_Value\t{configs['land_use_water_value']}\n")
        if configs['vdt_iterations']:           f.write(f"VDT_Database_NumIterations\t{configs['vdt_iterations']}\n")
        if configs['banks_from_land_use']:            f.write(f"FindBanksBasedOnLandCover\n")
        if configs['reach_average_curves']:               f.write(f"Reach_Average_Curve_File\n")
        if configs['q_fraction']:            f.write(f"Q_Fraction\t{configs['q_fraction']}\n")
        if configs['top_width_max_limit']:        f.write(f"TopWidthPlausibleLimit\t{configs['top_width_max_limit']}\n")
        if configs['top_width_factor']:        f.write(f"TW_MultFact\t{configs['top_width_factor']}\n")
        if configs['flood_local']:        f.write(f"LocalFloodOption\t{configs['flood_local']}\n")
        if configs['flood_obvious_stream_cells']:        f.write(f"Flood_WaterLC_and_STRM_Cells\t{configs['flood_obvious_stream_cells']}\n")

        f.write("\n# Optional ARC Bathymetry\n")
        if configs['output_bathymetry']:
            f.write(f"AROutBATHY\t{os.path.abspath(configs['output_bathymetry'])}\n")
            f.write(f"BATHY_Out_File\t{os.path.abspath(configs['output_bathymetry'])}\n")
            if configs['bathy_trap_h']:          f.write(f"Bathy_Trap_H\t{configs['bathy_trap_h']}\n")
            if configs['bathy_use_banks']:       f.write(f"Bathy_Use_Banks\tTrue\n")

        f.write("\n# Optional Curve2Flood Bathymetry\n")
        if configs['output_c2f_bathymetry']:
            f.write(f"FSOutBATHY\t{os.path.abspath(configs['output_c2f_bathymetry'])}\n")

    logging.info(f"Main input file saved to {out_path}")

def remove_oceans(raster_file: str, oceans_file: str) -> None:
    """
    Remove oceans from the raster file using the oceans shapefile.
    """
    if not os.path.exists(oceans_file):
        logging.error(f"Ocean shapefile {oceans_file} does not exist")
        return
    
    if not os.path.exists(raster_file):
        logging.error(f"Raster file {raster_file} does not exist")
        return

    # Open the raster file
    dem_ds: gdal.Dataset = gdal.Open(raster_file, gdal.GA_Update)
    gt = dem_ds.GetGeoTransform()
    proj = dem_ds.GetProjection()
    
    logging.info('Rasterizing oceans shapefile; this may take a 2 minutes...')
    options = gdal.RasterizeOptions(format='MEM',
                                    outputType=gdal.GDT_Byte,
                                    outputSRS=proj,
                                    xRes=abs(gt[1]),
                                    yRes=abs(gt[5]),
                                    burnValues=[1],
                                    allTouched=True,
                                    outputBounds=[gt[0], gt[3] + dem_ds.RasterYSize * gt[5], gt[0] + dem_ds.RasterXSize * gt[1], gt[3]],)
    oceans_ds = gdal.Rasterize('', oceans_file, options=options)

    # Read the in memory raster into an array
    oceans = oceans_ds.ReadAsArray()

    # Set the ocean values to 0 in the original raster
    dem: np.ndarray = dem_ds.ReadAsArray()
    dem[oceans == 1] = 0
    dem_ds.WriteArray(dem)
    dem_ds.FlushCache()
    dem_ds.GetRasterBand(1).SetNoDataValue(0)
    dem_ds = None
    oceans_ds = None

    return

def exit_handler():
    S3.close()

atexit.register(exit_handler)
