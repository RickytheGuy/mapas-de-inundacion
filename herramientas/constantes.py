import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FABDEM_GEOJSON_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/FABDEM_v1-2_tiles.geojson"
FABDEM_BASE_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"

VPU_BOUNDARIES_URL = "s3://geoglows-v2/streams-global/vpu-boundaries.gpkg"
STREAMLINES_BASE_URL = "s3://rfs-v2/hydrography/"
