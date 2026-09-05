from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import pdal
import json

app = FastAPI()

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

LA_COUNTY_GIS_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/0/query"

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/get-merged-topo")
async def get_merged_topo(payload: dict):
    apns = payload.get("apns", [])
    buffer_feet = payload.get("buffer_feet", 20.0)

    if not apns:
        raise HTTPException(status_code=400, detail="No APNs provided.")

    parcel_polygons = []
    
    # 1. Fetch LA County Parcel Geometry for each APN
    for apn in apns:
        clean_apn = apn.replace("-", "").strip()
        params = {
            'where': f"AIN='{clean_apn}' OR APN='{apn}'",
            'outFields': 'AIN,APN',
            'outSR': '4326',  # WGS84
            'f': 'geojson'
        }
        res = requests.get(LA_COUNTY_GIS_URL, params=params).json()
        
        if not res.get('features'):
            raise HTTPException(status_code=404, detail=f"APN {apn} not found.")
            
        geom = shape(res['features'][0]['geometry'])
        parcel_polygons.append(geom)

    # 2. Merge contiguous parcels & buffer boundary
    merged_boundary = unary_union(parcel_polygons)
    buffer_degrees = buffer_feet / 364000.0  # Approx conversion to degrees
    buffered_boundary = merged_boundary.buffer(buffer_degrees)
    
    minx, miny, maxx, maxy = buffered_boundary.bounds

    # 3. Query USGS 3DEP LiDAR Point Cloud via PDAL
    pdal_pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_CA_LosAngeles_2016/",
                "bounds": f"([{minx}, {maxx}], [{miny}, {maxy}])"
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Ground points only
            },
            {
                "type": "filters.reprojection",
                "out_srs": "EPSG:4326"
            }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pdal_pipeline))
    pipeline.execute()
    
    raw_points = pipeline.arrays[0]
    
    # Format points for client
    points = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in raw_points]

    return {
        "apns": apns,
        "merged_boundary": mapping(merged_boundary),
        "point_count": len(points),
        "points": points
    }