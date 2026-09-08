import os
import sys
import asyncio
import json
import uuid
import io
import requests
import concurrent.futures

import os
import sys
import threading
import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = get_resource_path("index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# Add your existing API routes here (/api/get-merged-topo, /api/jobs/{job_id}, etc.)

def run_server():
    """Start FastAPI server on a local port without logging."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    # Start the FastAPI backend server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Create native desktop window pointing to the local FastAPI server
    webview.create_window(
        title="LA County 3D Topo & Site Plan Generator",
        url="http://127.0.0.1:8000",
        width=1280,
        height=800,
        resizable=True
    )

    # Start the desktop GUI loop
    webview.start()

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

# Locate active conda prefix dynamically
conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
proj_dir = os.path.join(conda_prefix, "share", "proj")

# Export PROJ pathing and environment flags PRIOR to loading pyproj/PDAL C-extensions
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"] = proj_dir
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["PDAL_LOGLEVEL"] = "0"
os.environ["PDAL_DRIVER_PATH"] = ""

import pyproj
from pyproj import Transformer
pyproj.datadir.set_data_dir(proj_dir)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, shape, Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, transform
from shapely.vectorized import contains
import pdal
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import ezdxf
import rasterio
from rasterio.transform import from_origin
from xml.etree import ElementTree as ET
import xml.dom.minidom as minidom

app = FastAPI()

# Process pool executor for offloading heavy Delaunay/TIN computations off FastAPI event loop
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

# Ensure static directory exists before mounting
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

LA_COUNTY_PARCEL_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/0/query"
LA_COUNTY_ZONING_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/1/query"

# --- Lazy-Initialized Coordinate Transformer ---
_transformer = None

def get_transformer():
    global _transformer
    if _transformer is None:
        _transformer = Transformer.from_crs("EPSG:3857", "EPSG:2229", always_xy=True)
    return _transformer

def to_state_plane(x, y):
    return get_transformer().transform(x, y)

# --- In-Memory Cache & Job Tracking ---
CACHE = {
    "raw_points": [],
    "triangles": [],
    "boundary_2229": None,
    "boundary_coords": []
}

jobs = {}

# --- Helper Functions & Geometry Engine ---

def create_subtile_boxes(bounds_3857, tile_size_meters=60.0):
    minx, miny, maxx, maxy = bounds_3857
    boxes = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            sub_minx = x
            sub_maxx = min(x + tile_size_meters, maxx)
            sub_miny = y
            sub_maxy = min(y + tile_size_meters, maxy)
            boxes.append((sub_minx, sub_miny, sub_maxx, sub_maxy))
            y += tile_size_meters
        x += tile_size_meters
    return boxes

def adaptive_process_points(pts, lot_area_sqft=None, min_spacing=1.0):
    """
    Dynamically resamples point clouds based on site area and density.
    Preserves raw micro-topography on small parcels while preventing OOM crashes on large acreage.
    """
    point_count = len(pts)
    if point_count == 0:
        return []

    acreage = (lot_area_sqft / 43560.0) if lot_area_sqft else None

    # Small sites (< 2 acres or < 25,000 points): retain 100% of raw survey points
    if (acreage and acreage <= 2.0) or point_count < 25000:
        return pts

    # Medium sites (2-10 acres): light 1.0 ft grid filter
    elif (acreage and acreage <= 10.0) or point_count < 75000:
        grid_size = 1.0

    # Large parcels (> 10 acres / 76+ acres): resample to a larger grid spacing
    else:
        grid_size = max(2.0, min_spacing)

    grid = {}
    for p in pts:
        cell_key = (int(p["x"] // grid_size), int(p["y"] // grid_size))
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(p)

    downsampled = []
    for cell_pts in grid.values():
        zs = [p["z"] for p in cell_pts]
        med_z = float(np.median(zs))
        # Keep point closest to median Z elevation to avoid artificial height drift
        best_pt = min(cell_pts, key=lambda p: abs(p["z"] - med_z))
        downsampled.append(best_pt)

    return downsampled

def precalculate_tin(pts, boundary_geom, site_acreage):
    """
    High-speed TIN calculation optimized for large acreage.
    Uses vectorized NumPy masks for interior/exterior triangles and only 
    runs Shapely clipping on the thin outer perimeter band crossing the boundary.
    """
    if len(pts) < 3:
        return np.empty((0, 3), dtype=np.float64), [], pts

    coords = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float64)
    tri = Delaunay(coords[:, :2])
    simplices = tri.simplices

    # 1. Fast Vectorized Vertex Masking across all triangles
    p1s, p2s, p3s = coords[simplices[:, 0]], coords[simplices[:, 1]], coords[simplices[:, 2]]
    
    in1 = contains(boundary_geom, p1s[:, 0], p1s[:, 1])
    in2 = contains(boundary_geom, p2s[:, 0], p2s[:, 1])
    in3 = contains(boundary_geom, p3s[:, 0], p3s[:, 1])

    all_inside = in1 & in2 & in3
    all_outside = (~in1) & (~in2) & (~in3)
    boundary_crossers = ~(all_inside | all_outside)

    all_pts = list(pts)
    triangles = []

    # 2. Instantly append 100% interior triangles without Shapely execution
    inside_indices = np.where(all_inside)[0]
    for idx in inside_indices:
        triangles.append({
            "indices": [int(simplices[idx, 0]), int(simplices[idx, 1]), int(simplices[idx, 2])],
            "inside": True
        })

    # 3. Clip ONLY the thin outer perimeter band
    crosser_indices = np.where(boundary_crossers)[0]
    
    for idx in crosser_indices:
        p1, p2, p3 = p1s[idx], p2s[idx], p3s[idx]
        tri_poly = Polygon([p1[:2], p2[:2], p3[:2]])

        clipped = tri_poly.intersection(boundary_geom)
        if clipped.is_empty:
            continue

        clipped_polys = [clipped] if isinstance(clipped, Polygon) else (
            clipped.geoms if isinstance(clipped, MultiPolygon) else []
        )

        for poly in clipped_polys:
            ext_coords = list(poly.exterior.coords)[:-1]
            if len(ext_coords) < 3:
                continue

            poly_indices = []
            denom = ((p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))

            for cx, cy in ext_coords:
                # Interpolate planar Z elevations
                if abs(denom) > 1e-6:
                    w1 = ((p2[1] - p3[1]) * (cx - p3[0]) + (p3[0] - p2[0]) * (cy - p3[1])) / denom
                    w2 = ((p3[1] - p1[1]) * (cx - p3[0]) + (p1[0] - p3[0]) * (cy - p3[1])) / denom
                    w3 = 1.0 - w1 - w2
                    cz = float(w1 * p1[2] + w2 * p2[2] + w3 * p3[2])
                else:
                    cz = float((p1[2] + p2[2] + p3[2]) / 3.0)

                new_idx = len(all_pts)
                all_pts.append({"x": float(cx), "y": float(cy), "z": cz, "inside": True})
                poly_indices.append(new_idx)

            for i in range(1, len(poly_indices) - 1):
                triangles.append({
                    "indices": [poly_indices[0], poly_indices[i], poly_indices[i + 1]],
                    "inside": True
                })

    new_coords = np.array([[p["x"], p["y"], p["z"]] for p in all_pts], dtype=np.float64)
    return new_coords, triangles, all_pts

def calculate_contours_vectorized(coords, triangles, interval=5.0):
    """
    Fully vectorized contour extraction in NumPy. 
    Runs in milliseconds without Python loops over triangles.
    """
    if len(coords) == 0 or not triangles:
        return []

    zs = coords[:, 2]
    min_z = np.floor(zs.min() / interval) * interval
    max_z = np.ceil(zs.max() / interval) * interval
    levels = np.arange(min_z, max_z + interval, interval)

    valid_tris = [t for t in triangles if t.get("inside", True)]
    if not valid_tris:
        return []

    indices = np.array([t["indices"] for t in valid_tris], dtype=np.int32)
    tri_pts = coords[indices]  # Shape: (NumTriangles, 3, 3)

    min_tz = np.min(tri_pts[:, :, 2], axis=1)
    max_tz = np.max(tri_pts[:, :, 2], axis=1)

    contours = []

    for z_level in levels:
        active_mask = (min_tz <= z_level) & (z_level <= max_tz)
        if not np.any(active_mask):
            continue

        act_tris = tri_pts[active_mask]  # Shape: (M, 3, 3)

        # Extract edges (0->1, 1->2, 2->0)
        e0 = act_tris[:, 0, :]
        e1 = act_tris[:, 1, :]
        e2 = act_tris[:, 2, :]

        edges_a = np.stack([e0, e1, e2], axis=1)  # Shape: (M, 3, 3)
        edges_b = np.stack([e1, e2, e0], axis=1)  # Shape: (M, 3, 3)

        z_a = edges_a[:, :, 2]
        z_b = edges_b[:, :, 2]

        crosses = ((z_a <= z_level) & (z_level <= z_b)) | ((z_b <= z_level) & (z_level <= z_a))
        dz = z_b - z_a
        valid = crosses & (np.abs(dz) > 1e-6)

        dz_safe = np.where(valid, dz, 1.0)
        t_factor = np.where(valid, (z_level - z_a) / dz_safe, 0.0)

        ix = edges_a[:, :, 0] + t_factor * (edges_b[:, :, 0] - edges_a[:, :, 0])
        iy = edges_a[:, :, 1] + t_factor * (edges_b[:, :, 1] - edges_a[:, :, 1])

        # Keep triangles yielding exactly 2 edge intersections
        counts = np.sum(valid, axis=1)
        valid_tri_mask = counts == 2

        if not np.any(valid_tri_mask):
            continue

        valid_ix = ix[valid_tri_mask]
        valid_iy = iy[valid_tri_mask]
        valid_cross = valid[valid_tri_mask]

        pts_x = valid_ix[valid_cross].reshape(-1, 2)
        pts_y = valid_iy[valid_cross].reshape(-1, 2)

        formatted_segments = [
            {
                "pts": [
                    {"x": float(pts_x[i, 0]), "y": float(pts_y[i, 0]), "z": float(z_level), "inside": True},
                    {"x": float(pts_x[i, 1]), "y": float(pts_y[i, 1]), "z": float(z_level), "inside": True}
                ],
                "inside": True
            }
            for i in range(len(pts_x))
        ]

        if formatted_segments:
            is_index = bool((round(z_level, 2) % (interval * 5)) == 0)
            contours.append({
                "elevation": round(float(z_level), 2),
                "is_index": is_index,
                "segments": formatted_segments
            })

    return contours

def clip_contours_to_boundary(contours, boundary_geom):
    """
    Clips individual contour line segments exactly to the boundary polygon.
    """
    if not boundary_geom or not contours:
        return contours

    clipped_contours = []
    
    for c_group in contours:
        elevation = c_group["elevation"]
        is_index = c_group["is_index"]
        clipped_segments = []

        for seg in c_group["segments"]:
            p1 = seg["pts"][0]
            p2 = seg["pts"][1]
            line = LineString([(p1["x"], p1["y"]), (p2["x"], p2["y"])])

            intersection = line.intersection(boundary_geom)

            if intersection.is_empty:
                continue

            lines = [intersection] if isinstance(intersection, LineString) else (
                intersection.geoms if isinstance(intersection, MultiLineString) else []
            )

            for l in lines:
                coords = list(l.coords)
                for i in range(len(coords) - 1):
                    clipped_segments.append({
                        "pts": [
                            {"x": float(coords[i][0]), "y": float(coords[i][1]), "z": float(elevation), "inside": True},
                            {"x": float(coords[i+1][0]), "y": float(coords[i+1][1]), "z": float(elevation), "inside": True}
                        ],
                        "inside": True
                    })

        if clipped_segments:
            clipped_contours.append({
                "elevation": elevation,
                "is_index": is_index,
                "segments": clipped_segments
            })

    return clipped_contours

# --- Async Worker Task ---

async def execute_topo_processing(job_id: str, apns: list, buffer_feet: float):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Fetching parcel boundaries and zoning data..."

        parcel_polygons_3857 = []
        zoning_info_list = []

        for apn in apns:
            clean_apn = apn.replace("-", "").strip()
            params = {
                'where': f"AIN='{clean_apn}' OR APN='{apn}'",
                'outFields': '*',
                'outSR': '3857',
                'f': 'geojson'
            }
            res = requests.get(LA_COUNTY_PARCEL_URL, params=params, timeout=20.0).json()

            if not res.get('features'):
                raise Exception(f"APN {apn} not found.")

            feature = res['features'][0]
            geom_3857 = shape(feature['geometry'])
            parcel_polygons_3857.append(geom_3857)

            props = feature.get('properties', {})
            
            z_params = {
                'geometry': f"{geom_3857.centroid.x},{geom_3857.centroid.y}",
                'geometryType': 'esriGeometryPoint',
                'spatialRel': 'esriSpatialRelIntersects',
                'inSR': '3857',
                'outFields': '*',
                'f': 'json'
            }
            try:
                z_res = requests.get(LA_COUNTY_ZONING_URL, params=z_params, timeout=10.0).json()
            except Exception:
                z_res = {}

            zone_code = "N/A"
            zone_desc = "N/A"
            if z_res.get('features'):
                z_props = z_res['features'][0].get('attributes', {})
                zone_code = z_props.get('ZONE_CODE') or z_props.get('ZONING') or z_props.get('ZONE') or "R1"
                zone_desc = z_props.get('ZONE_BYTE') or z_props.get('ZONE_DESC') or "Single Family Residential"

            zoning_info_list.append({
                "apn": props.get("APN") or clean_apn,
                "use_code": props.get("UseCode") or props.get("USE_CODE") or "Residential",
                "use_description": props.get("UseType") or props.get("USE_DESC") or "Single Family Dwelling",
                "zoning_code": zone_code,
                "zoning_description": zone_desc,
                "situs_address": props.get("SitusAddress") or props.get("SITUS_ADDR") or "N/A"
            })

        merged_boundary_3857 = unary_union(parcel_polygons_3857) if len(parcel_polygons_3857) > 1 else parcel_polygons_3857[0]
        merged_boundary_2229 = transform(to_state_plane, merged_boundary_3857)
        buffered_boundary_3857 = merged_boundary_3857.buffer(buffer_feet * 0.3048)

        site_sqft = merged_boundary_2229.area
        site_acreage = site_sqft / 43560.0

        tile_boxes = create_subtile_boxes(buffered_boundary_3857.bounds, tile_size_meters=60.0)
        total_tiles = len(tile_boxes)
        jobs[job_id]["total_tiles"] = total_tiles

        grid_size = 2.5
        voxel_dict = {}

        for idx, box_bounds in enumerate(tile_boxes):
            b_minx, b_miny, b_maxx, b_maxy = box_bounds
            jobs[job_id]["completed_tiles"] = idx + 1
            jobs[job_id]["message"] = f"Streaming LiDAR data tile {idx + 1} of {total_tiles}..."

            pdal_pipeline = {
                "pipeline": [
                    {
                        "type": "readers.ept",
                        "filename": "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_CA_LosAngeles_2016_LAS_2018/ept.json",
                        "bounds": f"([{b_minx}, {b_maxx}], [{b_miny}, {b_maxy}])"
                    },
                    {
                        "type": "filters.range",
                        "limits": "Classification[2:2], Classification[6:6]"
                    }
                ]
            }

            try:
                pipeline = pdal.Pipeline(json.dumps(pdal_pipeline))
                pipeline.execute()
                arrays = pipeline.arrays
                if not arrays or len(arrays[0]) == 0:
                    continue

                raw_data = arrays[0]

                for row in raw_data:
                    x_3857 = float(row["X"])
                    y_3857 = float(row["Y"])
                    z_feet = float(row["Z"]) * 3.2808398950131233

                    x_ft, y_ft = to_state_plane(x_3857, y_3857)
                    cell_key = (int(x_ft // grid_size), int(y_ft // grid_size))

                    if cell_key not in voxel_dict:
                        inside = bool(merged_boundary_2229.contains(Point(x_ft, y_ft)))
                        cls_val = int(row["Classification"]) if "Classification" in row.dtype.names else 2
                        voxel_dict[cell_key] = {
                            "x": x_ft, "y": y_ft, "z": z_feet, 
                            "inside": inside, "cls": cls_val
                        }
            except Exception:
                continue

            await asyncio.sleep(0)

        raw_pts = list(voxel_dict.values())
        if not raw_pts:
            raise Exception("No ground or surface points found in property region.")

        # 1. Adaptive Downsampling (Skip on small lots; Resample on 10+ / 76+ acres)
        jobs[job_id]["message"] = "Optimizing topographic point density..."
        pts = adaptive_process_points(raw_pts, lot_area_sqft=site_sqft)

        CACHE["boundary_2229"] = merged_boundary_2229
        b_coords = []
        geoms = [merged_boundary_2229] if isinstance(merged_boundary_2229, Polygon) else merged_boundary_2229.geoms
        for geom in geoms:
            coords_list = list(geom.exterior.coords)
            b_coords.append([{"x": float(c[0]), "y": float(c[1])} for c in coords_list])
        CACHE["boundary_coords"] = b_coords

        # 2. Compute TIN Mesh Topology off the main event loop
        jobs[job_id]["triangulation_progress"] = 50
        jobs[job_id]["message"] = "Building clipped Delaunay spatial TIN mesh..."
        await asyncio.sleep(0)

        loop = asyncio.get_running_loop()
        coords_arr, final_triangles, pts = await loop.run_in_executor(
            executor, precalculate_tin, pts, merged_boundary_2229, site_acreage
        )

        CACHE["raw_points"] = pts
        CACHE["triangles"] = final_triangles

        # 3. Vectorized Contouring Execution
        jobs[job_id]["triangulation_progress"] = 80
        jobs[job_id]["message"] = "Extracting vectorized site contours..."
        await asyncio.sleep(0)

        contours_1ft = calculate_contours_vectorized(coords_arr, final_triangles, interval=1.0)
        contours_5ft = calculate_contours_vectorized(coords_arr, final_triangles, interval=5.0)
        contours_10ft = calculate_contours_vectorized(coords_arr, final_triangles, interval=10.0)

        # Clip contour lines to property boundary line
        contours_1ft = clip_contours_to_boundary(contours_1ft, merged_boundary_2229)
        contours_5ft = clip_contours_to_boundary(contours_5ft, merged_boundary_2229)
        contours_10ft = clip_contours_to_boundary(contours_10ft, merged_boundary_2229)

        # Prepare Three.js visualization offset coordinates
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        zs = [p["z"] for p in pts]

        cx_center = float(sum(xs) / len(xs))
        cy_center = float(sum(ys) / len(ys))
        min_z = float(min(zs))

        three_points = [
            {
                "x": float(p["x"] - cx_center),
                "y": float(p["z"] - min_z),
                "z": float(-(p["y"] - cy_center)),
                "inside": p["inside"],
                "cls": p.get("cls", 2)
            }
            for p in pts
        ]

        three_boundary = [
            [
                {
                    "x": float(pt["x"] - cx_center),
                    "z": float(-(pt["y"] - cy_center))
                }
                for pt in ring
            ]
            for ring in b_coords
        ]

        result_payload = {
            "point_count": len(three_points),
            "points": three_points,
            "triangles": final_triangles,
            "boundary": three_boundary,
            "zoning_info": zoning_info_list,
            "center_offset": {"cx": cx_center, "cy": cy_center, "min_z": min_z},
            "contours": {
                "1": contours_1ft,
                "5": contours_5ft,
                "10": contours_10ft
            }
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Processing completed successfully."
        jobs[job_id]["result"] = result_payload

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# --- API Endpoints ---

@app.get("/")
async def serve_index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return Response(content="<h1>LA County 3D Topo Server Running</h1>", media_type="text/html")


@app.post("/api/get-merged-topo")
async def get_merged_topo_async(payload: dict, background_tasks: BackgroundTasks):
    apns = payload.get("apns", ["4464013013", "4464013012"])
    buffer_feet = payload.get("buffer_feet", 25.0)

    if not apns:
        raise HTTPException(status_code=400, detail="No APNs provided.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "completed_tiles": 0,
        "total_tiles": 0,
        "triangulation_progress": 0,
        "message": "Job initiated...",
        "result": None,
        "error": None
    }

    background_tasks.add_task(execute_topo_processing, job_id, apns, buffer_feet)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return Response(
        content=json.dumps(jobs[job_id]), 
        media_type="application/json"
    )


# --- UPDATED CIVIL ENGINEERING & SURVEYING DOWNLOAD ROUTES ---

@app.get("/api/download/surface")
async def download_surface(trimmed: bool = Query(True), format: str = Query("landxml")):
    """
    Exports 3D TIN surfaces in industry-standard civil formats (LandXML, HEC-RAS GeoTIFF, DXF, OBJ).
    """
    if not CACHE["raw_points"] or not CACHE["triangles"]:
        raise HTTPException(status_code=400, detail="No surface data available. Generate topo first.")

    pts = CACHE["raw_points"]
    triangles = CACHE["triangles"]
    fmt = format.lower().strip()

    # 1. LandXML (.xml) — Civil 3D, MicroStation, Carlson, OpenRoads
    if fmt in ["landxml", "xml"]:
        landxml = ET.Element('LandXML', {
            'version': '1.2',
            'xmlns': 'http://www.landxml.org/schema/LandXML-1.2'
        })
        surfaces = ET.SubElement(landxml, 'Surfaces')
        surface = ET.SubElement(surfaces, 'Surface', {'name': 'LiDAR_TIN_Surface'})
        definition = ET.SubElement(surface, 'Definition', {'surfType': 'TIN'})
        pnts = ET.SubElement(definition, 'Pnts')

        vertex_map = {}
        valid_idx = 1
        for idx, p in enumerate(pts):
            if trimmed and not p["inside"]:
                continue
            vertex_map[idx] = valid_idx
            # LandXML standard point ordering: Northing (Y), Easting (X), Elevation (Z)
            p_elem = ET.SubElement(pnts, 'P', {'id': str(valid_idx)})
            p_elem.text = f"{p['y']:.4f} {p['x']:.4f} {p['z']:.4f}"
            valid_idx += 1

        faces = ET.SubElement(definition, 'Faces')
        for t in triangles:
            if trimmed and not t["inside"]:
                continue
            i1, i2, i3 = t["indices"]
            if i1 in vertex_map and i2 in vertex_map and i3 in vertex_map:
                f_elem = ET.SubElement(faces, 'F')
                f_elem.text = f"{vertex_map[i1]} {vertex_map[i2]} {vertex_map[i3]}"

        xml_str = minidom.parseString(ET.tostring(landxml)).toprettyxml(indent="  ")
        filename = f"topo_surface_{'trimmed' if trimmed else 'full'}.xml"
        return Response(content=xml_str, media_type="application/xml", headers={"Content-Disposition": f"attachment; filename={filename}"})

    # 2. HEC-RAS / GIS DEM GeoTIFF (.tif) — HEC-RAS RAS Mapper, QGIS, ArcGIS
    elif fmt in ["geotiff", "tif", "hec-ras", "hecras"]:
        valid_pts = [p for p in pts if (p["inside"] if trimmed else True)]
        if not valid_pts:
            raise HTTPException(status_code=400, detail="No valid points within export bounds.")

        coords_arr = np.array([[p["x"], p["y"], p["z"]] for p in valid_pts])
        x_min, x_max = coords_arr[:, 0].min(), coords_arr[:, 0].max()
        y_min, y_max = coords_arr[:, 1].min(), coords_arr[:, 1].max()

        resolution = 1.0  # 1-foot grid cell size
        width = max(1, int(np.ceil((x_max - x_min) / resolution)))
        height = max(1, int(np.ceil((y_max - y_min) / resolution)))

        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, width),
            np.linspace(y_max, y_min, height)
        )

        grid_z = griddata(coords_arr[:, :2], coords_arr[:, 2], (grid_x, grid_y), method='linear')
        transform_gtif = from_origin(x_min, y_max, resolution, resolution)

        memfile = io.BytesIO()
        with rasterio.open(
            memfile, 'w', driver='GTiff',
            height=height, width=width, count=1,
            dtype=rasterio.float32, crs="EPSG:2229",  # CA State Plane Zone 5 (US Feet)
            transform=transform_gtif, nodata=-9999
        ) as dst:
            dst.write(np.nan_to_num(grid_z, nan=-9999).astype(np.float32), 1)

        memfile.seek(0)
        filename = f"hecras_dtm_{'trimmed' if trimmed else 'full'}.tif"
        return StreamingResponse(memfile, media_type="image/tiff", headers={"Content-Disposition": f"attachment; filename={filename}"})

    # 3. 3D DXF Face Mesh (.dxf) — AutoCAD, MicroStation
    elif fmt == "dxf":
        doc = ezdxf.new(dxfversion="R2000")
        msp = doc.modelspace()

        for t in triangles:
            if trimmed and not t["inside"]:
                continue
            p1, p2, p3 = pts[t["indices"][0]], pts[t["indices"][1]], pts[t["indices"][2]]
            msp.add_3dface([
                (p1["x"], p1["y"], p1["z"]),
                (p2["x"], p2["y"], p2["z"]),
                (p3["x"], p3["y"], p3["z"]),
                (p3["x"], p3["y"], p3["z"])
            ])

        out_stream = io.StringIO()
        doc.write(out_stream)
        filename = f"topo_mesh_{'trimmed' if trimmed else 'full'}.dxf"
        return Response(content=out_stream.getvalue(), media_type="application/dxf", headers={"Content-Disposition": f"attachment; filename={filename}"})

    # 4. Wavefront OBJ (.obj) — Blender, Rhino, 3ds Max
    elif fmt == "obj":
        cx = float(sum(p["x"] for p in pts) / len(pts))
        cy = float(sum(p["y"] for p in pts) / len(pts))
        min_z = float(min(p["z"] for p in pts))

        output = ["# 3D Topo Surface Mesh\n"]
        vertex_map = {}
        valid_pts = []

        for idx, p in enumerate(pts):
            if trimmed and not p["inside"]:
                continue
            vertex_map[idx] = len(valid_pts) + 1
            valid_pts.append(p)
            output.append(f"v {p['x'] - cx:.4f} {p['z'] - min_z:.4f} {-(p['y'] - cy):.4f}\n")

        for t in triangles:
            if trimmed and not t["inside"]:
                continue
            i1, i2, i3 = t["indices"]
            if i1 in vertex_map and i2 in vertex_map and i3 in vertex_map:
                output.append(f"f {vertex_map[i1]} {vertex_map[i2]} {vertex_map[i3]}\n")

        filename = f"topo_mesh_{'trimmed' if trimmed else 'full'}.obj"
        return Response(content="".join(output), media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={filename}"})

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported surface format: {format}")


@app.get("/api/download/contours")
async def download_contours(interval: float = Query(1.0), format: str = Query("dxf")):
    """
    Exports 3D elevation contours to DXF polylines or GIS GeoJSON vectors.
    """
    if not CACHE["raw_points"] or not CACHE["triangles"]:
        raise HTTPException(status_code=400, detail="No surface data available. Generate topo first.")

    coords_arr = np.array([[p["x"], p["y"], p["z"]] for p in CACHE["raw_points"]], dtype=np.float64)
    contours = calculate_contours_vectorized(coords_arr, CACHE["triangles"], interval=interval)
    
    if CACHE.get("boundary_2229"):
        contours = clip_contours_to_boundary(contours, CACHE["boundary_2229"])

    fmt = format.lower().strip()

    # 1. 3D Contour DXF PolyLines (.dxf) — AutoCAD, Civil 3D
    if fmt == "dxf":
        doc = ezdxf.new(dxfversion="R2000")
        msp = doc.modelspace()

        for c_group in contours:
            elev = c_group["elevation"]
            is_index = c_group["is_index"]
            layer_name = "C-TOPO-INDEX" if is_index else "C-TOPO-INTER"
            color = 1 if is_index else 2  # Red for index, Yellow for intermediate

            for seg in c_group["segments"]:
                p1, p2 = seg["pts"][0], seg["pts"][1]
                msp.add_polyline3d(
                    [(p1["x"], p1["y"], elev), (p2["x"], p2["y"], elev)],
                    dxfattribs={"layer": layer_name, "color": color}
                )

        out_stream = io.StringIO()
        doc.write(out_stream)
        filename = f"contours_{interval}ft.dxf"
        return Response(content=out_stream.getvalue(), media_type="application/dxf", headers={"Content-Disposition": f"attachment; filename={filename}"})

    # 2. GIS GeoJSON LineStrings (.geojson) — QGIS, ArcGIS Pro
    elif fmt in ["geojson", "json"]:
        features = []
        for c_group in contours:
            elev = c_group["elevation"]
            for seg in c_group["segments"]:
                p1, p2 = seg["pts"][0], seg["pts"][1]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[p1["x"], p1["y"], elev], [p2["x"], p2["y"], elev]]
                    },
                    "properties": {
                        "ELEVATION": elev,
                        "IS_INDEX": c_group["is_index"]
                    }
                })

        geojson_payload = {"type": "FeatureCollection", "features": features}
        filename = f"contours_{interval}ft.geojson"
        return Response(content=json.dumps(geojson_payload), media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported contour format: {format}")


@app.get("/api/download/points")
async def download_points(trimmed: bool = Query(False), format: str = Query("dxf")):
    """
    Exports point cloud data to DXF points or CSV point lists.
    """
    if not CACHE["raw_points"]:
        raise HTTPException(status_code=400, detail="No point cloud data available to export.")

    pts = CACHE["raw_points"]
    fmt = format.lower().strip()

    if fmt == "dxf":
        output = ["0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"]

        for p in pts:
            if trimmed and not p["inside"]:
                continue
            output.append("0\nPOINT\n8\nTOPOGRAPHY\n")
            output.append(f"10\n{p['x']:.4f}\n")
            output.append(f"20\n{p['y']:.4f}\n")
            output.append(f"30\n{p['z']:.4f}\n")

        output.append("0\nENDSEC\n0\nEOF\n")
        content = "".join(output)
        filename = f"topo_points_{'trimmed' if trimmed else 'untrimmed'}.dxf"

        return Response(
            content=content,
            media_type="application/dxf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif fmt == "csv":
        output = ["x,y,z,inside,cls\n"]
        for p in pts:
            if trimmed and not p["inside"]:
                continue
            output.append(f"{p['x']:.4f},{p['y']:.4f},{p['z']:.4f},{p['inside']},{p.get('cls', 2)}\n")
        content = "".join(output)
        filename = f"topo_points_{'trimmed' if trimmed else 'untrimmed'}.csv"

        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported point export format: {format}")