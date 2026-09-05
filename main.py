import os
import sys
import json 
import io
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Locate active conda prefix dynamically
conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
proj_dir = os.path.join(conda_prefix, "share", "proj")

# Export PROJ pathing prior to loading C-extensions
os.environ["PROJ_DATA"] = proj_dir
os.environ["PROJ_LIB"] = proj_dir
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["PDAL_LOGLEVEL"] = "0"

# Explicitly notify pyproj before initializing CRS transformers
import pyproj
pyproj.datadir.set_data_dir(proj_dir)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
from shapely.geometry import shape, Point, Polygon
from shapely.ops import unary_union, split, transform
from pyproj import Transformer
import pdal
import numpy as np
from scipy.spatial import Delaunay

os.environ["PDAL_LOGLEVEL"] = "0"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

LA_COUNTY_PARCEL_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/0/query"
LA_COUNTY_ZONING_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/1/query"

# Coordinate Transformer: EPSG:3857 (Web Mercator) -> EPSG:2229 (CA State Plane Zone 5 US Feet)
to_state_plane = Transformer.from_crs("EPSG:3857", "EPSG:2229", always_xy=True).transform

CACHE = {
    "raw_points": [],
    "triangles": [],
    "boundary_2229": None,
    "boundary_coords": []
}

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

def interpolate_z_barycentric(p1, p2, p3, x, y):
    x1, y1, z1 = p1["x"], p1["y"], p1["z"]
    x2, y2, z2 = p2["x"], p2["y"], p2["z"]
    x3, y3, z3 = p3["x"], p3["y"], p3["z"]

    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-9:
        return (z1 + z2 + z3) / 3.0

    w1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    w2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    w3 = 1.0 - w1 - w2

    return float(w1*z1 + w2*z2 + w3*z3)

def calculate_contours(pts, triangles, interval=5.0):
    if not pts or not triangles:
        return []

    zs = [p["z"] for p in pts]
    min_z = np.floor(min(zs) / interval) * interval
    max_z = np.ceil(max(zs) / interval) * interval
    levels = np.arange(min_z, max_z + interval, interval)

    contours = []
    for z_level in levels:
        segments = []
        for t in triangles:
            idx = t["indices"]
            p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            v_zs = [p1["z"], p2["z"], p3["z"]]

            if min(v_zs) <= z_level <= max(v_zs):
                pts_tri = [p1, p2, p3]
                intersect_pts = []

                for i in range(3):
                    a = pts_tri[i]
                    b = pts_tri[(i + 1) % 3]

                    if (a["z"] <= z_level <= b["z"]) or (b["z"] <= z_level <= a["z"]):
                        if abs(b["z"] - a["z"]) > 1e-6:
                            t_factor = (z_level - a["z"]) / (b["z"] - a["z"])
                            ix = float(a["x"] + t_factor * (b["x"] - a["x"]))
                            iy = float(a["y"] + t_factor * (b["y"] - a["y"]))
                            
                            inside = bool(CACHE["boundary_2229"].contains(Point(ix, iy)))
                            intersect_pts.append({"x": ix, "y": iy, "z": float(z_level), "inside": inside})

                if len(intersect_pts) == 2:
                    is_seg_inside = intersect_pts[0]["inside"] and intersect_pts[1]["inside"]
                    segments.append({
                        "pts": intersect_pts,
                        "inside": is_seg_inside
                    })

        if segments:
            is_index = bool((round(z_level, 2) % (interval * 5)) == 0)
            contours.append({
                "elevation": round(float(z_level), 2),
                "is_index": is_index,
                "segments": segments
            })

    return contours

@app.post("/api/get-merged-topo")
async def get_merged_topo(payload: dict):
    try:
        apns = payload.get("apns", ["4464013013", "4464013012"])
        buffer_feet = payload.get("buffer_feet", 25.0)

        if not apns:
            raise HTTPException(status_code=400, detail="No APNs provided.")

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
            res = requests.get(LA_COUNTY_PARCEL_URL, params=params, timeout=15.0).json()

            if not res.get('features'):
                raise HTTPException(status_code=404, detail=f"APN {apn} not found.")

            feature = res['features'][0]
            geom_3857 = shape(feature['geometry'])
            parcel_polygons_3857.append(geom_3857)

            props = feature.get('properties', {})
            
            # Fetch Zoning Info
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

        # Convert boundary to EPSG:2229 (CA State Plane Zone 5 Feet)
        merged_boundary_2229 = transform(to_state_plane, merged_boundary_3857)

        # Buffer boundary in true feet
        buffered_boundary_3857 = merged_boundary_3857.buffer(buffer_feet * 0.3048)
        minx, miny, maxx, maxy = buffered_boundary_3857.bounds
        wkt_crop = buffered_boundary_3857.wkt

        # PDAL query using USGS EPT LiDAR
        pdal_pipeline = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_CA_LosAngeles_2016_LAS_2018/ept.json",
                    "bounds": f"([{minx}, {maxx}], [{miny}, {maxy}])"
                },
                {
                    "type": "filters.crop",
                    "polygon": wkt_crop
                },
                {
                    "type": "filters.range",
                    "limits": "Classification[2:2]"
                }
            ]
        }

        pipeline = pdal.Pipeline(json.dumps(pdal_pipeline))
        pipeline.execute()

        arrays = pipeline.arrays
        if not arrays or len(arrays[0]) == 0:
            raise HTTPException(status_code=404, detail="No ground points found.")

        raw_data = arrays[0]

        # --- 2.0 ft Grid Binning in EPSG:2229 State Plane Feet ---
        grid_size = 2.0
        voxel_dict = {}

        for row in raw_data:
            x_3857 = float(row["X"])
            y_3857 = float(row["Y"])
            z_feet = float(row["Z"]) * 3.2808398950131233  # Meters to US Survey Feet

            # Reproject (X, Y) from Web Mercator to State Plane Feet
            x_ft, y_ft = to_state_plane(x_3857, y_3857)

            cell_key = (int(x_ft // grid_size), int(y_ft // grid_size))

            if cell_key not in voxel_dict:
                inside = bool(merged_boundary_2229.contains(Point(x_ft, y_ft)))
                voxel_dict[cell_key] = {"x": x_ft, "y": y_ft, "z": z_feet, "inside": inside}

        pts = list(voxel_dict.values())

        CACHE["boundary_2229"] = merged_boundary_2229

        b_coords = []
        geoms = [merged_boundary_2229] if isinstance(merged_boundary_2229, Polygon) else merged_boundary_2229.geoms
        for geom in geoms:
            coords = list(geom.exterior.coords)
            b_coords.append([{"x": float(c[0]), "y": float(c[1])} for c in coords])
        CACHE["boundary_coords"] = b_coords

        coords_2d = np.array([[p["x"], p["y"]] for p in pts])
        tri = Delaunay(coords_2d)

        final_pts = list(pts)
        final_triangles = []
        boundary_line = merged_boundary_2229.boundary

        for simplex in tri.simplices:
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            tri_poly = Polygon([(p1["x"], p1["y"]), (p2["x"], p2["y"]), (p3["x"], p3["y"])])

            if not merged_boundary_2229.intersects(tri_poly):
                final_triangles.append({
                    "indices": [int(simplex[0]), int(simplex[1]), int(simplex[2])],
                    "inside": False
                })
                continue

            if merged_boundary_2229.contains(tri_poly):
                final_triangles.append({
                    "indices": [int(simplex[0]), int(simplex[1]), int(simplex[2])],
                    "inside": True
                })
                continue

            split_geom = split(tri_poly, boundary_line)
            
            for sub_poly in split_geom.geoms:
                sub_coords = list(sub_poly.exterior.coords)[:-1]
                if len(sub_coords) < 3:
                    continue

                sub_indices = []
                for cx, cy in sub_coords:
                    matched_idx = -1
                    for idx_check, pt_check in enumerate(final_pts):
                        if abs(pt_check["x"] - cx) < 1e-4 and abs(pt_check["y"] - cy) < 1e-4:
                            matched_idx = idx_check
                            break
                    
                    if matched_idx == -1:
                        interp_z = interpolate_z_barycentric(p1, p2, p3, cx, cy)
                        inside_check = bool(merged_boundary_2229.contains(Point(cx, cy)) or merged_boundary_2229.boundary.contains(Point(cx, cy)))
                        new_pt = {"x": float(cx), "y": float(cy), "z": interp_z, "inside": inside_check}
                        final_pts.append(new_pt)
                        matched_idx = len(final_pts) - 1

                    sub_indices.append(matched_idx)

                if len(sub_indices) == 3:
                    sub_cx = sum([final_pts[i]["x"] for i in sub_indices]) / 3.0
                    sub_cy = sum([final_pts[i]["y"] for i in sub_indices]) / 3.0
                    inside_sub = bool(merged_boundary_2229.contains(Point(sub_cx, sub_cy)))
                    final_triangles.append({"indices": sub_indices, "inside": inside_sub})
                elif len(sub_indices) > 3:
                    for k in range(1, len(sub_indices) - 1):
                        t_inds = [sub_indices[0], sub_indices[k], sub_indices[k+1]]
                        sub_cx = sum([final_pts[i]["x"] for i in t_inds]) / 3.0
                        sub_cy = sum([final_pts[i]["y"] for i in t_inds]) / 3.0
                        inside_sub = bool(merged_boundary_2229.contains(Point(sub_cx, sub_cy)))
                        final_triangles.append({"indices": t_inds, "inside": inside_sub})

        CACHE["raw_points"] = final_pts
        CACHE["triangles"] = final_triangles

        xs = [p["x"] for p in final_pts]
        ys = [p["y"] for p in final_pts]
        zs = [p["z"] for p in final_pts]

        cx_center = float(sum(xs) / len(xs))
        cy_center = float(sum(ys) / len(ys))
        min_z = float(min(zs))

        three_points = [
            {
                "x": float(p["x"] - cx_center),
                "y": float(p["z"] - min_z),
                "z": float(-(p["y"] - cy_center)),
                "inside": p["inside"]
            }
            for p in final_pts
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

        contours_1ft = calculate_contours(final_pts, final_triangles, interval=1.0)
        contours_5ft = calculate_contours(final_pts, final_triangles, interval=5.0)
        contours_10ft = calculate_contours(final_pts, final_triangles, interval=10.0)

        return {
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
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Processing Error: {str(e)}")

@app.get("/api/download/surface")
async def download_surface(trimmed: bool = Query(True), format: str = Query("obj")):
    if not CACHE["raw_points"] or not CACHE["triangles"]:
        raise HTTPException(status_code=400, detail="No topo data available to export. Generate topo first.")

    pts = CACHE["raw_points"]
    triangles = CACHE["triangles"]
    
    # Calculate centroid to offset geometry to origin (matching Three.js space)
    cx = float(sum(p["x"] for p in pts) / len(pts))
    cy = float(sum(p["y"] for p in pts) / len(pts))
    min_z = float(min(p["z"] for p in pts))

    output = []
    output.append("# LA County 3D Topo Surface Mesh Export\n")

    # Filter points/vertices if trimmed
    vertex_map = {}
    valid_pts = []
    
    for idx, p in enumerate(pts):
        if trimmed and not p["inside"]:
            continue
        new_idx = len(valid_pts) + 1
        vertex_map[idx] = new_idx
        valid_pts.append(p)
        # Export in local feet centered coordinates
        x_local = p["x"] - cx
        y_local = p["z"] - min_z
        z_local = -(p["y"] - cy)
        output.append(f"v {x_local:.4f} {y_local:.4f} {z_local:.4f}\n")

    # Export face indices
    for t in triangles:
        if trimmed and not t["inside"]:
            continue
        i1, i2, i3 = t["indices"]
        if i1 in vertex_map and i2 in vertex_map and i3 in vertex_map:
            output.append(f"f {vertex_map[i1]} {vertex_map[i2]} {vertex_map[i3]}\n")

    content = "".join(output)
    filename = f"topo_surface_{'trimmed' if trimmed else 'untrimmed'}.obj"

    return Response(
        content=content,
        media_type="application/text",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/download/points")
async def download_points(trimmed: bool = Query(False), format: str = Query("dxf")):
    if not CACHE["raw_points"]:
        raise HTTPException(status_code=400, detail="No point cloud data available to export.")

    pts = CACHE["raw_points"]

    output = []
    # Simple ASCII DXF Section Header for Points
    output.append("0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")

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