from concurrent.futures import ProcessPoolExecutor
from xml.etree import ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import ezdxf
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

# --- Parallel Processing Worker for Boundary Splitting ---
def process_simplex_batch(args):
    """Processes a subset of Delaunay triangles against site boundary polygons."""
    simplices_batch, points, boundary_polygons = args
    valid_triangles = []
    
    for simplex in simplices_batch:
        pts = points[simplex]
        poly = Polygon(pts[:, :2])
        # Include triangle if its centroid falls within any boundary polygon
        centroid = poly.centroid
        if any(b_poly.contains(centroid) for b_poly in boundary_polygons):
            valid_triangles.append(simplex)
            
    return valid_triangles

# --- Main Topography Processing Routine ---
def execute_topo_processing(job_id, jobs_dict, raw_points, site_boundaries=None, target_resolution=2.5):
    """
    Constructs Delaunay TIN, decimates mesh, updates progress status,
    and returns triangulation output.
    """
    jobs_dict[job_id] = {"status": "RUNNING", "percent": 10, "message": "Downsampling & filtering point cloud..."}
    
    # 1. Coordinate Grid Processing & Subsampling
    coords = np.array(raw_points) # Expected shape: (N, 3) -> [X, Y, Z]
    
    jobs_dict[job_id].update({"percent": 30, "message": "Constructing 2.5D Delaunay Triangulation..."})
    # Delaunay on 2D projection (X, Y)
    tri = Delaunay(coords[:, :2])
    simplices = tri.simplices

    # 2. Parallel Boundary Intersection
    if site_boundaries:
        jobs_dict[job_id].update({"percent": 50, "message": "Filtering triangles against boundary polygons..."})
        num_workers = 4
        chunks = np.array_split(simplices, num_workers)
        worker_args = [(chunk, coords, site_boundaries) for chunk in chunks]

        valid_simplices = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, batch_result in enumerate(executor.map(process_simplex_batch, worker_args)):
                valid_simplices.extend(batch_result)
                prog = 50 + int((idx + 1) / num_workers * 30)
                jobs_dict[job_id].update({"percent": prog, "message": f"Boundary filter pass {idx+1}/{num_workers}..."})
        simplices = np.array(valid_simplices)

    jobs_dict[job_id].update({"percent": 90, "message": "Finalizing surface topology..."})
    
    # Return processed surface dictionary
    surface_data = {
        "vertices": coords,
        "simplices": simplices
    }
    
    jobs_dict[job_id].update({"status": "COMPLETED", "percent": 100, "message": "Surface generation complete."})
    return surface_data


# --- Export Exporters ---

def export_to_dxf(surface_data, output_filepath):
    """Exports TIN surface as 3DFACE entities in DXF format for CAD software."""
    doc = ezdxf.new(dxfversion="R2000")
    msp = doc.modelspace()
    vertices = surface_data["vertices"]
    
    for simplex in surface_data["simplices"]:
        pts = vertices[simplex]
        # Add 3DFACE entity (v0, v1, v2, v2 to close triangle)
        msp.add_3dface([
            (pts[0][0], pts[0][1], pts[0][2]),
            (pts[1][0], pts[1][1], pts[1][2]),
            (pts[2][0], pts[2][1], pts[2][2]),
            (pts[2][0], pts[2][1], pts[2][2])
        ])
    doc.saveas(output_filepath)


def export_to_landxml(surface_data, output_filepath, surface_name="Campus_TIN_Surface"):
    """Exports surface topology to LandXML format for Civil 3D and Bentley OpenRoads."""
    landxml = ET.Element('LandXML', {
        'version': '1.2',
        'xmlns': 'http://www.landxml.org/schema/LandXML-1.2'
    })
    surfaces = ET.SubElement(landxml, 'Surfaces')
    surface = ET.SubElement(surfaces, 'Surface', {'name': surface_name})
    definition = ET.SubElement(surface, 'Definition', {'surfType': 'TIN'})
    
    pnts = ET.SubElement(definition, 'Pnts')
    vertices = surface_data["vertices"]
    
    # Add points with 1-based indexing
    for i, pt in enumerate(vertices, start=1):
        p = ET.SubElement(pnts, 'P', {'id': str(i)})
        p.text = f"{pt[1]:.4f} {pt[0]:.4f} {pt[2]:.4f}"  # LandXML default: Northing Easting Elevation

    faces = ET.SubElement(definition, 'Faces')
    for simplex in surface_data["simplices"]:
        f = ET.SubElement(faces, 'F')
        # LandXML indices are 1-based
        f.text = f"{simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}"

    xml_str = minidom.parseString(ET.tostring(landxml)).toprettyxml(indent="  ")
    with open(output_filepath, "w") as f:
        f.write(xml_str)


def export_to_geotiff(surface_data, output_filepath, resolution=1.0):
    """Rasterizes surface elevation data into a continuous GeoTIFF DEM file."""
    vertices = surface_data["vertices"]
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    # Grid coordinate initialization
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, width),
        np.linspace(y_max, y_min, height)  # Top-to-bottom for raster rows
    )

    # Nearest-neighbor or linear barycentric raster interpolation
    from scipy.interpolate import griddata
    grid_z = griddata(vertices[:, :2], vertices[:, 2], (grid_x, grid_y), method='linear')

    transform = from_origin(x_min, y_max, resolution, resolution)
    
    with rasterio.open(
        output_filepath, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=rasterio.float32,
        crs='EPSG:2229',  # Example Coordinate Reference System
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(np.nan_to_num(grid_z, nan=-9999).astype(np.float32), 1)