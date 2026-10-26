"""Audited Geo2SigMap API integration for LiDAR heights and aligned DEM terrain.

The installed CLI disables LiDAR height calibration. Its DEM exporter also
moves the mesh minimum to zero, independently of the building origin. This
adapter leaves the installed package unchanged and records the corrections.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def strip_ply_obj_info(path):
    """Drop optional VTK header records rejected by Mitsuba; preserve mesh bytes."""
    path = Path(path)
    data = path.read_bytes()
    end = data.find(b'end_header\n')
    if end < 0:
        raise ValueError(f'Missing PLY header terminator: {path}')
    end += len(b'end_header\n')
    header = b''.join(line for line in data[:end].splitlines(keepends=True)
                      if not line.startswith(b'obj_info '))
    if header != data[:end]:
        path.write_bytes(header + data[end:])


def generate_scene(box, target_directory, settings, lidar_terrain, dem_terrain):
    import numpy as np
    import pyvista as pv
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from scene_generation import core
    from scene_generation import USGS_LiDAR_HAG as lidar
    from scene_generation.dem_utils import clip_reproject_dem_to_wgs84_utm
    from scene_generation.dem_gpkg_data import get_fesm_paths
    from scene_generation.USGS1mLocator_parallel import USGS1mLocator

    out = Path(target_directory).resolve()
    seed = int(settings.get('seed', 71))
    random.seed(seed)
    np.random.seed(seed)
    points = [(box['min_lon'], box['min_lat']), (box['min_lon'], box['max_lat']),
              (box['max_lon'], box['max_lat']), (box['max_lon'], box['min_lat'])]
    crs = core.get_utm_epsg_code_from_gps(*points[0])
    forward = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    center = Polygon([forward.transform(*p) for p in points]).envelope.centroid
    origin = np.array([center.x, center.y])
    audit = {'source': 'Geo2SigMap', 'seed': seed,
             'source_module': str(Path(core.__file__).resolve()),
             'source_module_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
             'horizontal_crs': crs.to_string(), 'horizontal_origin_m': origin.tolist(),
             'height_method': 'mean of up to 30 valid LiDAR HAG samples per footprint',
             'buildings': [], 'terrain': None}
    (out/'source_inputs').mkdir(exist_ok=True)
    original_hag = lidar.generate_hag
    original_pipeline = lidar.pdal.Pipeline
    original_features = core.ox.features.features_from_bbox
    original_height = core.random_building_height
    original_dem = core.generate_terrain_mesh_dem
    original_cache = (core.ox.settings.use_cache, core.ox.settings.cache_folder)
    handler = None

    class IsolatedPipeline:
        # PDAL's conda GDAL/PROJ conflicts with wheel-bundled rasterio/pyproj
        # libraries in this interpreter. Execute the identical pipeline in the
        # environment's standalone CLI and preserve its exact JSON input.
        def __init__(self, payload):
            self.payload = payload
        def execute(self):
            path = out/'source_inputs'/'lidar_pipeline.json'
            path.write_text(self.payload)
            env = dict(os.environ, PROJ_DATA=str(Path(sys.prefix)/'share/proj'))
            subprocess.run([str(Path(sys.prefix)/'bin/pdal'), 'pipeline', str(path),
                            '--metadata', str(out/'source_inputs'/'lidar_metadata.json')],
                           env=env, check=True)

    def audited_hag(polygon, directory, CRS):
        original_hag(polygon, directory, CRS)
        # Upstream produces pointcloud_test.laz but core expects test_hag.laz.
        cloud = Path(directory)/'pointcloud_test.laz'
        expected = Path(directory)/'test_hag.laz'
        if cloud.exists() and not expected.exists():
            os.link(cloud, expected)
        if not expected.is_file() or not (Path(directory)/'test_hag.tif').is_file():
            raise RuntimeError('LiDAR generation did not produce the required LAZ and HAG raster')

    def audited_features(*args, **kwargs):
        core.ox.settings.use_cache = True
        core.ox.settings.cache_folder = str(out/'source_inputs'/'osm_cache')
        gdf = original_features(*args, **kwargs)
        (out/'source_inputs'/'osm_buildings.geojson').write_text(gdf.to_json())
        return gdf

    def calibrated_height(building, polygon, *args, **kwargs):
        nonlocal handler
        if handler is None:
            handler = core.GeoTIFFHandler(str(out/'test_hag.tif'))
        heights = []
        for point in core.generate_random_points(polygon, 30):
            values = np.asarray(handler.query(inverse.transform(point.x, point.y), False)).ravel()
            heights.extend(float(v) for v in values if np.isfinite(v) and v > 2 and v != -9999)
        if heights:
            height, method = float(np.mean(heights)), 'lidar_hag'
        else:
            height, method = float(original_height(building, polygon, *args, **kwargs)), 'osm_or_seeded_fallback'
        audit['buildings'].append({'bounds_projected_m': list(polygon.bounds),
                                  'height_m': height, 'source': method,
                                  'valid_lidar_samples': len(heights)})
        return height

    def aligned_dem(aoi, ply_path):
        # Use Geo2SigMap's DEM selection/download/reprojection, but preserve the
        # same XY origin as its buildings rather than upstream mesh.rezero().
        paths = get_fesm_paths()
        locator = USGS1mLocator(paths[1], paths[0], debug=False)
        links = locator.get_links(aoi)
        if not links:
            raise RuntimeError('No DEM coverage for the scene')
        result = clip_reproject_dem_to_wgs84_utm(
            aoi, links, mode='multi_zone' if len(links)>1 else 'single_zone',
            mosaic=True, out_prefix=Path(ply_path).parent/'dem')
        tif = Path(ply_path).parent/'dem.tif'
        with rasterio.open(tif) as ds:
            values = ds.read(1, masked=True)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
            rows, cols = np.indices(values.shape)
            xs, ys = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel(), offset='center')
            xs, ys = np.asarray(xs), np.asarray(ys)
            to_scene = Transformer.from_crs(ds.crs, crs, always_xy=True)
            xs, ys = to_scene.transform(xs, ys)
            z_origin = float(values[valid].mean())
            vertices = np.column_stack([xs-origin[0], ys-origin[1], values.filled(z_origin).ravel()-z_origin])
            h, w = values.shape
            a = (rows[:-1,:-1]*w+cols[:-1,:-1]).ravel()
            faces = np.concatenate([np.column_stack([a,a+w,a+w+1]),
                                    np.column_stack([a,a+w+1,a+1])])
            faces = faces[valid.ravel()[faces].all(axis=1)]
            if not len(faces):
                raise RuntimeError('DEM contains no valid terrain triangles')
            mesh = pv.PolyData(vertices, np.column_stack([np.full(len(faces),3),faces]).ravel())
            mesh.clean().save(ply_path)
            strip_ply_obj_info(ply_path)
            audit['terrain'] = {'source': 'USGS_1m_DEM', 'selection': result,
                'requested_links': list(links), 'raster_crs': str(ds.crs),
                'raster_resolution_m': list(ds.res), 'raster_bounds': list(ds.bounds),
                'valid_fraction': float(valid.mean()), 'vertical_origin_m': z_origin,
                'vertical_reference': 'DEM elevation minus recorded valid-cell mean; source datum retained in inputs',
                'horizontal_alignment': 'raster pixel centers projected to building CRS and origin',
                'bounds_local_m': list(mesh.bounds)}

    lidar.pdal.Pipeline = IsolatedPipeline
    lidar.generate_hag = audited_hag
    core.ox.features.features_from_bbox = audited_features
    core.random_building_height = calibrated_height
    core.generate_terrain_mesh_dem = aligned_dem
    old_cwd = os.getcwd()
    try:
        os.chdir(out)
        core.Scene()(points=points, data_dir=str(out), hag_tiff_path=str(out/'test_hag.tif'),
            # Calibration is performed by the audited height hook above. This
            # avoids upstream's unbound-variable error in its fallback branch.
            lidar_calibration=False, generate_building_map=True,
            lidar_terrain=lidar_terrain, dem_terrain=dem_terrain,
            ground_scale=float(settings.get('ground_scale',1.5)),
            ground_material_type=settings.get('ground_material','mat-itu_medium_dry_ground'),
            wall_material_type=settings.get('wall_material','mat-itu_concrete'),
            rooftop_material_type=settings.get('rooftop_material','mat-itu_concrete'))
        if not (out/'scene.xml').is_file():
            raise RuntimeError('Geo2SigMap did not produce scene.xml')
        audit['lidar_building_count'] = sum(b['source']=='lidar_hag' for b in audit['buildings'])
        audit['fallback_building_count'] = len(audit['buildings'])-audit['lidar_building_count']
        if not audit['lidar_building_count']:
            raise RuntimeError('No building heights were derived from LiDAR')
        return audit
    finally:
        lidar.generate_hag = original_hag
        lidar.pdal.Pipeline = original_pipeline
        core.ox.features.features_from_bbox = original_features
        core.random_building_height = original_height
        core.generate_terrain_mesh_dem = original_dem
        core.ox.settings.use_cache, core.ox.settings.cache_folder = original_cache
        os.chdir(old_cwd)
        if handler is not None:
            handler.src.close()
        (out/'generation_sources.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
