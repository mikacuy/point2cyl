import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))

### Global variables for the project ###

RAW_ROOT_DIR = "/orion/group/ReconstructionSubset/d7/"

### For renderer
g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'
g_azimuth_deg = -70
g_elevation_deg = 20
g_theta_deg = 0

g_zero_tol = 1.0e-6

small_area_threshold = 0.02

EXTRUSION_OPERATION_DICT = {"NewBodyFeatureOperation":0,
							"JoinFeatureOperation":0,
							"CutFeatureOperation":1,
							"IntersectFeatureOperation":2}