# Mikaela Uy (mikacuy@cs.stanford.edu)
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import sys
import importlib
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, '..','data_preprocessing'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

## For implicit
sys.path.append(os.path.join(BASE_DIR, 'IGR'))
from sampler import *
from network import *
from general import *
from plots import plot_surface_2d
from chamferdist import ChamferDistance
chamferDist = ChamferDistance()

from global_variables import *
from utils import *
from data_utils import *
from dataloader import AutodeskDataset_h5_sketches
from losses import *

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

import pickle
import trimesh

### For extent clustering
from sklearn.cluster import DBSCAN
from sklearn import metrics

from plyfile import PlyData, PlyElement

MAX_NUM_INSTANCES = 8

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/')

# parser.add_argument('--model_id', type=str, default='115629_40d61053_0000_1')
# parser.add_argument('--model_id', type=str, default='126491_c931419a_0003_1')
# parser.add_argument('--model_id', type=str, default='111773_ab926952_0000_1')
parser.add_argument('--model_id', type=str, default='55838_a1513314_0000_1')
parser.add_argument('--data_split', type=str, default='test')
parser.add_argument('--out_fname', type=str, default='test_sdf.ply')
parser.add_argument('--dump_dir', default= "dump_visu/", type=str)
parser.add_argument('--num_points', type=int, default=2048)

### For marching cubes
parser.add_argument('--resolution', type=int, default=512)
parser.add_argument('--range', type=float, default=1.5)
parser.add_argument('--level', type=float, default=0.0)

### Load network
parser.add_argument('--model', type=str, default='pointnet_extrusion', help='model name')

parser.add_argument("--logdir", default="./results/Point2Cyl/", help="path to the log directory", type=str)

parser.add_argument("--ckpt", default="model.pth", help="checkpoint", type=str)
parser.add_argument('--K', type=int,  default=8, help='Max number of extrusions')
parser.add_argument('--num_sk_point', type=int,  default=1024, help='Point Number [default: 2048]')

parser.add_argument('--pred_seg', action='store_false')
parser.add_argument('--pred_normal', action='store_false')
parser.add_argument('--pred_bb', action='store_false')
parser.add_argument('--pred_extrusion', action='store_false')
parser.add_argument('--pred_op', action='store_true')
parser.add_argument('--norm_eig', action='store_true')
parser.add_argument('--use_whole_pc', action='store_true')
parser.add_argument('--use_extrusion_axis_feat', action='store_true')

##Pre-trained implicit network
### Sparse
parser.add_argument("--im_logdir", default="./results/IGR_sparse/", help="path to the log directory", type=str)
### Dense
# parser.add_argument("--im_logdir", default="./results/IGR_dense/", help="path to the log directory", type=str)

parser.add_argument("--im_ckpt", default="latest.pth", help="checkpoint", type=str)
##########

parser.add_argument('--use_gt_3d', action='store_true')

parser.add_argument('--with_direct_opt', action='store_true')
parser.add_argument('--separate', action='store_true')
parser.add_argument('--use_pretrained_2d', action='store_true')

### For post processing
parser.add_argument('--seg_post_process', action='store_true')
parser.add_argument('--scale_post_process', action='store_true')
parser.add_argument('--extent_post_process', action='store_true')
parser.add_argument('--igr_post_process', action='store_true')
parser.add_argument('--igr_post_process_reinit', action='store_true')

#### Automation based on order and operation
parser.add_argument('--design_option', type=int,  default=1, help='Design option modes')


### Output folder to copy
parser.add_argument('--output_dir', default= "output_visu/", type=str)


# torch.manual_seed(10) ## bad
torch.manual_seed(1234)
# torch.manual_seed(0)	## good

np.random.seed(0)

FLAGS = parser.parse_args()

DESIGN_OPTION = FLAGS.design_option


if DESIGN_OPTION == 1:
	ops = np.array([1, 1, 1, 1, 1, 1, 1, 1])
	perm = np.array([0, 1, 2, 3, 4, 5, 6, 7])	

elif DESIGN_OPTION == 2:
	ops = np.array([-1, 1, 1])
	perm = np.array([1, 0, 2])

elif DESIGN_OPTION == 3:
	ops = np.array([-1, -1, 1, 1])
	perm = np.array([2, 1, 0, 3])

elif DESIGN_OPTION == 4:
	ops = np.array([1, -1, 1])
	perm = np.array([0, 1, 2])

elif DESIGN_OPTION == 5:
	ops = np.array([1, 1, -1])
	perm = np.array([0,1,2])

DATA_SPLIT = FLAGS.data_split
DATA_DIR = FLAGS.data_dir
MODEL_ID = FLAGS.model_id
NUM_POINTS = FLAGS.num_points

OUT_FNAME = FLAGS.out_fname
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)
plot_fol = os.path.join(DUMP_DIR, "plot")
if not os.path.exists(plot_fol): os.mkdir(plot_fol)

OUTPUT_DIR = FLAGS.output_dir
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
recons_fol = os.path.join(OUTPUT_DIR, "reconstruction")
if not os.path.exists(recons_fol): os.mkdir(recons_fol)
pc_input_fol = os.path.join(OUTPUT_DIR, "input_point_clouds")
if not os.path.exists(pc_input_fol): os.mkdir(pc_input_fol)
intermediate_fol = os.path.join(OUTPUT_DIR, "intermediate_volumes")
if not os.path.exists(intermediate_fol): os.mkdir(intermediate_fol)

#### Visu for debugging
filename = "render.sh"
f = open(os.path.join(DUMP_DIR, filename), "w")

## To store the output image files
filename = "image_files.sh"
g = open(os.path.join(DUMP_DIR, filename), "w")    

os.makedirs(os.path.join(DUMP_DIR, "point_cloud"), exist_ok=True)
os.makedirs(os.path.join(DUMP_DIR, "tmp"), exist_ok=True)
os.makedirs(os.path.join(DUMP_DIR, "rendering_point_cloud"), exist_ok=True)
#######

### Marching cubes
RES = FLAGS.resolution
RANGE = FLAGS.range
LEVEL = FLAGS.level

### Network
MODEL = FLAGS.model
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt
K = FLAGS.K
NUM_SK_POINT = FLAGS.num_sk_point

PRED_SEG = FLAGS.pred_seg
PRED_NORMAL = FLAGS.pred_normal
PRED_EXT = FLAGS.pred_extrusion
PRED_BB = FLAGS.pred_bb
PRED_OP = FLAGS.pred_op
NORM_EIG = FLAGS.norm_eig

USE_WHOLE_PC = FLAGS.use_whole_pc
USE_EXTRUSION_AXIS_FEAT = FLAGS.use_extrusion_axis_feat

IM_LOGDIR = FLAGS.im_logdir
IM_CKPT = FLAGS.im_ckpt

USE_GT_3D = FLAGS.use_gt_3d

DIRECT_OPT = FLAGS.with_direct_opt
SEPARATE = FLAGS.separate
USE_PRETRAINED_2D = FLAGS.use_pretrained_2d


### For postprocess
SEG_PP = FLAGS.seg_post_process
SCALE_PP = FLAGS.scale_post_process
EXTENT_PP = FLAGS.extent_post_process
IGR_PP = FLAGS.igr_post_process
IGR_PP_INIT = FLAGS.igr_post_process_reinit

#######

### Load the geometry

# Individual model files
h5_file = h5_file = os.path.join(DATA_DIR+DATA_SPLIT, "h5", str(MODEL_ID)+'.h5')
point_cloud, normals, extrusion_labels, extrusion_axes, extrusion_distances,\
		 n_instances, vertices, faces, face_normals, face_to_ids, norm_factor, operation = get_model(h5_file, mesh_info=True, operation=True)

### For current sample
curr_pc = point_cloud
# curr_pc = curr_pc.astype(float)
curr_n_instances = n_instances[0]
print("Number of extrusion instances: "+str(curr_n_instances))

### Downsample depending on number of points ##
idx = np.arange(curr_pc.shape[0])
np.random.shuffle(idx)
curr_pc = curr_pc[idx[:NUM_POINTS]]


#### Save input point cloud too
#### Output the input depth point cloud
verts_tuple = np.zeros((NUM_POINTS,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

for j in range(0, NUM_POINTS):
    verts_tuple[j] = tuple(curr_pc[j, :])

el_verts = PlyElement.describe(verts_tuple, "vertex")

print(verts_tuple)

ply_filename_out = os.path.join(pc_input_fol, MODEL_ID+"_input.ply")
PlyData([el_verts], text=True).write(ply_filename_out)        
#######


### Initialize and load network
device = torch.device('cuda')
MODEL_IMPORTED = importlib.import_module(MODEL)

pred_sizes = []
if PRED_NORMAL:
	pred_sizes.append(3)
else:
	pred_sizes.append(1) ##dummy DO NOT USE in prediction

if PRED_SEG and PRED_BB:
	# 2K classes instead of K
	pred_sizes.append(2*K)
elif PRED_SEG:
	pred_sizes.append(K)
else:
	pred_sizes.append(1) ##dummy DO NOT USE in prediction

model = MODEL_IMPORTED.backbone(output_sizes=pred_sizes)
GLOBAL_SIGMA = 1.8
LOCAL_SIGMA = 0.01
D_IN = 2
LATENT_SIZE = 256
sampler = NormalPerPoint(GLOBAL_SIGMA, LOCAL_SIGMA)
## Implicit
implicit_net = ImplicitNet(d_in=D_IN+LATENT_SIZE, dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ], skip_in = [4], geometric_init= True, radius_init = 1, beta=100)
## PointNet
if not USE_WHOLE_PC:
	pn_encoder = PointNetEncoder(LATENT_SIZE, D_IN, with_normals=True)
else:
	if USE_EXTRUSION_AXIS_FEAT:
		pn_encoder = PointNetEncoder(LATENT_SIZE, 7, with_normals=False) ## 3d pc plus confidence mask, plus extrusion axis
	else:
		pn_encoder = PointNetEncoder(LATENT_SIZE, 4, with_normals=False) ## 3d pc plus confidence mask, plus extrusion axis


fname = os.path.join(LOG_DIR, CKPT)
model.load_state_dict(torch.load(fname)["model"])
implicit_net.load_state_dict(torch.load(fname)["implicit_net"])
pn_encoder.load_state_dict(torch.load(fname)["pn_encoder"])

model.to(device)
implicit_net.to(device)
pn_encoder.to(device)
model.eval()
implicit_net.eval()
pn_encoder.eval()

print("Model loaded.")

## Loaded pre-trained sketch only encoder
IM_LOGDIR = FLAGS.im_logdir
IM_CKPT = FLAGS.im_ckpt

if USE_PRETRAINED_2D:
	loaded_pn_encoder = PointNetEncoder(LATENT_SIZE, D_IN, with_normals=True)
	loaded_pn_encoder.to(device)

	fname = os.path.join(IM_LOGDIR, IM_CKPT)
	implicit_net.load_state_dict(torch.load(fname)["model_state_dict"])	
	loaded_pn_encoder.load_state_dict(torch.load(fname)["encoder_state_dict"])	
	print("Pre-trained fixed implicit model loaded.")
	loaded_pn_encoder.eval()
print()
##########

start_time = time.time()

#### Extrusion parameters
if USE_GT_3D:
	print("Non-implemented for this type of loading...")
	exit()	

else:
	with torch.no_grad():
		gt_extrusion_labels = torch.from_numpy(extrusion_labels[idx[:NUM_POINTS]]).unsqueeze(0).to(device)
		NUM_POINT = curr_pc.shape[0]

		### Use network
		curr_pc = torch.from_numpy(curr_pc).unsqueeze(0).to(device).to(torch.float)
		# X, W_raw, O, _, _ = model(curr_pc)
		X, W_raw = model(curr_pc)

		X = F.normalize(X, p=2, dim=2, eps=1e-12)

		W_2K = torch.softmax(W_raw, dim=2)

		## 2K classes were predicted, create segmentation pred
		# Barrel
		W_barrel = W_2K[:, :, ::2]
		# Base
		W_base = W_2K[:, :, 1::2]

		W = W_barrel + W_base

		# Base or barrel segmentation
		'''
		0 for barrel
		1 for base
		''' 
		BB = torch.zeros(1, NUM_POINT, 2).to(device)                
		for j in range(K):
			BB[:,:,0] += W_2K[:, :, j*2]
			BB[:,:,1] += W_2K[:, :, j*2+1]


		W_ = hard_W_encoding(W, to_null_mask=True)

		matching_indices, mask = hungarian_matching(W_, gt_extrusion_labels, with_mask=True)
		mask = mask.float()

		## For visualization
		W_reordered_unmasked = torch.gather(W_, 2, matching_indices.unsqueeze(1).expand(1, NUM_POINT, K)) # BxNxK
		W_reordered = torch.where((mask).unsqueeze(1).expand(1, NUM_POINT, K)==1, W_reordered_unmasked, torch.ones_like(W_reordered_unmasked)* -1.)

		## Get original W probabilities
		W_soft_reordered_unmasked = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(1, NUM_POINT, K)) # BxNxK
		W_soft_reordered = torch.where((mask).unsqueeze(1).expand(1, NUM_POINT, K)==1, W_soft_reordered_unmasked, torch.ones_like(W_soft_reordered_unmasked)* -1.)

		label = torch.argmax(W_reordered, dim=-1)
		pred_bb_label = torch.argmax(BB, dim=-1)

		EA_X = X
		EA_W = W_reordered

		W_barrel_reordered = torch.gather(W_barrel, 2, matching_indices.unsqueeze(1).expand(1, NUM_POINT, K)) # BxNxK
		W_base_reordered = torch.gather(W_base, 2, matching_indices.unsqueeze(1).expand(1, NUM_POINT, K)) # BxNxK

		E_AX = estimate_extrusion_axis(EA_X, W_barrel_reordered, W_base_reordered, label, pred_bb_label, normalize=NORM_EIG)


		## Extrusion centers
		## For center prediction
		predicted_centroids = torch.zeros((1, curr_n_instances, 3)).to(device)
		found_centers_mask = torch.zeros((1, curr_n_instances)).to(device)

		## Calculate centroids of each segment
		for j in range(curr_n_instances):
			### Get points on the segment
			curr_segment_W = EA_W[:, :, j]
			indices_pred = curr_segment_W==1
			indices_pred = indices_pred.nonzero()

			for b in range(1):
				## get indices in current point cloud
				curr_batch_idx = indices_pred[:,0]==b

				## No points found in segment (1 point found is considered no points to handle .squeeze() function)
				if (curr_batch_idx.nonzero().shape[0]<=1):
					found_centers_mask[b,j] = 0.0
					continue

				curr_batch_idx = curr_batch_idx.nonzero().squeeze()
				curr_batch_pt_idx = indices_pred[:,1][curr_batch_idx]
				curr_segment_pc = torch.gather(curr_pc[b,:,:], 0, curr_batch_pt_idx.unsqueeze(-1).repeat(1,3))

				## Get center
				pred_centroid = torch.mean(curr_segment_pc, axis=0)

				predicted_centroids[b, j, :] = pred_centroid
				found_centers_mask[b,j] = 1.0

	extents, _ = get_extrusion_extents(curr_pc, label, pred_bb_label, E_AX[:,:curr_n_instances], predicted_centroids, num_points_to_sample=1024)
	extents = extents.permute(1,0,2)

	## Extrusion parameters
	curr_pc = curr_pc.squeeze().to("cpu").detach().numpy()
	curr_normal = X.squeeze().to("cpu").detach().numpy()
	curr_extrusion_labels = label.squeeze().to("cpu").detach().numpy()
	curr_bb_labels = pred_bb_label.squeeze().to("cpu").detach().numpy()
	curr_extrusion_axes = E_AX.squeeze()[:curr_n_instances].to("cpu").detach().numpy()
	curr_extrusion_centers = predicted_centroids.squeeze(0).to("cpu").detach().numpy()
	curr_extrusion_extents = extents.squeeze().to("cpu").detach().numpy()
	W_soft_reordered = W_soft_reordered.squeeze().to("cpu").detach().numpy()
####

######################################
######### Sketch extraction ###########
######################################

with torch.no_grad():
	### Projection based on extrusion parameters for implicit net condition
	projected_pc, projected_normal, pred_scales = sketch_implicit_projection(torch.from_numpy(curr_pc).unsqueeze(0).to(device), \
													torch.from_numpy(curr_normal).unsqueeze(0).to(device), \
													torch.from_numpy(curr_extrusion_labels).unsqueeze(0).to(device), \
													torch.from_numpy(curr_bb_labels).unsqueeze(0).to(device), \
													torch.from_numpy(curr_extrusion_axes).unsqueeze(0).to(device), \
													torch.from_numpy(curr_extrusion_centers).unsqueeze(0).to(device), num_points_to_sample=NUM_SK_POINT)
	projected_pc = projected_pc[:curr_n_instances]
	projected_normal = projected_normal[:curr_n_instances]
	pred_scales = pred_scales[:curr_n_instances]

	pred_scales_repeated = pred_scales.unsqueeze(-1).unsqueeze(-1).repeat(1,1, projected_pc.shape[-2], projected_pc.shape[-1])
	projected_pc /= pred_scales_repeated

	projected_pc = projected_pc.reshape(-1, NUM_SK_POINT, 2)
	projected_normal = projected_normal.reshape(-1, NUM_SK_POINT, 2)

	global_pc = torch.cat((projected_pc, projected_normal), dim=-1) 


	if USE_PRETRAINED_2D:
		latent_codes = loaded_pn_encoder(global_pc)
	else:
		latent_codes = pn_encoder(global_pc)
	#####

latent_codes_init = latent_codes
######################################

### Marching cubes hyperparameters
resol = (RES,RES,RES)
ranges = ((-RANGE, RANGE), (-RANGE, RANGE), (-RANGE, RANGE))
level = LEVEL
eps = (ranges[0][1]-ranges[0][0]) / resol[0]

## Initialize volume
xy_flat = compute_grid2D(resol, ranges=ranges).unsqueeze(0).cuda()
z_dim = resol[2]
z_range = ranges[2][1] - ranges[2][0]
z_lin = np.linspace(ranges[2][0], ranges[2][1], z_dim, endpoint=False) + z_range / z_dim * 0.5

volume = torch.ones([resol[2], resol[1], resol[0]]).cuda() * -1.0
###########

######################################
##### Insert post-processing here ####
######################################
W_soft_reordered = W_soft_reordered[:, :curr_n_instances]
row_sums = W_soft_reordered.sum(axis=-1)
W_soft_reordered = W_soft_reordered / row_sums[:, np.newaxis]

### Check previous segmentation accuracy
acc = np.sum(curr_extrusion_labels==extrusion_labels[idx[:NUM_POINTS]])/curr_pc.shape[0]
print("Original accuracy: "+str(acc))
print()
###

### Hyperparameters
NEIGHBORHOOD_PERCENT = 0.02
UNCONFIDENT_PRED_LABEL = 0.6
CONSENSUS_THRESH_PERCENT = 0.8
RELABEL_THRESH_PERCENT = 0.7
NUM_ITERATIONS = 10

if SEG_PP:
	## Get local neighborhood of each point in the point cloud
	pc_nbrs = KDTree(curr_pc)
	num_neighbors=int(curr_pc.shape[0] * NEIGHBORHOOD_PERCENT) ## let the local neighborhood be a proportion of the total number of points
	distances, indices = pc_nbrs.query(curr_pc,k=num_neighbors)
	indices_reshaped = np.reshape(indices, (-1))

	### Do relabeling 
	extrusion_relabeled = []
	consensus_threshold = num_neighbors * CONSENSUS_THRESH_PERCENT
	relabel_threshold = num_neighbors * RELABEL_THRESH_PERCENT

	prev_labels = np.copy(curr_extrusion_labels)


	### Make labels (curr_n_instances) if the confidence is too low
	prob_pred_label = np.max(W_soft_reordered, axis=-1)
	indices_to_mask = prob_pred_label < UNCONFIDENT_PRED_LABEL
	num_unknown_labels = np.sum(indices_to_mask)
	print("Num unknown = "+str(num_unknown_labels))

	### Mask label for unknown
	prev_labels[indices_to_mask] = curr_n_instances


	##### When a label as a disconnected component unlabel it
	for i in range(curr_n_instances):
		### Get points with label of a current instance
		segment_idx = np.where(prev_labels == i)[0]
		segment_points = curr_pc[segment_idx]

		print(segment_points.shape)

		if (segment_points.shape[0]==0):
			continue

		db = DBSCAN(eps=0.2, min_samples=20).fit(segment_points)
		labels = db.labels_
		# print(labels)
		# exit()

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)	
		print("Number of clusters for label " + str(i) + ": " + str(n_clusters_))

		### Unlabel for -1
		mask_idx = np.where(labels == -1)[0]
		prev_labels[segment_idx[mask_idx]] = curr_n_instances

		if n_clusters_ > 1:
			### Find dominant segment
			dominant_cluster = np.bincount(labels+1).argmax()
			mask_idx = labels != (dominant_cluster-1)
			prev_labels[segment_idx[mask_idx]] = curr_n_instances
	##################


	for j in range(NUM_ITERATIONS):
		corresponding_labels = np.reshape(prev_labels[indices_reshaped], (curr_pc.shape[0], -1))
		### Check for consensus in the neighborhood
		hist = np.apply_along_axis(lambda x: np.bincount(x, minlength= (curr_n_instances+1)), axis=-1, arr=corresponding_labels)


		extrusion_relabeled = []
		for i in range(curr_pc.shape[0]):
			### For unknown labeled points
			if prev_labels[i] == curr_n_instances:
				label_consensus = np.argmax(hist[i])

				if label_consensus == curr_n_instances:
					label_consensus = np.argsort(hist[i])
					label_consensus = label_consensus[1]

				extrusion_relabeled.append(label_consensus)

			### For known labels
			else:
				### If current label agrees with most of the neighbors, continue
				if hist[i][prev_labels[i]] > consensus_threshold:
					extrusion_relabeled.append(prev_labels[i])

				### Otherwise if there is a majority, relabel
				else:
					### Max in histogram
					label_consensus = np.argsort(hist[i])
					found = False

					for j in range(curr_n_instances):
						if hist[i][label_consensus[j]] > relabel_threshold:
							extrusion_relabeled.append(label_consensus[j])
							found = True
							break

					if not found:
						extrusion_relabeled.append(prev_labels[i])
		extrusion_relabeled = np.array(extrusion_relabeled)
		prev_labels = extrusion_relabeled

		acc = np.sum(extrusion_relabeled==extrusion_labels[idx[:NUM_POINTS]])/curr_pc.shape[0]
		print("Refined accuracy: "+str(acc))
		print()

	visualize_segmentation_pc_bb_v2(MODEL_ID, DUMP_DIR, curr_pc, extrusion_labels[idx[:NUM_POINTS]], curr_extrusion_labels, curr_bb_labels, curr_bb_labels, f, g)
	# visualize_segmentation_pc_bb_v2(MODEL_ID, DUMP_DIR, curr_pc, curr_extrusion_labels, extrusion_relabeled, curr_bb_labels, curr_bb_labels, f, g)
	f.close()
	g.close()
	# exit()

else:
	extrusion_relabeled = curr_extrusion_labels


if SCALE_PP:
	### With RANSAC ###
	with torch.no_grad():
		### Projection based on extrusion parameters for implicit net condition
		pred_scales_refined = scale_ransac(torch.from_numpy(curr_pc).unsqueeze(0).to(device), \
											torch.from_numpy(extrusion_relabeled).unsqueeze(0).to(device), \
											torch.from_numpy(curr_bb_labels).unsqueeze(0).to(device), \
											torch.from_numpy(curr_extrusion_axes).unsqueeze(0).to(device), \
											torch.from_numpy(curr_extrusion_centers).unsqueeze(0).to(device), num_points_to_sample=NUM_SK_POINT)

	pred_scales_refined = pred_scales_refined.squeeze().to("cpu").detach().numpy()
	print(pred_scales_refined)
	pred_scales = pred_scales_refined
#########################

if EXTENT_PP:
	##### RANSAC for extent #####
	extents, _ = extents_clustering(torch.from_numpy(curr_pc).unsqueeze(0).to(device), \
												torch.from_numpy(extrusion_relabeled).unsqueeze(0).to(device), \
												torch.from_numpy(curr_bb_labels).unsqueeze(0).to(device), \
												E_AX[:,:curr_n_instances], \
												torch.from_numpy(curr_extrusion_centers).unsqueeze(0).to(device), num_points_to_sample=2048)

	curr_extrusion_extents = extents
	print(curr_extrusion_extents)
	############################


###### Render current sketches ########
for i in range(curr_n_instances):
	# pnts = sketches[model_idx][i]
	pnts = None
	curr_latent = latent_codes_init[i]
	plot_surface_2d(decoder=implicit_net,
					path=plot_fol,
					epoch=str(i),
					shapename=MODEL_ID,
					points=pnts,
					latent=curr_latent,
					resolution=512,mc_value=0.0,is_uniform_grid=True,verbose=False,save_html=False,save_ply=False,overwrite=True)
#######################################

######################################
######################################
######################################

if IGR_PP:
	im_lr_schedules = get_learning_rate_schedules([{
													"Type" : "Step",
													"Initial" : 0.001,
													# "Initial" : 0.0001,
													"Interval" : 500,
													"Factor" : 0.5
													},
													{
													"Type" : "Step",
													"Initial" : 0.001,
													"Interval" : 1000,
													"Factor" : 0.5
													}])
	im_weight_decay = 0

	optimizer = torch.optim.Adam(
		[ 	{
				"params": implicit_net.parameters(),
				"lr": im_lr_schedules[0].get_learning_rate(0),
				"weight_decay": im_weight_decay
			}
		])

	### Project the prediction
	projected_pc, projected_normal, pred_scales = sketch_implicit_projection(torch.from_numpy(curr_pc).unsqueeze(0).to(device), \
													torch.from_numpy(curr_normal).unsqueeze(0).to(device), \
													# torch.from_numpy(curr_extrusion_labels).unsqueeze(0).to(device), \
													torch.from_numpy(extrusion_relabeled).unsqueeze(0).to(device), \
													torch.from_numpy(curr_bb_labels).unsqueeze(0).to(device), \
													torch.from_numpy(curr_extrusion_axes).unsqueeze(0).to(device), \
													torch.from_numpy(curr_extrusion_centers).unsqueeze(0).to(device), num_points_to_sample=NUM_SK_POINT)

	with torch.no_grad():
		pred_scales_repeated = pred_scales.unsqueeze(-1).unsqueeze(-1).repeat(1,1, projected_pc.shape[-2], projected_pc.shape[-1])
		projected_pc /= pred_scales_repeated

		projected_pc_ = projected_pc.reshape(-1, NUM_SK_POINT, 2)
		projected_normal_ = projected_normal.reshape(-1, NUM_SK_POINT, 2)

		global_pc = torch.cat((projected_pc_, projected_normal_), dim=-1) 


		if USE_PRETRAINED_2D:
			latent_codes = loaded_pn_encoder(global_pc)
		else:
			latent_codes = pn_encoder(global_pc)
		#####

		latent_codes_init = latent_codes


### Loop through each extrusion segment and compose the volume
found = False
for i in range(curr_n_instances):
	j = perm[i]

	ax = curr_extrusion_axes[j]
	c = curr_extrusion_centers[j]
	extent = curr_extrusion_extents[j]
	scale = pred_scales[j]

	if np.abs(extent[0] - extent[1]) < 0.01:
		print("Extrusion segment too shallow. Skipping.")
		print()
		continue

	##### IGR Direct optimization
	if IGR_PP:
		if not IGR_PP_INIT:
			### Always start with preloaded then directly optimize
			fname = os.path.join(IM_LOGDIR, IM_CKPT)
			implicit_net.load_state_dict(torch.load(fname)["model_state_dict"])
			print("Loaded implcit net.")

		else:
			implicit_net = ImplicitNet(d_in=D_IN+LATENT_SIZE, dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ], skip_in = [4], geometric_init= True, radius_init = 1, beta=100)		
			implicit_net.to(device)

		implicit_net.train()
		global_step = 0
		curr_implicit_latent_code = latent_codes_init[j]
		curr_implicit_latent_code = curr_implicit_latent_code.unsqueeze(0)	
		sk_pnts_orig = projected_pc[j]
		sk_normals = projected_normal[j]

		prev_lost = None
		eps_lost = 1e-5
		# eps_lost = 1e-7

		# for it in range(1000):
		for it in range(10000):
			nonmnfld_pnts = sampler.get_points(sk_pnts_orig)
			sk_pnts = add_latent(sk_pnts_orig, curr_implicit_latent_code)
			nonmnfld_pnts = add_latent(nonmnfld_pnts, curr_implicit_latent_code)

			# forward pass
			sk_pnts.requires_grad_()
			nonmnfld_pnts.requires_grad_()

			sk_pred = implicit_net(sk_pnts)
			nonmnfld_pred = implicit_net(nonmnfld_pnts)

			mnfld_grad = gradient(sk_pnts, sk_pred)
			nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

			sk_pred = sk_pred.reshape(1, -1, 1)
			nonmnfld_grad = nonmnfld_grad.reshape(1, -1, 2)
			mnfld_grad = mnfld_grad.reshape(1, -1, 2)
			sk_normals = sk_normals.reshape(1, -1, 2)

			mnfld_loss = (sk_pred.abs()).mean(dim=-1).mean(dim=-1).mean()

			# eikonal loss
			grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean(dim=-1).mean()

			# normals loss
			norm_sub = (mnfld_grad - sk_normals).norm(2, dim=-1)
			norm_add = (mnfld_grad + sk_normals).norm(2, dim=-1)

			values = torch.cat((norm_sub.unsqueeze(-1), norm_add.unsqueeze(-1)), dim=-1)
			normals_loss = torch.min(values, dim=-1)[0]
			normals_loss = normals_loss.mean(dim=-1).mean()

			im_loss = mnfld_loss + 0.1 * grad_loss
			im_loss = im_loss + 1.0 * normals_loss
			optimizer.zero_grad()

			im_loss.backward()
			optimizer.step()
			global_step += 1

			if it%100 ==0:
				print("IGR loss: "+str(im_loss.item()))

			if prev_lost is not None:
				if torch.abs(im_loss - prev_lost) < eps_lost:
					break

			prev_lost = im_loss

		implicit_net.eval()
		# pnts = sketches[model_idx][j]
		pnts = None
		curr_latent = latent_codes_init[j]
		plot_surface_2d(decoder=implicit_net,
						path=plot_fol,
						epoch=str(j),
						shapename=MODEL_ID+"_refined",
						points=pnts,
						latent=curr_latent,
						resolution=512,mc_value=0.0,is_uniform_grid=True,verbose=False,save_html=False,save_ply=False,overwrite=True)
		#############################

	# # Edit 2
	# if i == 1:
	# 	print("Editing...")
	# 	c -= np.array([0, 0.3, 0])
		# extent = np.abs(extent) - 0.3

	with torch.no_grad():
		## This is for a single segment	
		#### Extrusion Parameters
		ax = torch.from_numpy(ax).unsqueeze(0).to(xy_flat.device).float()
		c = torch.from_numpy(c).unsqueeze(0).to(xy_flat.device).float()

		# # ## Edit 1
		# if i == 0:
		# 	print("Editing...")
		# 	scale *= 2

		##### For transformation to sketch coordinate space
		rotation_matrices = get_visualizer_rotation_matrix(ax, xy_flat.device)
		#####

		print("For extrusion "+str(j))
		print("Extrusion axis")
		print(ax)
		print("Extrusion center")
		print(c)
		print("Extrusion scale")
		print(scale)
		print("Extrusion extent")
		print(extent)
		print()

		curr_implicit_latent_code = latent_codes_init[j]
		curr_implicit_latent_code = curr_implicit_latent_code.unsqueeze(0)

		### Intermediate_volume
		volume_intermdiate = torch.ones([resol[2], resol[1], resol[0]]).cuda() * -1.0

		for z_ind, z_val in enumerate(z_lin):
			xyz_coord = torch.cat([xy_flat, torch.ones(1, xy_flat.shape[1], 1).cuda() * z_val], 2)

			### Check if inside the sketch
			xyz_coord_projected = transform_to_sketch_plane(xyz_coord, rotation_matrices, c, scale)

			### Compute for occupancy

			### Slow
			net_input = add_latent(xyz_coord_projected, curr_implicit_latent_code)
			sk_pred = implicit_net(net_input)

			occupancy_sdf = (sk_pred <= 0.0).to(torch.float).T

			curr_occupancy = occupancy_sdf
			curr_sdf1 = sk_pred.to(torch.float).T
			##########

			### Check if inside the extent
			dist = get_distances_on_extrusion_axis(xyz_coord, ax, c)

			### Make extent bigger if it is a cut for better visualization
			if ops[j] == -1:
				# eps = np.max((eps, np.max(np.abs(extent))*0.02))
				eps = np.max(np.abs(extent))*0.5

			occupancy_extent = (torch.abs(dist) <= np.max(np.abs(extent))+eps).to(torch.float)
			curr_occupancy *= occupancy_extent
			curr_sdf2 = (np.max(np.abs(extent)) - torch.abs(dist)).to(torch.float) 


			multiplier = torch.ones(curr_sdf1.shape).to(torch.float).to(curr_sdf1.device) * -1.0
			mask = torch.where((occupancy_sdf==1)&(occupancy_extent==1))
			multiplier[mask] = 1.0

			curr_sdf = torch.min(torch.abs(torch.cat((curr_sdf1, curr_sdf2), dim=0)), dim=0)[0] * multiplier * scale
			#####		

			## For SDF
			curr_sdf = curr_sdf.reshape([resol[0], resol[1]])

			## First operation
			if i == 0:
				volume[z_ind] = (curr_sdf * ops[j])

			else:
				if ops[j] == -1:
					occupancy_sdf = (sk_pred <= 0.0001).to(torch.float).T  				### Some threshold to make it smooth
				else:
					occupancy_sdf = (sk_pred <= 0.05).to(torch.float).T  				### Some threshold to make it smooth

				occupancy_sdf = occupancy_sdf.reshape([resol[0], resol[1]])
				occupancy_extent = occupancy_extent.reshape([resol[0], resol[1]])

				### Works but a bit hacky --> Current working version
				mask = torch.where((occupancy_sdf==1)&(occupancy_extent==1))
				volume[z_ind][mask] = (curr_sdf * ops[j])[mask]

			### Output intermediate volume
			volume_intermdiate[z_ind] = curr_sdf
		
		### Save intermediate volume
		volume_intermdiate = volume_intermdiate.to("cpu")
		try:
			convert_sdf_samples_to_ply(volume_intermdiate, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], os.path.join(intermediate_fol, MODEL_ID+str(i)+".ply"), level=level)
		except:
			continue
		found = True

volume = volume.to("cpu")
print("Constructed occupancy volume")
print(torch.min(volume))
print(torch.max(volume))

try:
	convert_sdf_samples_to_ply(volume, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], os.path.join(recons_fol, MODEL_ID+".ply"), level=level)
except:
	pass

if -1 in ops:
	### Remove holes and artifacts
	mesh = trimesh.load_mesh(os.path.join(recons_fol, MODEL_ID+".ply"))
	whole_volume = mesh.volume
	components = mesh.split()

	thresh = 0.1
	components_to_take = []
	for comp in components:
		vol = comp.volume
		if vol > whole_volume*thresh:
			components_to_take.append(comp)

	mesh = trimesh.util.concatenate(components_to_take)
	mesh.export(os.path.join(recons_fol, MODEL_ID+".ply"))

print()
print('Total time: {}'.format(time.time() - start_time))
















