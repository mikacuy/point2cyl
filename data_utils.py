# Mikaela Uy (mikacuy@cs.stanford.edu)
import os, sys
import json
import numpy as np
import h5py
import random
import trimesh
from PIL import Image
import importlib.util
import torch
import torch.nn.functional as F
import torchgeometry as tgm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, '..','data_preprocessing'))

from global_variables import *
from utils import *

### For marching cube
from skimage import measure
import skimage
import plyfile, logging
import time

### For extent clustering
from sklearn.cluster import DBSCAN
from sklearn import metrics

TORCH_PI = torch.acos(torch.zeros(1)).item() * 2

def rotate_point_cloud_with_normal(batch_xyz, batch_normal):
    ''' Randomly rotate XYZ, normal point cloud.
    '''
    for k in range(batch_xyz.shape[0]):
        rotation_angle = torch.rand(1) * 2 * TORCH_PI
        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor([[cosval, 0., sinval],
                                    [0., 1., 0.],
                                    [-sinval, 0., cosval]])
        shape_pc = batch_xyz[k,:,:]
        shape_normal = batch_normal[k,:,:]
        batch_xyz[k,:,:] = torch.matmul(shape_pc.view((-1, 3)), rotation_matrix)
        batch_normal[k,:,:] = torch.matmul(shape_normal.view((-1, 3)), rotation_matrix)

    return batch_xyz, batch_normal

def rotate_point_cloud_with_normal_discretized(batch_xyz, batch_normal):
	''' Randomly rotate XYZ, normal point cloud.
	'''
	for k in range(batch_xyz.shape[0]):
		## Randomly pick an axis
		# 0:x, 1:y, 2:z
		axis_selection = torch.randint(0, 3, (1,))

		rotation_angle = torch.randint(0, 4, (1,)) * 0.5 * TORCH_PI

		cosval = torch.cos(rotation_angle[0])
		sinval = torch.sin(rotation_angle[0])

		if axis_selection[0] == 0:
			#x-axis
			rotation_matrix = torch.tensor([[1., 0., 0.],
											[0., cosval, -sinval],
											[0., sinval, cosval]])
		elif axis_selection[0] == 1:
			rotation_matrix = torch.tensor([[cosval, 0., sinval],
											[0., 1., 0.],
											[-sinval, 0., cosval]])
		else:
			rotation_matrix = torch.tensor([[cosval, -sinval, 0.],
											[sinval, cosval, 0.],
											[0., 0., 1.]])	 

		shape_pc = batch_xyz[k,:,:]
		shape_normal = batch_normal[k,:,:]
		batch_xyz[k,:,:] = torch.matmul(shape_pc.view((-1, 3)), rotation_matrix)
		batch_normal[k,:,:] = torch.matmul(shape_normal.view((-1, 3)), rotation_matrix)

	return batch_xyz, batch_normal

def add_noise(batch_xyz, batch_normal, sigma=0.01):
	'''
	Adds a random gaussian noise for each point in the direction of the normal
	'''
	# print("adding noise")
	batch_size, num_points, _ = batch_xyz.shape

	sampled_noise = np.random.normal(0.0, sigma, (batch_size, num_points))
	sampled_noise = np.tile(np.expand_dims(sampled_noise, axis=-1), [1,1,3])

	noisy_pc = batch_xyz + torch.tensor(sampled_noise)*batch_normal

	return noisy_pc


def estimate_extrusion_axis(X, W_barrel, W_base, gt_bb_labels, gt_extrusion_instances, normalize=False):
	'''
	X : (batch_size, num_points, 3) predicted normals
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	W_base : (batch_size, num_points, K) segmentation prediction (odd rows from W_2K)
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''
	batch_size, num_points, K = W_barrel.shape

	'''
	for i = 0, 1, ..., K
	Objective: min(BTB-CTC) 
		where BTB is the barrel matrix and CTC is the base matrix
	 	Bx = 0, Cx = 1
	B = diag(w_k_barrel)*X
	C = diag(w_k_base)*X
	'''
	E_AX = torch.zeros((batch_size, K, 3)).to(W_base.device)
	for i in range(K):
		w_i_barrel = W_barrel[:,:,i]
		w_i_base = W_base[:,:,i]

		# print(w_i_barrel)
		# print(w_i_barrel.shape)
		# print(w_i_base.shape)

		w_i_barrel = torch.diag_embed(w_i_barrel)
		w_i_base = torch.diag_embed(w_i_base)

		# print(w_i_barrel)
		# print(w_i_barrel.shape)
		# print(w_i_base.shape)

		if normalize:
			# Compute weights from gt number of points in base/barrel
			ind_i = torch.where(gt_extrusion_instances==i, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
			ind_barrel = torch.where(gt_bb_labels==0, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
			ind_base = torch.where(gt_bb_labels==1, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
			ind_barrel_i = ind_i*ind_barrel
			ind_base_i = ind_i*ind_base

			norm_barrel = torch.sum(ind_barrel_i, dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, num_points, 3)  # (B, 4096, 3)
			norm_base = torch.sum(ind_base_i, dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, num_points, 3)  # (B, 4096, 3)
			norm_barrel = torch.sqrt(norm_barrel)
			norm_base = torch.sqrt(norm_base)

			# print(ind_i)
			# print(ind_base)
			# print(ind_base_i)
			# print(ind_barrel_i.shape)
			# print(ind_base_i.shape)
			# print(norm_barrel.shape)
			# print(norm_base.shape)
			# print()

		B = torch.bmm(w_i_barrel, X)
		C = torch.bmm(w_i_base, X)

		if normalize:
			B = torch.div(B, norm_barrel+1.0)
			C = torch.div(C, norm_base+1.0)

		BTB = torch.bmm(torch.transpose(B, 1, 2), B)
		CTC = torch.bmm(torch.transpose(C, 1, 2), C)

		# print(B.shape)
		# print(C.shape)
		# print(BTB.shape)
		# print(CTC.shape)

		e, v = torch.symeig(BTB-CTC, eigenvectors=True)
		ax = v[:,:,0]
		E_AX[:, i, :] = ax

		# print(e[0])
		# print(v[0])

	return E_AX

def estimate_extrusion_axis_seperate(X, W_bb, W_seg, gt_bb_labels, gt_extrusion_instances, normalize=False):
    '''
    X: (batch_size, num_points, 3) predicted normals
    W_bb: (batch_size, num_points, 2) 0th col for barrel, 1st col for base
    W_seg: (batch_size, num_points, K) segmentation prediction
    gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
    gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
    '''
    batch_size, num_points, K = W_seg.shape
    W_barrel = W_seg * W_bb[:, :, 0].unsqueeze(-1).repeat(1, 1, K)
    W_base = W_seg * W_bb[:, :, 1].unsqueeze(-1).repeat(1, 1, K)
    E_AX = torch.zeros((batch_size, K, 3)).to(W_seg.device)

    for i in range(K):
        w_i_barrel = W_barrel[:,:,i]
        w_i_base = W_base[:,:,i]

        # print(w_i_barrel)
        # print(w_i_barrel.shape)
        # print(w_i_base.shape)

        w_i_barrel = torch.diag_embed(w_i_barrel)
        w_i_base = torch.diag_embed(w_i_base)

        # print(w_i_barrel)
        # print(w_i_barrel.shape)
        # print(w_i_base.shape)

        if normalize:
            # Compute weights from gt number of points in base/barrel
            ind_i = torch.where(gt_extrusion_instances==i, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
            ind_barrel = torch.where(gt_bb_labels==0, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
            ind_base = torch.where(gt_bb_labels==1, torch.tensor([1.0]).to(W_base.device), torch.tensor([0.0]).to(W_base.device))
            ind_barrel_i = ind_i*ind_barrel
            ind_base_i = ind_i*ind_base

            norm_barrel = torch.sum(ind_barrel_i, dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, num_points, 3)  # (B, 4096, 3)
            norm_base = torch.sum(ind_base_i, dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, num_points, 3)  # (B, 4096, 3)
            norm_barrel = torch.sqrt(norm_barrel)
            norm_base = torch.sqrt(norm_base)

            # print(ind_i)
            # print(ind_base)
            # print(ind_base_i)
            # print(ind_barrel_i.shape)
            # print(ind_base_i.shape)
            # print(norm_barrel.shape)
            # print(norm_base.shape)
            # print()

        B = torch.bmm(w_i_barrel, X)
        C = torch.bmm(w_i_base, X)

        if normalize:
            B = torch.div(B, norm_barrel+1.0)
            C = torch.div(C, norm_base+1.0)

        BTB = torch.bmm(torch.transpose(B, 1, 2), B)
        CTC = torch.bmm(torch.transpose(C, 1, 2), C)

        # print(B.shape)
        # print(C.shape)
        # print(BTB.shape)
        # print(CTC.shape)

        e, v = torch.symeig(BTB-CTC, eigenvectors=True)
        ax = v[:,:,0]
        E_AX[:, i, :] = ax

        # print(e[0])
        # print(v[0])

    return E_AX

def estimate_extrusion_centers(W, pcs):
	## Input: W (B, N, K), pcs (B, N, 3)
	## Output: pred_centers (B, K, 3)

	batch_size, num_points, K = W.shape

	W = W.permute(0, 2, 1)
	W_reshaped = W.unsqueeze(-1).repeat(1,1,1,3)
	pcs_reshaped = pcs.unsqueeze(1).repeat(1,K,1,1)

	weighted_pcs = W_reshaped * pcs_reshaped #(B, K, N, 3)
	pred_centers = weighted_pcs.mean(dim=-2)
	
	return pred_centers


def sketch_projection(P, W, W_barrel, extrusion_axes, gt_bb_labels, gt_extrusion_instances, use_gt_seg = True, use_gt_bb = True):
	'''
	P : (batch_size, num_points, 3) input point cloud
	W: (batch_size, num_points, K) extrusion segmentation prediction (combined every two rows of W_2K)
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''	

	batch_size, num_points, K = W.shape

	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	## Project barrel points on each segment
	if use_gt_bb and use_gt_seg:
		W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))
	elif use_gt_bb and (not use_gt_seg):
		W_b = torch.where(gt_bb_labels_==0, W.float(), torch.tensor([0.0]).to(W.device))
	else:
		W_b = W_barrel


	P_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	
	for i in range(K):
		ax = extrusion_axes[:, i, :]

		# Compute centroid
		w_i_gt = gt_W_b[:,:,i]
		w_i_diag_gt = torch.diag_embed(w_i_gt)
		centroid = torch.mean(torch.bmm(w_i_diag_gt, P), dim=1)
		centroid = centroid.unsqueeze(1).repeat(1, num_points, 1)
		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		# w_i = EA_W[:,:,i].float()
		# w_i_diag = torch.diag_embed(w_i)

		w_i = W_b[:,:,i]
		w_i_diag = torch.diag_embed(w_i)

		points_in_segment = torch.bmm(w_i_diag, P)

		points_centered = torch.bmm(w_i_diag, points_in_segment - centroid)
		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = torch.bmm(w_i_diag, points_in_segment - delta) 

		P_projected[i, :, :, :] = points_projected

	return P_projected

def sketch_projection_v2(P, W, W_barrel, extrusion_axes, gt_bb_labels, gt_extrusion_instances, use_gt_seg = True, use_gt_bb = True):
	'''
	P : (batch_size, num_points, 3) input point cloud
	W: (batch_size, num_points, K) extrusion segmentation prediction (combined every two rows of W_2K)
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''	

	batch_size, num_points, K = W.shape

	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	## Project barrel points on each segment
	if use_gt_bb and use_gt_seg:
		W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))
	elif use_gt_bb and (not use_gt_seg):
		W_b = torch.where(gt_bb_labels_==0, W.float(), torch.tensor([0.0]).to(W.device))
	else:
		W_b = W_barrel


	P_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	
	for i in range(K):
		ax = extrusion_axes[:, i, :]

		# Compute centroid
		w_i_gt = gt_W_b[:,:,i]
		w_i_diag_gt = torch.diag_embed(w_i_gt)

		## Corrected centroid calculation
		points_in_segment_gt = torch.bmm(w_i_diag_gt, P)
		pt_norm = (torch.square(points_in_segment_gt).sum(-1) != 0).sum(dim=-1) # count number of points in segment
		pt_norm = pt_norm.unsqueeze(-1).repeat(1,3)
		sum_pts = torch.sum(points_in_segment_gt, dim=1)
		centroid = torch.div(sum_pts, pt_norm+g_zero_tol)	
		centroid = centroid.unsqueeze(1).repeat(1, num_points, 1)

		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		# w_i = EA_W[:,:,i].float()
		# w_i_diag = torch.diag_embed(w_i)

		w_i = W_b[:,:,i]
		w_i_diag = torch.diag_embed(w_i)

		points_in_segment = torch.bmm(w_i_diag, P)

		points_centered = torch.bmm(w_i_diag, points_in_segment - centroid)
		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = torch.bmm(w_i_diag, points_in_segment - delta) 

		P_projected[i, :, :, :] = points_projected

	return P_projected

def sketch_projection_v3(P, extrusion_axes, gt_bb_labels, gt_extrusion_instances):
	'''
	P : (batch_size, num_points, 3) input point cloud
	W: (batch_size, num_points, K) extrusion segmentation prediction (combined every two rows of W_2K)
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''	

	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	
	for i in range(K):
		ax = extrusion_axes[:, i, :]

		# Compute centroid
		w_i_gt = gt_W_b[:,:,i]
		w_i_diag_gt = torch.diag_embed(w_i_gt)

		## Corrected centroid calculation
		points_in_segment_gt = torch.bmm(w_i_diag_gt, P)
		pt_norm = (torch.square(points_in_segment_gt).sum(-1) != 0).sum(dim=-1) # count number of points in segment
		pt_norm = pt_norm.unsqueeze(-1).repeat(1,3)
		sum_pts = torch.sum(points_in_segment_gt, dim=1)
		centroid = torch.div(sum_pts, pt_norm+g_zero_tol)	
		centroid = centroid.unsqueeze(1).repeat(1, num_points, 1)

		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		points_centered = P - centroid

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = P - delta

		P_projected[i, :, :, :] = points_projected

	return P_projected

def gt_axis_sketch_projection(P, extrusion_axes, gt_bb_labels, gt_extrusion_instances, extrusion_centers, num_gt_points_to_sample=512):
	'''
	P : (batch_size, num_points, 3) input point cloud
	W: (batch_size, num_points, K) extrusion segmentation prediction (combined every two rows of W_2K)
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''	

	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	P_soft_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	gt_projected = torch.zeros((K, batch_size, num_gt_points_to_sample, 3)).to(P.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]
		centroid_p = centroid.unsqueeze(1).repeat(1, num_points, 1)

		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		###### For predicted segments
		points_centered = P - centroid_p

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = P - delta

		P_projected[i, :, :, :] = points_projected
		###############

		###### For ground truth
		curr_segment_mask = gt_W_b[:,:,i]
		indices = curr_segment_mask==1
		indices = indices.nonzero().squeeze()

		# print(indices)
		# print(indices.shape)
		batch_gt_unprojected = torch.zeros((batch_size, num_gt_points_to_sample, 3)).to(P.device)

		for j in range(batch_size):
			curr_sample_indices = indices[:,0]==j
			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			# print(curr_sample_indices)
			if (curr_sample_indices.shape[0]==0):
				batch_gt_unprojected[j, :, :] = torch.zeros((num_gt_points_to_sample, 3)).to(P.device)
				continue	

			curr_sample_pt_idx = indices[:,1][curr_sample_indices]
			# print(curr_sample_pt_idx)

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_gt_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]
			# print(sampled_idx)
			# print(P[j,:,:].shape)
			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			# print(sampled_gt_segment_pc.shape)
			
			batch_gt_unprojected[j, :, :] = sampled_gt_segment_pc

		centroid_g = centroid.unsqueeze(1).repeat(1, num_gt_points_to_sample, 1)
		points_centered = batch_gt_unprojected - centroid_g

		ax_expanded = ax.unsqueeze(1).repeat(1, num_gt_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_gt_points_to_sample, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = batch_gt_unprojected - delta

		gt_projected[i, :, :, :] = points_projected
		##################

	return P_projected, gt_projected

def gt_axis_sketch_projection_v2(P, W_barrel, extrusion_axes, gt_bb_labels, gt_extrusion_instances, extrusion_centers, num_gt_points_to_sample=512, num_soft_points_to_sample=512):
	'''
	P : (batch_size, num_points, 3) input point cloud
	W: (batch_size, num_points, K) extrusion segmentation prediction (combined every two rows of W_2K)
	W_barrel : (batch_size, num_points, K) segmentation prediction (even rows from W_2K)
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to
	gt_bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	gt_extrusion_instances: (batch_size, num_points) labels for the K extrusion segments
	'''	

	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points, 3)).to(P.device)
	P_soft_projected = torch.zeros((K, batch_size, num_soft_points_to_sample, 3)).to(P.device)
	gt_projected = torch.zeros((K, batch_size, num_gt_points_to_sample, 3)).to(P.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]
		centroid_p = centroid.unsqueeze(1).repeat(1, num_points, 1)

		###### For predicted segments
		points_centered = P - centroid_p

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = P - delta

		P_projected[i, :, :, :] = points_projected
		###############

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1
		indices = indices.nonzero().squeeze()

		curr_segment_pred_mask = W_barrel[:,:,i]
		indices_pred = curr_segment_pred_mask>=0.3
		indices_pred = indices_pred.nonzero()

		# print(curr_segment_pred_mask)
		# print(indices_pred.shape)
		# print(curr_segment_pred_mask.shape)

		# print(indices)
		# print(indices.shape)
		batch_gt_unprojected = torch.zeros((batch_size, num_gt_points_to_sample, 3)).to(P.device)
		batch_pred_unprojected = torch.zeros((batch_size, num_soft_points_to_sample, 3)).to(P.device)

		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j
			
			if (curr_sample_indices.nonzero().shape[0]<=1):
				batch_gt_unprojected[j, :, :] = torch.zeros((num_gt_points_to_sample, 3)).to(P.device)
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			# print(curr_sample_indices)
			if (curr_sample_indices.shape[0]==0):
				batch_gt_unprojected[j, :, :] = torch.zeros((num_gt_points_to_sample, 3)).to(P.device)
				continue	

			curr_sample_pt_idx = indices[:,1][curr_sample_indices]
			# print(curr_sample_pt_idx)

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_gt_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]
			# print(sampled_idx)
			# print(P[j,:,:].shape)
			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			# print(sampled_gt_segment_pc.shape)
			
			batch_gt_unprojected[j, :, :] = sampled_gt_segment_pc


			## For soft pred
			curr_sample_indices = indices_pred[:,0]==j

			if (curr_sample_indices.nonzero().shape[0]<=1):
				batch_pred_unprojected[j, :, :] = torch.zeros((num_soft_points_to_sample, 3)).to(P.device)
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			# print(curr_sample_indices)
			# print()
			if (curr_sample_indices.shape[0]==0):
				batch_pred_unprojected[j, :, :] = torch.zeros((num_soft_points_to_sample, 3)).to(P.device)
				continue	

			curr_sample_pt_idx = indices_pred[:,1][curr_sample_indices]
			# print(curr_sample_pt_idx)

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_soft_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]
			# print(sampled_idx)
			# print(P[j,:,:].shape)
			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			# print(sampled_gt_segment_pc.shape)
			
			batch_pred_unprojected[j, :, :] = sampled_gt_segment_pc

		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		centroid_g = centroid.unsqueeze(1).repeat(1, num_gt_points_to_sample, 1)
		points_centered = batch_gt_unprojected - centroid_g

		ax_expanded = ax.unsqueeze(1).repeat(1, num_gt_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_gt_points_to_sample, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = batch_gt_unprojected - delta
		gt_projected[i, :, :, :] = points_projected


		centroid_p = centroid.unsqueeze(1).repeat(1, num_soft_points_to_sample, 1)
		points_centered = batch_pred_unprojected - centroid_p

		ax_expanded = ax.unsqueeze(1).repeat(1, num_soft_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_soft_points_to_sample, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = batch_pred_unprojected - delta

		P_soft_projected[i, :, :, :] = points_projected
		###############

	return P_projected, gt_projected, P_soft_projected


### For evaluation
def sketch_projection_evaluation(P, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	'''
	P : (batch_size, num_points, 3) input point cloud
	seg_label: (batch_size, num_points) extrusion segmentation label
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to 
	extrusion_centers : (batch_size, K, 3) extrusion segment centers
	bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	num_points_to_sample: num points to sample
	'''	

	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 3)).to(P.device)
	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			found_centers_mask[j, i] = 1.0

		''' 
		# Make centroid the origin of the plane
		v = point - centroid
		# Get distance from point to plane
		dist = dot(v, extrusion_axis
		# Get projection
		proj = point - dist*extrusion_axis
		'''

		centroid_g = centroid.unsqueeze(1).repeat(1, num_points_to_sample, 1)
		points_centered = batch_projected - centroid_g

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points_to_sample, 1)

		delta = dist.repeat(1,1,3) * ax_expanded

		## project all points
		points_projected = batch_projected - delta
		P_projected[i, :, :, :] = points_projected
		###############

	return P_projected, found_centers_mask


def sketch_projection_evaluation_2d(P, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	'''
	P : (batch_size, num_points, 3) input point cloud
	seg_label: (batch_size, num_points) extrusion segmentation label
	extrusion_axes : (batch_size, K, 3) extrusion axis to project to 
	extrusion_centers : (batch_size, K, 3) extrusion segment centers
	bb_labels: (batch_size, num_points) 0 for barrel, 1 for base
	num_points_to_sample: num points to sample
	'''	

	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)
	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)
	
	scales = torch.ones((K, batch_size)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			found_centers_mask[j, i] = 1.0
		
		## Debug
		# curr_barrel_pc = batch_projected.to("cpu").detach().numpy()
		# # curr_barrel_pc = np.array([curr_barrel_pc[7, :, 0], curr_barrel_pc[7, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc[0])
		# o3d.io.write_point_cloud("unprojected.ply", pcd)
		########

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.view(-1,3).unsqueeze(1)
		

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)

		### Debug ###
		# curr_barrel_pc = points_projected.to("cpu").detach().numpy()
		# curr_barrel_pc = np.array([curr_barrel_pc[0, :, 0], curr_barrel_pc[0, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc.T)
		# o3d.io.write_point_cloud("projected.ply", pcd)
		#########

		## Center sketch
		# print(centroid.shape)
		# print(rotation_matrices.shape)
		# print()
		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)

		# print(centroid_projected.shape)
		# print(points_projected.shape)
		points_projected -= centroid_projected
		# print(points_projected.shape)
		# exit()

		## Rescale barrel
		scale = torch.max(torch.sum(torch.abs(points_projected)**2, axis=-1)**0.5, dim=-1)[0]
		#scale = torch.where(scale!=0, scale, torch.tensor([1e20]).to(scale.device))

		scales[i, :] = scale

		P_projected[i, :, :, :] = points_projected
		###############
	scales = torch.where(found_centers_mask.T==1, scales, torch.tensor([1.0]).to(scales.device))

	return P_projected, found_centers_mask, scales

def sketch_implicit_projection(P, X, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)
	X_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)

	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)
	
	scales = torch.ones((K, batch_size)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)
		batch_normals_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			sampled_gt_segment_normal = torch.gather(X[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			batch_normals_projected[j, :, :] = sampled_gt_segment_normal
			found_centers_mask[j, i] = 1.0
		
		## Debug
		# curr_barrel_pc = batch_projected.to("cpu").detach().numpy()
		# # curr_barrel_pc = np.array([curr_barrel_pc[7, :, 0], curr_barrel_pc[7, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc[0])
		# o3d.io.write_point_cloud("unprojected.ply", pcd)
		########

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.view(-1,3).unsqueeze(1)
		normals_to_project = batch_normals_projected.view(-1,3).unsqueeze(1)
		

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)
		normals_projected = torch.bmm(normals_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)

		### Debug ###
		# curr_barrel_pc = points_projected.to("cpu").detach().numpy()
		# curr_barrel_pc = np.array([curr_barrel_pc[0, :, 0], curr_barrel_pc[0, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc.T)
		# o3d.io.write_point_cloud("projected.ply", pcd)
		#########

		## Center sketch
		# print(centroid.shape)
		# print(rotation_matrices.shape)
		# print()
		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)

		# print(centroid_projected.shape)
		# print(points_projected.shape)
		points_projected -= centroid_projected
		# print(points_projected.shape)
		# exit()

		## Rescale barrel --> change this scale for training --> use gt??
		scale = torch.max(torch.sum(torch.abs(points_projected)**2, axis=-1)**0.5, dim=-1)[0]

		scales[i, :] = scale

		P_projected[i, :, :, :] = points_projected
		X_projected[i, :, :, :] = normals_projected
		###############

	scales = torch.where(found_centers_mask.T==1, scales, torch.tensor([1.0]).to(scales.device))

	return P_projected, X_projected, scales


def sketch_implicit_projection2(P, X, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)
	X_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)

	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)
	
	scales = torch.ones((K, batch_size)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)
		batch_normals_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			sampled_gt_segment_normal = torch.gather(X[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			batch_normals_projected[j, :, :] = sampled_gt_segment_normal
			found_centers_mask[j, i] = 1.0
		
		## Debug
		# curr_barrel_pc = batch_projected.to("cpu").detach().numpy()
		# # curr_barrel_pc = np.array([curr_barrel_pc[7, :, 0], curr_barrel_pc[7, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc[0])
		# o3d.io.write_point_cloud("unprojected.ply", pcd)
		########

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.view(-1,3).unsqueeze(1)
		normals_to_project = batch_normals_projected.view(-1,3).unsqueeze(1)
		

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)
		normals_projected = torch.bmm(normals_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)

		### Debug ###
		# curr_barrel_pc = points_projected.to("cpu").detach().numpy()
		# curr_barrel_pc = np.array([curr_barrel_pc[0, :, 0], curr_barrel_pc[0, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc.T)
		# o3d.io.write_point_cloud("projected.ply", pcd)
		#########

		## Center sketch
		# print(centroid.shape)
		# print(rotation_matrices.shape)
		# print()
		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)

		# print(centroid_projected.shape)
		# print(points_projected.shape)
		points_projected -= centroid_projected
		# print(points_projected.shape)
		# exit()

		## Rescale barrel --> change this scale for training --> use gt??
		scale = torch.max(torch.sum(torch.abs(points_projected)**2, axis=-1)**0.5, dim=-1)[0]

		scales[i, :] = scale

		P_projected[i, :, :, :] = points_projected
		X_projected[i, :, :, :] = normals_projected
		###############

	scales = torch.where(found_centers_mask.T==1, scales, torch.tensor([1.0]).to(scales.device))

	return P_projected, X_projected, scales, found_centers_mask


def sketch_implicit_projection3(P, X, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=8192):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, torch.tensor([1.0]).to(gt_EA_W.device), torch.tensor([1.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)
	X_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)

	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)
	
	scales = torch.ones((K, batch_size)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)
		batch_normals_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			# rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			# sampled_idx = curr_sample_pt_idx[rand_idx]
			sampled_idx = curr_sample_pt_idx

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			sampled_gt_segment_normal = torch.gather(X[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			batch_normals_projected[j, :, :] = sampled_gt_segment_normal
			found_centers_mask[j, i] = 1.0
		
		## Debug
		# curr_barrel_pc = batch_projected.to("cpu").detach().numpy()
		# # curr_barrel_pc = np.array([curr_barrel_pc[7, :, 0], curr_barrel_pc[7, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc[0])
		# o3d.io.write_point_cloud("unprojected.ply", pcd)
		########

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.view(-1,3).unsqueeze(1)
		normals_to_project = batch_normals_projected.view(-1,3).unsqueeze(1)
		

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)
		normals_projected = torch.bmm(normals_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)

		### Debug ###
		# curr_barrel_pc = points_projected.to("cpu").detach().numpy()
		# curr_barrel_pc = np.array([curr_barrel_pc[0, :, 0], curr_barrel_pc[0, :, 1], np.zeros(curr_barrel_pc.shape[1])])
		# print(curr_barrel_pc.shape)
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc.T)
		# o3d.io.write_point_cloud("projected.ply", pcd)
		#########

		## Center sketch
		# print(centroid.shape)
		# print(rotation_matrices.shape)
		# print()
		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)

		# print(centroid_projected.shape)
		# print(points_projected.shape)
		points_projected -= centroid_projected
		# print(points_projected.shape)
		# exit()

		## Rescale barrel --> change this scale for training --> use gt??
		scale = torch.max(torch.sum(torch.abs(points_projected)**2, axis=-1)**0.5, dim=-1)[0]

		scales[i, :] = scale

		P_projected[i, :, :, :] = points_projected
		X_projected[i, :, :, :] = normals_projected
		###############

	scales = torch.where(found_centers_mask.T==1, scales, torch.tensor([1.0]).to(scales.device))

	return P_projected, X_projected, scales, found_centers_mask


def project_single_gt_extrusion_instance(P, X, seg_label, bb_labels, extrusion_axes, extrusion_centers, ext_id, num_points_to_sample=1024):
	K, _ = extrusion_axes.shape
	num_points = P.shape[0]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(num_points, K)

	# print(gt_EA_W)
	# print(gt_EA_W.shape)
	# print(num_points)
	# print(K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	P_projected = torch.zeros((K, num_points_to_sample, 2)).to(P.device)
	X_projected = torch.zeros((K, num_points_to_sample, 2)).to(P.device)
	scales = torch.ones(K).to(gt_EA_W.device)

	## Sample barrel points on the segment
	curr_segment_gt_mask = gt_W_b[:,ext_id]
	indices = curr_segment_gt_mask==1
	indices = indices.nonzero().squeeze(-1)
	# print(indices.shape)
	if indices.shape[0] == 0:
		return None, None, None
	# Random sampling from gt barrel points
	rand_idx = torch.randint(0, indices.shape[0], (num_points_to_sample,))
	sampled_idx = indices[rand_idx]
	# print(sampled_idx)
	# print(sampled_idx.shape)

	sampled_gt_segment_pc = torch.gather(P, 0, sampled_idx.unsqueeze(-1).repeat(1,3))
	sampled_gt_segment_normal = torch.gather(X, 0, sampled_idx.unsqueeze(-1).repeat(1,3))

	# print(sampled_gt_segment_pc.shape)
	# print(sampled_gt_segment_normal.shape)
	# exit()


	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[i, :].unsqueeze(0)
		centroid = extrusion_centers[i, :].unsqueeze(0)
		
		# print(ax.shape)
		# print(centroid.shape)

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).to(P.device).float().unsqueeze(0)
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angle = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		# print(Z_AXIS.shape)
		# print(dot_product.shape)
		# print(angle)
		# exit()

		## Get rotation matrices for non-zero angles
		if angle > g_zero_tol:
			rot_axis = torch.cross(ax, Z_AXIS)

			# print(rot_axis.shape)

			rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle))

			rotation_matrices = rot_matrix[0, :3, :3]
		else:
			rotation_matrices = torch.eye(3).to(P.device).float()

		rotation_matrices_expanded = rotation_matrices.unsqueeze(0).repeat(num_points_to_sample, 1, 1).view(-1,3,3)

		# print(rotation_matrices_expanded.shape)
		# exit()

		points_to_project = sampled_gt_segment_pc.view(-1,3).unsqueeze(1)
		normals_to_project = sampled_gt_segment_normal.view(-1,3).unsqueeze(1)
		

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(num_points_to_sample, 2)
		normals_projected = torch.bmm(normals_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(num_points_to_sample, 2)


		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices.unsqueeze(0)).squeeze(1)[:, :2]

		# print(points_projected)
		# print(centroid_projected)
		# print(points_projected.shape)
		# print(centroid_projected.shape)
		# print(centroid.shape)
		# exit()

		points_projected -= centroid_projected

		## Rescale barrel --> change this scale for training --> use gt??
		scale = torch.max(torch.sum(torch.abs(points_projected)**2, axis=-1)**0.5, dim=-1)[0]

		scales[i] = scale

		# print(points_projected.shape)
		# print(normals_projected.shape)
		# print(scale)
		# exit()

		P_projected[i, :, :] = points_projected
		X_projected[i, :, :] = normals_projected
		###############
	# print(scales)
	return P_projected, X_projected, scales


def project_all_points_fitting_loss(P, X, extrusion_axes, extrusion_centers, extrusion_scales,  bb_labels, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	# print(extrusion_axes.shape)
	# print(extrusion_centers.shape)
	# print(extrusion_scales.shape)

	P_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(P.device)
	X_projected = torch.zeros((K, batch_size, num_points_to_sample, 2)).to(X.device)
	found_centers_mask = torch.zeros((batch_size, K)).to(X.device)

	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	# gt_W_b = torch.where(bb_labels_==0, torch.tensor([1.0]).to(P_projected.device), torch.tensor([0.0]).to(P_projected.device))
	gt_W_b = torch.where(bb_labels_==0, torch.tensor([1.0]).to(P_projected.device), torch.tensor([1.0]).to(P_projected.device))

	### NOTE!!!
	### What to do with base points....

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]
		scale = extrusion_scales[i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)
		batch_normals_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			#sampled_idx = curr_sample_pt_idx[rand_idx]
			sampled_idx = curr_sample_pt_idx

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			sampled_gt_segment_normal = torch.gather(X[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			batch_normals_projected[j, :, :] = sampled_gt_segment_normal
			found_centers_mask[j, i] = 1.0
	

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.reshape(-1,3).unsqueeze(1)
		normals_to_project = batch_normals_projected.reshape(-1,3).unsqueeze(1)

		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)
		normals_projected = torch.bmm(normals_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)


		## Center sketch
		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)
		points_projected -= centroid_projected

		## Rescale barrel --> change this scale for training --> use gt??
		# print(scale.shape)
		# print(points_projected.shape)
		#points_projected /= scale.unsqueeze(-1).unsqueeze(-1)
		points_projected /= scale.unsqueeze(-1)

		P_projected[i, :, :, :] = points_projected
		X_projected[i, :, :, :] = normals_projected
		###############

	return P_projected, X_projected, found_centers_mask


## Get min max of projected along the extrusion axis
def get_extrusion_extents(P, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))


	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)	
	extents = torch.zeros((K, batch_size, 2)).to(gt_EA_W.device) ## get min max ranges along the extrusion axis

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			found_centers_mask[j, i] = 1.0
		
		centroid_g = centroid.unsqueeze(1).repeat(1, num_points_to_sample, 1)
		points_centered = batch_projected - centroid_g

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points_to_sample)

		# print(dist)
		# print(dist.shape)

		min_dist, _ = torch.min(dist, axis=-1)
		max_dist, _ = torch.max(dist, axis=-1)

		# print(min_dist)
		# print(max_dist)
		extents[i, :, 0] = min_dist
		extents[i, :, 1] = max_dist

		# exit()

	return extents, found_centers_mask

#############

def output_pc_sketch(fname, pc):
	import open3d as o3d
	curr_barrel_pc = pc.to("cpu").detach().numpy()
	curr_barrel_pc = np.array([curr_barrel_pc[ :, 0], curr_barrel_pc[ :, 1], np.zeros(curr_barrel_pc.shape[0])])
	print(curr_barrel_pc.shape)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(curr_barrel_pc.T)
	o3d.io.write_point_cloud(fname, pcd)
	print("Done " + fname)

def visualize_segmentation_pc(model_id, output_folder, pc, pred_label, gt_label, filehandle_1, filehandle_2):
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save pred point ids.
	pred_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_ids.txt')
	np.savetxt(pred_point_ids_file, pred_label, fmt='%d')
	print("Saved '{}'.".format(pred_point_ids_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Render pred point_cloud.
	pred_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_pred')
	render_point_cloud(out_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Save filename to combine
	line = pred_snapshot_file+".png" + " " + gt_snapshot_file+".png" + "\n"
	filehandle_2.write(line)

def visualize_sketch(model_id, output_folder, pred_sk, gt_sk, pred_label, gt_label, filehandle_1, filehandle_2):
	### Point Cloud ###
	# Save point cloud.
	pred_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_points.xyz')
	np.savetxt(pred_point_cloud_file, pred_sk, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(pred_point_cloud_file))

	gt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_points.xyz')
	np.savetxt(gt_point_cloud_file, gt_sk, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(gt_point_cloud_file))

	# Save pred point ids.
	pred_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_ids.txt')
	np.savetxt(pred_point_ids_file, pred_label, fmt='%d')
	print("Saved '{}'.".format(pred_point_ids_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Render pred point_cloud.
	pred_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_pred')
	render_sketch(pred_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_sketch(gt_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Save filename to combine
	line = pred_snapshot_file+".png" + " " + gt_snapshot_file+".png" + "\n"
	filehandle_2.write(line)

def visualize_single_sketch(model_id, output_folder, gt_sk, gt_label, filehandle_1, filehandle_2):
	### Point Cloud ###
	gt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_points.xyz')
	np.savetxt(gt_point_cloud_file, gt_sk, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(gt_point_cloud_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_sketch(gt_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1, adjust_camera=False)

	# Save filename to combine
	line = pred_snapshot_file+".png" + " " + gt_snapshot_file+".png" + "\n"
	filehandle_2.write(line)

def visualize_segmentation_pc_bb(model_id, output_folder, pc, pred_label, gt_label, pred_bb_label, filehandle_1, filehandle_2):
	###
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save pred point ids.
	pred_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_ids.txt')
	np.savetxt(pred_point_ids_file, pred_label, fmt='%d')
	print("Saved '{}'.".format(pred_point_ids_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Save base and barrel points
	pred_bb_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_bb_ids.txt')
	np.savetxt(pred_bb_ids_file, pred_bb_label, fmt='%d')
	print("Saved '{}'.".format(pred_bb_ids_file))	

	# Render pred point_cloud.
	pred_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_pred')
	render_point_cloud(out_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render base or barrel 
	bb_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_pred_bb')
	render_point_cloud(out_point_cloud_file, pred_bb_ids_file,
			bb_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	# Save filename to combine
	line = pred_snapshot_file + ".png" + " " + gt_snapshot_file + ".png" + " " + bb_snapshot_file + ".png"+ "\n"
	filehandle_2.write(line)

	return

def visualize_segmentation_pc_bb_v2(model_id, output_folder, pc, pred_label, gt_label, pred_bb_label, gt_bb_labels, filehandle_1, filehandle_2):
	###
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save pred point ids.
	pred_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_ids.txt')
	np.savetxt(pred_point_ids_file, pred_label, fmt='%d')
	print("Saved '{}'.".format(pred_point_ids_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Save base and barrel points
	pred_bb_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_bb_ids.txt')
	np.savetxt(pred_bb_ids_file, pred_bb_label, fmt='%d')
	print("Saved '{}'.".format(pred_bb_ids_file))	

	# Save base and barrel points
	gt_bb_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_bb_ids.txt')
	np.savetxt(gt_bb_ids_file, gt_bb_labels, fmt='%d')
	print("Saved '{}'.".format(gt_bb_ids_file))		

	# Render pred point_cloud.
	pred_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_pred')
	render_point_cloud(out_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1)
	render_point_cloud(out_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1, default_angle=1)	

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1)
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1, default_angle=1)
	
	# Render base or barrel 
	pred_bb_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_pred_bb')
	render_point_cloud(out_point_cloud_file, pred_bb_ids_file,
			pred_bb_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	gt_bb_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_gt_bb')
	render_point_cloud(out_point_cloud_file, gt_bb_ids_file,
			gt_bb_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	# Save filename to combine
	line = pred_snapshot_file + ".png" + " " + gt_snapshot_file + ".png" + " " + pred_bb_snapshot_file + ".png"+ " " + gt_bb_snapshot_file + ".png"+ "\n"
	filehandle_2.write(line)

	return

def visualize_segmentation_pc_bb_op(model_id, output_folder, pc, pred_label, gt_label, pred_bb_label, gt_bb_labels, pred_op_label, gt_op_labels, filehandle_1, filehandle_2):
	###
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save pred point ids.
	pred_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_ids.txt')
	np.savetxt(pred_point_ids_file, pred_label, fmt='%d')
	print("Saved '{}'.".format(pred_point_ids_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Save base and barrel points
	pred_bb_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_bb_ids.txt')
	np.savetxt(pred_bb_ids_file, pred_bb_label, fmt='%d')
	print("Saved '{}'.".format(pred_bb_ids_file))	

	# Save base and barrel points
	gt_bb_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_bb_ids.txt')
	np.savetxt(gt_bb_ids_file, gt_bb_labels, fmt='%d')
	print("Saved '{}'.".format(gt_bb_ids_file))		

	# Save op points
	pred_op_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_pred_op_ids.txt')
	np.savetxt(pred_op_ids_file, pred_op_label, fmt='%d')
	print("Saved '{}'.".format(pred_op_ids_file))	

	# Save op points
	gt_op_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_op_ids.txt')
	np.savetxt(gt_op_ids_file, gt_op_labels, fmt='%d')
	print("Saved '{}'.".format(gt_op_ids_file))	


	# Render pred point_cloud.
	pred_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_pred')
	render_point_cloud(out_point_cloud_file, pred_point_ids_file,
			pred_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_points_gt')
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Render base or barrel 
	pred_bb_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_pred_bb')
	render_point_cloud(out_point_cloud_file, pred_bb_ids_file,
			pred_bb_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	gt_bb_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_gt_bb')
	render_point_cloud(out_point_cloud_file, gt_bb_ids_file,
			gt_bb_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	# Render op 
	pred_op_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_pred_op')
	render_point_cloud(out_point_cloud_file, pred_op_ids_file,
			pred_op_snapshot_file, outfile=True, filehandle_=filehandle_1)	

	gt_op_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_gt_op')
	render_point_cloud(out_point_cloud_file, gt_op_ids_file,
			gt_op_snapshot_file, outfile=True, filehandle_=filehandle_1)

	# Save filename to combine
	line = pred_snapshot_file + ".png" + " " + gt_snapshot_file + ".png" + " " + pred_bb_snapshot_file + ".png"+ " " + gt_bb_snapshot_file + ".png"+ " "+ pred_op_snapshot_file + ".png"+ " " + gt_op_snapshot_file + ".png"+ "\n"
	filehandle_2.write(line)

	return

## Currently used to debug extrusion axis prediction from gt
def visualize_pc(model_id, output_folder, pc, gt_label, filehandle):
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save gt point ids.
	gt_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_gt_ids.txt')
	np.savetxt(gt_point_ids_file, gt_label, fmt='%d')
	print("Saved '{}'.".format(gt_point_ids_file))

	# Render gt point_cloud.
	gt_snapshot_file = os.path.join(output_folder, 'debug_extrusion_gt', model_id)
	render_point_cloud(out_point_cloud_file, gt_point_ids_file,
			gt_snapshot_file, outfile=True, filehandle_=filehandle)


########################################
######## For post-processing ###########
#########################################

def scale_ransac(P, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)
	
	scales = torch.ones((K, batch_size)).to(gt_EA_W.device)

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			found_centers_mask[j, i] = 1.0

		''' 
		Rotate extrusion axis to align with z axis
		Project to 2D coordinate by removing the z-value
		'''

		## angle between ext_axis and z-axis
		Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).repeat(batch_size,1).to(P.device).float()
		dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
		angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)

		rotation_matrices = torch.eye(3).to(P.device).float().reshape(1,3,3).repeat(batch_size, 1, 1)

		## Get rotation matrices for non-zero angles
		for a in range(batch_size):
			angle = angles[a]
			if angle > g_zero_tol:
				rot_axis = torch.cross(ax[a], Z_AXIS[a])

				rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))

				rotation_matrices[a, :, :] = rot_matrix[0, :3, :3]

		rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, num_points_to_sample, 1, 1).view(-1,3,3)

		points_to_project = batch_projected.view(-1,3).unsqueeze(1)


		points_projected = torch.bmm(points_to_project, rotation_matrices_expanded).squeeze()[:, :2].reshape(batch_size, num_points_to_sample, 2)


		centroid_projected = torch.bmm(centroid.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2].unsqueeze(1)
		points_projected -= centroid_projected

		#### Ransac ###
		points_projected = points_projected.squeeze().to("cpu").detach().numpy()

		SMALL_PERCENT = 0.01
		AGREEMENT_PERCENT_THRESH = 0.8

		small_point_sample = int(SMALL_PERCENT * num_points_to_sample)
		num_iterations = 1000
		best_agreement_percentage = -1
		best_scale = 1

		for it in range(num_iterations):
			idx = np.arange(points_projected.shape[0])
			np.random.shuffle(idx)

			sampled = points_projected[idx[:small_point_sample]]
			curr_scale = np.max(np.sum(np.abs(sampled)**2, axis=-1)**0.5, axis=-1)

			### Check how many points are in agreement
			all_norms = np.sum(np.abs(points_projected)**2, axis=-1)**0.5
			num_agreed = np.sum(all_norms<curr_scale)
			percent_agreed = num_agreed/num_points_to_sample

			if percent_agreed > AGREEMENT_PERCENT_THRESH:
				best_scale = curr_scale
				break

			best_agreement_percentage = percent_agreed
			best_scale = curr_scale

		scales[i, :] = torch.from_numpy(np.array(best_scale)).to(scales.device)

		###############

	scales = torch.where(found_centers_mask.T==1, scales, torch.tensor([1.0]).to(scales.device))

	return scales

def extents_clustering(P, seg_label, bb_labels, extrusion_axes, extrusion_centers, num_points_to_sample=1024):
	batch_size, K, _ = extrusion_axes.shape
	num_points = P.shape[1]

	exlabel_ = seg_label.view(-1)
	gt_EA_W = F.one_hot(exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	bb_labels_ = bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device))

	found_centers_mask = torch.zeros((batch_size, K)).to(gt_EA_W.device)	
	extents = np.zeros((K, batch_size, 2)) ## get min max ranges along the extrusion axis

	## Project all points onto plane defined by gt axis and center
	for i in range(K):
		ax = extrusion_axes[:, i, :]
		centroid = extrusion_centers[:, i, :]

		curr_segment_gt_mask = gt_W_b[:,:,i]
		indices = curr_segment_gt_mask==1

		if (indices.nonzero().shape[0]<=1):
			found_centers_mask[:,i] = 0.0
			continue

		indices = indices.nonzero().squeeze()

		batch_projected = torch.zeros((batch_size, num_points_to_sample, 3)).to(P.device)

		## Find segments to project
		for j in range(batch_size):
			## For gt
			curr_sample_indices = indices[:,0]==j

			## No points found in segment (1 point found is considered no points to handle .squeeze() function)
			if (curr_sample_indices.nonzero().shape[0]<=1):
				found_centers_mask[j,i] = 0.0
				continue

			curr_sample_indices = curr_sample_indices.nonzero().squeeze()
			curr_sample_pt_idx = indices[:,1][curr_sample_indices]

			# Random sampling from gt barrel points
			rand_idx = torch.randint(0, curr_sample_pt_idx.shape[0], (num_points_to_sample,))
			sampled_idx = curr_sample_pt_idx[rand_idx]

			sampled_gt_segment_pc = torch.gather(P[j,:,:], 0, sampled_idx.unsqueeze(-1).repeat(1,3))
			
			batch_projected[j, :, :] = sampled_gt_segment_pc
			found_centers_mask[j, i] = 1.0
		
		centroid_g = centroid.unsqueeze(1).repeat(1, num_points_to_sample, 1)
		points_centered = batch_projected - centroid_g

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points_to_sample, 1) # (B, N, 3)
		points_centered = points_centered.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

		points_centered = points_centered.unsqueeze(1)	#(B*N, 1, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dist = torch.bmm(points_centered, ax_expanded_collapsed)
		dist = dist.view(-1, num_points_to_sample)

		#### Clustering here instead of using min/max
		dist = dist.squeeze().unsqueeze(-1).to("cpu").detach().numpy()
		# print(dist)
		# print(dist.shape)

		### DBSCAN
		min_sample = int(0.5*num_points_to_sample)
		db = DBSCAN(eps=0.05, min_samples=min_sample).fit(dist)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		# print(n_clusters_)

		dominant_cluster = np.bincount(labels+1).argmax()
		mask_idx = labels == (dominant_cluster-1)
		dist = dist[mask_idx]
		min_dist = np.min(dist)
		max_dist = np.max(dist)
		###########

		extents[i, :, 0] = min_dist
		extents[i, :, 1] = max_dist


	return extents, found_centers_mask


###################################
######## For Visualizer ###########
###################################

### Utils for marching cube
def compute_grid2D(shape2D, ranges=((0., 1.), (0., 1.)), flatten=True):
    x_dim = shape2D[1]
    y_dim = shape2D[0]
    x_range = ranges[0][0] - ranges[0][1]; y_range = ranges[1][0] - ranges[1][1]
    x_lin = np.linspace(ranges[0][0], ranges[0][1], x_dim, endpoint=False) + x_range / x_dim * 0.5
    y_lin = np.linspace(ranges[1][0], ranges[1][1], y_dim, endpoint=False) + y_range / y_dim * 0.5
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    if not flatten:
        return x_grid, y_grid
    x_t = torch.from_numpy(x_grid).cuda().float()
    y_t = torch.from_numpy(y_grid).cuda().float()
    x_flat = torch.reshape(x_t, [x_dim*y_dim, 1])
    y_flat = torch.reshape(y_t, [x_dim*y_dim, 1])
    xy = torch.cat((x_flat, y_flat), 1)
    return xy

### Marching Cubes function
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 2]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 0]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

    print('Marching cubes took: {}'.format(time.time() - start_time))


def get_visualizer_rotation_matrix(ax, device):
	Z_AXIS = torch.from_numpy(np.array([0.0, 0.0, 1.0])).unsqueeze(0).to(device).float()
	dot_product = torch.bmm(ax.unsqueeze(1), Z_AXIS.unsqueeze(-1))
	angles = torch.acos(dot_product).squeeze(-1).squeeze(-1)
	rotation_matrices = torch.eye(3).to(device).float().reshape(1,3,3)

	## Get rotation matrices for non-zero angles
	angle = angles[0]
	if angle > g_zero_tol:
		rot_axis = torch.cross(ax[0], Z_AXIS[0])
		rot_matrix = tgm.angle_axis_to_rotation_matrix((rot_axis * angle).unsqueeze(0))
		rotation_matrices[0, :, :] = rot_matrix[0, :3, :3]	

	return rotation_matrices

def transform_to_sketch_plane(xyz_coord, rotation_matrices, c, scale):
	rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(1, xyz_coord.shape[1], 1, 1).view(-1,3,3)
	xyz_coord_reshape = xyz_coord.reshape(-1,3).unsqueeze(1)
	xyz_coord_projected = torch.bmm(xyz_coord_reshape, rotation_matrices_expanded).squeeze()[:, :2].reshape(1, xyz_coord.shape[1], 2)

	centroid_projected = torch.bmm(c.unsqueeze(1), rotation_matrices).squeeze(1)[:, :2]
	xyz_coord_projected -= centroid_projected
	xyz_coord_projected /= scale

	return xyz_coord_projected

def get_distances_on_extrusion_axis(xyz_coord, ax, c):
	centroid_g = c.unsqueeze(1).repeat(1, xyz_coord.shape[1], 1)
	xyz_coord_centered = xyz_coord - centroid_g

	ax_expanded = ax.unsqueeze(1).repeat(1, xyz_coord.shape[1], 1) # (B, N, 3)
	xyz_coord_centered = xyz_coord_centered.view(-1,3) # (B*N, 3)
	ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)

	xyz_coord_centered = xyz_coord_centered.unsqueeze(1)	#(B*N, 1, 3)
	ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

	dist = torch.bmm(xyz_coord_centered, ax_expanded_collapsed)
	dist = dist.view(-1, xyz_coord.shape[1])

	return dist