# Mikaela Uy (mikacuy@cs.stanford.edu)
import os, sys
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, '..','data_preprocessing'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from global_variables import *

from chamferdist import ChamferDistance
chamferDist = ChamferDistance()

TORCH_PI = torch.acos(torch.zeros(1)).item() * 2

######################################
###### For Extrusion Segmentation
######################################
def hungarian_matching(W_pred, I_gt, with_mask=False):
	# This non-tf function does not backprob gradient, only output matching indices
	# W_pred - BxNxK
	# I_gt - BxN, may contain -1's
	# Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
	#   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance

	batch_size, n_points, n_max_labels = W_pred.size()
	matching_indices = torch.zeros([batch_size, n_max_labels], dtype=torch.long).to(W_pred.device)
	mask = torch.zeros([batch_size, n_max_labels], dtype=torch.bool).to(W_pred.device)

	for b in range(batch_size):
		# assuming I_gt[b] does not have gap
		# n_gt_labels = torch.max(I_gt[b]).item()  # this is K'
		n_gt_labels = torch.max(I_gt[b]).item() + 1  # this is K'

		W_gt = torch.eye(n_gt_labels+1).to(I_gt.device)[I_gt[b]]
		dot = torch.mm(W_gt.transpose(0,1), W_pred[b])
		denominator = torch.sum(W_gt, dim=0).unsqueeze(1) + torch.sum(W_pred[b], dim=0).unsqueeze(0) - dot
		cost = dot / torch.clamp(denominator, min=1e-10, max=None)  # K'xK
		cost = cost[:n_gt_labels, :]  # remove last row, corresponding to matching gt background instance
		_, col_ind = linear_sum_assignment(-cost.detach().cpu().numpy())  # want max solution
		col_ind = torch.from_numpy(col_ind).long().to(matching_indices.device)
		matching_indices[b, :n_gt_labels] = col_ind

		mask[b, :n_gt_labels] = True

	if not with_mask:
		return matching_indices
	else:
		return matching_indices, mask

# Converting W to hard encoding
def hard_W_encoding(W, to_null_mask=False, W_null_threshold = 0.005):
	# W - BxNxK
	_, n_points, n_max_instances = W.size()

	hardW = torch.eye(n_max_instances).to(W.device)[torch.argmax(W, dim=2)].float()

	if to_null_mask:
		W_column_sum = torch.sum(W, axis = 1)
		null_mask = torch.where(W_column_sum < (float(n_points) * W_null_threshold), torch.tensor([1.0]).to(W.device), torch.tensor([0.0]).to(W.device))
		null_mask_W_like = null_mask.unsqueeze(1)
		null_mask_W_like = null_mask_W_like.repeat(1, n_points, 1)
		hardW = hardW * (1.0 - null_mask_W_like)

	return hardW

def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = lengths.unsqueeze(dim=-1)
    mask = row_vector < matrix
    return mask

def get_mask_gt(I_gt, n_max_instances):
    n_instances_gt = torch.max(I_gt, dim=1)[0] + 1  # only count known primitive type instances, as -1 will be ignored
    mask_gt = sequence_mask(n_instances_gt, maxlen=n_max_instances)
    return mask_gt

def reduce_mean_masked_instance(loss, mask_gt):
    # loss: BxK
    loss = torch.where(mask_gt, loss, torch.zeros_like(loss))
    reduced_loss = torch.sum(loss, axis=1) # B
    denom = torch.sum(mask_gt.float(), dim=1) # B
    return torch.where(denom > 0, reduced_loss / denom, torch.zeros_like(reduced_loss)) # B

def compute_miou_loss(W, I_gt, matching_indices, div_eps=1e-10):
	# W - BxNxK
	# I_gt - BxN
	batch_size, n_points, n_max_labels = W.size() 
	_, n_labels = matching_indices.size()
	W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(batch_size, n_points, n_labels)) # BxNxK
	# notice in tf.one_hot, -1 will result in a zero row, which is what we want
	W_gt = torch.eye(n_labels+2).to(I_gt.device)[I_gt]
	W_gt = W_gt[:,:,:n_labels]
	dot = torch.sum(W_gt * W_reordered, axis=1) # BxK
	denominator = torch.sum(W_gt, dim=1) + torch.sum(W_reordered, dim=1) - dot
	mIoU = dot / (denominator + div_eps) # BxK

	return 1.0 - mIoU, 1 - dot / n_points, W_reordered

# Segmentation mIoU
def compute_segmentation_iou(W, I_gt, matching_indices, mask):# W - BxNxK
	mIoU = 1 - compute_miou_loss(W, I_gt, matching_indices)[0]
	mIoU = torch.sum(mask * mIoU, dim=1) / torch.sum(mask, dim=1)
	return mIoU

def compute_weighted_segmentation_iou(W, I_gt, matching_indices, mask, weights):# W - BxNxK
	batch_size, n_points, n_max_labels = W.size()
	mIoU = 1 - compute_miou_loss(W, I_gt, matching_indices)[0]
	weighted_mIoU = (mIoU * weights) / float(n_points)
	reduced_loss = torch.sum(weighted_mIoU, axis=1) # B 

	return reduced_loss
#######################################

######################################
##### For Normal loss
######################################
def acos_safe(x):
	return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))
		

def compute_normal_loss(normal, normal_gt, angle_diff, collapse=True):
	# normal, normal_gt: BxNx3
	# Assume normals are unoriented
	dot_abs = torch.abs(torch.sum(normal * normal_gt, dim=2)) # BxN
	# Use oriented for rebuttal 
	# dot_abs = torch.sum(normal * normal_gt, dim=2) # BxN
	if angle_diff:
		if collapse:
			return torch.mean(acos_safe(dot_abs), dim=1)
		else:
			return acos_safe(dot_abs)

	else:
		if collapse:
			return torch.mean(1.0 - dot_abs, dim=1)
		else:
			return 1.0 - dot_abs


def compute_normal_difference(X, X_gt, in_radians=True, collapse=True):
	if in_radians:
		if collapse:
			normal_difference = torch.mean(acos_safe(torch.abs(torch.sum(X*X_gt, dim=2))), dim=1)
		else:
			normal_difference = acos_safe(torch.abs(torch.sum(X*X_gt, dim=2)))

	else:
		if collapse:
			normal_difference = torch.mean(acos_safe(torch.abs(torch.sum(X*X_gt, dim=2))) * 180.0/TORCH_PI, dim=1)
		else:
			normal_difference = acos_safe(torch.abs(torch.sum(X*X_gt, dim=2))) * 180.0/TORCH_PI

	return normal_difference
#######################################

######################################
##### For Sketch loss
######################################
def get_sketch_loss(projected_pcs, gt_projected_pcs):
	## gt_projected_pcs = (K,B,N,3)
	K, batch_size, num_points, _ = gt_projected_pcs.shape

	num_points_in_seg = (torch.square(gt_projected_pcs).sum(-1) != 0).sum(dim=-1) # count number of points in segment (K,B)
	# num_points_in_seg = num_points_in_seg.unsqueeze(-1).repeat(1,1,3).unsqueeze(-2).repeat(1,1,num_points,1)

	sketch_loss = torch.div(torch.square(gt_projected_pcs - projected_pcs).sum(dim=-1).sum(dim=-1), num_points_in_seg + g_zero_tol) # to avoid nan in empty segment
	sketch_loss = sketch_loss.T #(B,K)

	return sketch_loss

def get_sketch_loss_v2(projected_pcs, gt_projected_pcs, gt_bb_labels, gt_extrusion_instances):
	## gt_projected_pcs = (K,B,N,3)
	K, batch_size, num_points, _ = gt_projected_pcs.shape

	### For masking weights: only compute different with the gt barrel points of each extrusion
	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device)) #(B,N,K)
	#####

	point_distances = torch.square(gt_projected_pcs - projected_pcs).sum(dim=-1) #(K,B,N)
	point_distances = torch.transpose(point_distances, 0,1) #(B,K,N)
	point_distances = torch.transpose(point_distances, 1,2) #(B,N,K)

	num_points_in_seg = (gt_W_b != 0).sum(dim=1) # count number of points in segment (K,B)
	
	# print(num_points_in_seg)
	# print(num_points_in_seg.shape)

	masked_point_distances = point_distances*gt_W_b
	# print(masked_point_distances)
	# print(masked_point_distances.shape)

	sketch_loss = torch.div(masked_point_distances.sum(dim=1), num_points_in_seg + g_zero_tol) # to avoid nan in empty segment
	# print(sketch_loss)
	# print(sketch_loss.shape)
	# exit()

	return sketch_loss

### Weighted Chamfer loss for sketches
def get_weighted_cd_loss(P_projected, gt_projected, P_soft_projected, W_barrel, multiplier=10.0):
	## Unweighted chamfer loss
	P_projected_reshaped = P_projected.view(-1, P_projected.shape[-2], P_projected.shape[-1])
	gt_projected_reshaped = gt_projected.view(-1, gt_projected.shape[-2], gt_projected.shape[-1])
	P_soft_projected_reshaped = P_soft_projected.view(-1, P_soft_projected.shape[-2], P_soft_projected.shape[-1])

	dist_forward= chamferDist(P_projected_reshaped, gt_projected_reshaped)	
	dist_backward= chamferDist(gt_projected_reshaped, P_soft_projected_reshaped)	
	dist_forward = dist_forward.view(P_projected.shape[0], P_projected.shape[1], P_projected.shape[2])
	dist_backward = dist_backward.view(gt_projected.shape[0], gt_projected.shape[1], gt_projected.shape[2])

	## Weight dist_forward (projected-to-ground_truth)
	dist_forward = dist_forward.permute(1,2,0)
	dist_backward = dist_backward.permute(1,2,0)

	cd_loss_forward = torch.mean(dist_forward*W_barrel, dim=1) * multiplier # Rescale
	cd_loss_backward = torch.mean(dist_backward, dim=1) * multiplier/2.

	return cd_loss_forward, cd_loss_backward

def get_cd_loss_evaluation(A_projected, B_projected):
	## Unweighted chamfer loss
	A_projected_reshaped = A_projected.reshape(-1, A_projected.shape[-2], A_projected.shape[-1])
	B_projected_reshaped = B_projected.reshape(-1, B_projected.shape[-2], B_projected.shape[-1])

	dist_forward= chamferDist(A_projected_reshaped, B_projected_reshaped)	
	dist_forward = dist_forward.view(A_projected.shape[0], A_projected.shape[1], A_projected.shape[2])

	## Weight dist_forward (projected-to-ground_truth)
	dist_forward = dist_forward.permute(1,2,0)

	cd_loss_forward = torch.mean(dist_forward, dim=1)

	return cd_loss_forward

#######################################

#####################################################
##### For Extrusion Axis - Normal Regularization loss
#####################################################

def axis_normal_regularization_loss(X, E_AX, gt_bb_labels, gt_extrusion_instances):
	K = E_AX.shape[1]
	batch_size, num_points, _ = X.shape

	### For masking weights: only compute different with the gt barrel points of each extrusion
	gt_exlabel_ = gt_extrusion_instances.view(-1)
	gt_EA_W = F.one_hot(gt_exlabel_, num_classes=K)
	gt_EA_W = gt_EA_W.view(batch_size, num_points, K)

	# Get barrel points
	gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
	gt_W_b = torch.where(gt_bb_labels_==0, gt_EA_W.float(), torch.tensor([0.0]).to(gt_EA_W.device)) #(B,N,K)
	#####

	# print(gt_W_b.shape) # (B, N, K)
	# print(X.shape)		# (B, N, 3)
	# print(E_AX.shape)	# (B, K, 3)

	all_dot_products = torch.zeros((K, batch_size, num_points)).to(X.device)
	
	X = X.reshape(-1, 3)
	X = X.unsqueeze(1) #(B*N, 1, 3)

	for i in range(K):
		ax = E_AX[:, i, :]

		ax_expanded = ax.unsqueeze(1).repeat(1, num_points, 1) # (B, N, 3)
		ax_expanded_collapsed = ax_expanded.view(-1,3) # (B*N, 3)
		ax_expanded_collapsed = ax_expanded_collapsed.unsqueeze(2)	#(B*N, 3, 1)

		dot_prods = torch.bmm(X, ax_expanded_collapsed)
		dot_prods = dot_prods.view(-1, num_points) # (B, N)

		all_dot_products[i, :, :] = dot_prods

	all_dot_products = torch.abs(all_dot_products)
	all_dot_products = torch.transpose(all_dot_products, 0, 1)
	all_dot_products = torch.transpose(all_dot_products, 1, 2) # (B, N, K)

	# print(all_dot_products)
	# print(all_dot_products.shape)

	## Barrels
	barrel_dots = gt_W_b * all_dot_products
	## Base
	base_dots = (1-gt_W_b)*all_dot_products

	# print(barrel_dots)
	# print(barrel_dots.shape)
	# print(base_dots)
	# print(base_dots.shape)

	ax_reg_loss = barrel_dots - base_dots

	ax_reg_loss = ax_reg_loss.mean(dim=1) # (B, K)
	# print(ax_reg_loss)
	# print(ax_reg_loss.shape)
	# exit()

	return ax_reg_loss


#######################################

def compute_all_losses(P, W, I_gt, X, X_gt,
                       normal_loss_multiplier, miou_loss_multiplier, return_match_indices=False, collapse=True):
	batch_size, _, n_max_instances = W.size()

	mask_gt = get_mask_gt(I_gt, n_max_instances)
	if normal_loss_multiplier>0:
		normal_loss = compute_normal_loss(X, X_gt, angle_diff=False)
	else:
		normal_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)

	if miou_loss_multiplier > 0:
		matching_indices, mask = hungarian_matching(W, I_gt, with_mask=True)
		miou_loss, _, _ = compute_miou_loss(W, I_gt, matching_indices)
		avg_miou_loss = reduce_mean_masked_instance(miou_loss, mask_gt)
	else:
		avg_miou_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)

	if collapse:
		total_miou_loss = torch.mean(avg_miou_loss)
		total_normal_loss = torch.mean(normal_loss)

		total_loss = 0
		total_loss += miou_loss_multiplier * total_miou_loss
		total_loss += normal_loss_multiplier * total_normal_loss

	else:
		total_miou_loss = avg_miou_loss
		total_normal_loss = normal_loss

		total_loss	= 	miou_loss_multiplier * total_miou_loss + normal_loss_multiplier * total_normal_loss

	if return_match_indices:
		return total_loss, total_normal_loss, total_miou_loss, matching_indices, mask
	else:
		return total_loss, total_normal_loss, total_miou_loss




