# Mikaela Uy (mikacuy@cs.stanford.edu)
import os, sys
import json
import numpy as np
from torch.utils.data import Dataset
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, '..','data_preprocessing'))
from global_variables import *
from utils import *
from data_utils import *

class AutodeskDataset_h5(Dataset):
	def __init__(self, filename, num_points, max_instances, op=False, center=False, extent=False):
		if not extent:
			if op and not center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations = load_h5(filename, op=True)
			elif not op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_centers = load_h5(filename, center=True)
			elif op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, extrusion_centers = load_h5(filename, op=True, center=True)
			else:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances = load_h5(filename)
		else:
			if op and not center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, extrusion_extents = load_h5(filename, op=True, extent=True)
			elif not op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_centers, extrusion_extents = load_h5(filename, center=True, extent=True)
			elif op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, extrusion_centers, extrusion_extents = load_h5(filename, op=True, center=True, extent=True)
			else:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_extents = load_h5(filename, extent=True)			

		self.pcs = point_cloud
		self.normals = normals
		self.extrusion_labels = extrusion_labels

		self.extrusion_distances = extrusion_distances
		self.extrusion_axes = extrusion_axes
		self.bb_labels = bb_labels
		self.n_samples = self.pcs.shape[0]
		self.n_instances = n_instances

		self.npoints = num_points
		self.K = max_instances

		## Includes extrusion operation label (add=0, cut=1, intersect=3)
		self.op = op
		if op:
			self.operations = extrusion_operations

		self.center = center
		if center:
			self.extrusion_centers = extrusion_centers

		self.extent = extent
		if extent:
			self.extrusion_extents = extrusion_extents		

		print(np.max(self.n_instances))
		if (np.max(self.n_instances) != self.K):
			print("WARNING. K= " + str(self.K) + ", max_instance in data= " + str(np.max(self.n_instances)))

		print("Number of samples: "+str(self.n_samples))


	def __getitem__(self, index):
		### Shuffle and sample point cloud
		idx = torch.randperm(self.pcs.shape[1])
		if (self.pcs.shape[1] < self.npoints):
			print("ERROR. Sampling more points than point cloud resolution.")

		selected_idx = idx[:self.npoints]
		# print(selected_idx)

		sampled_pcs = self.pcs[index][selected_idx,:]
		sampled_normals = self.normals[index][selected_idx,:]
		sampled_extrusion_labels = self.extrusion_labels[index][selected_idx]
		sampled_bb_labels = self.bb_labels[index][selected_idx]

		per_point_extrusion_axes = self.extrusion_axes[index][sampled_extrusion_labels]
		per_point_extrusion_distances = self.extrusion_distances[index][sampled_extrusion_labels]

		extrusion_axes = self.extrusion_axes[index][:self.K]
		extrusion_distances = self.extrusion_distances[index][:self.K]

		if not self.extent:
			if not self.op and not self.center:
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances
			elif not self.op and self.center:
				sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers
			elif self.op and not self.center:
				sampled_extrusion_op = self.operations[index][selected_idx]
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op					
			else:
				sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
				sampled_extrusion_op = self.operations[index][selected_idx]
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers
		else:
			sampled_extrusion_extents = self.extrusion_extents[index][:self.K]			

			if not self.op and not self.center:
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_extents
			elif not self.op and self.center:
				sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers, sampled_extrusion_extents
			elif self.op and not self.center:
				sampled_extrusion_op = self.operations[index][selected_idx]
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_extents					
			else:
				sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
				sampled_extrusion_op = self.operations[index][selected_idx]
				return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
						per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers, sampled_extrusion_extents

	def __len__(self):
		return self.n_samples

class AutodeskDataset_h5_sketches(Dataset):
	def __init__(self, filename, num_points, num_sk_points, max_instances, op=False, center=False, with_scale=False, extent=False):

		if not extent:
			if op and not center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, sketches, sketches_norm_factors = load_h5_sk(filename, op=True)
			elif not op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_centers, sketches, sketches_norm_factors = load_h5_sk(filename, center=True)
			elif op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, extrusion_centers, sketches, sketches_norm_factors = load_h5_sk(filename, op=True, center=True)
			else:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, sketches, sketches_norm_factors = load_h5_sk(filename)
		else:
			if op and not center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, sketches, sketches_norm_factors, extrusion_extents = load_h5_sk(filename, op=True, extent=True)
			elif not op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_centers, sketches, sketches_norm_factors, extrusion_extents = load_h5_sk(filename, center=True, extent=True)
			elif op and center:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations, extrusion_centers, sketches, sketches_norm_factors, extrusion_extents = load_h5_sk(filename, op=True, center=True, extent=True)
			else:
				point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, sketches, sketches_norm_factors, extrusion_extents = load_h5_sk(filename, extent=True)			

		self.pcs = point_cloud
		self.normals = normals
		self.extrusion_labels = extrusion_labels

		self.extrusion_distances = extrusion_distances
		self.extrusion_axes = extrusion_axes
		self.bb_labels = bb_labels
		self.n_samples = self.pcs.shape[0]
		self.n_instances = n_instances

		self.npoints = num_points
		self.K = max_instances

		## For sketches
		self.sketches = sketches
		self.num_sk_points = num_sk_points

		self.with_scale = with_scale
		self.sk_norm_factors = sketches_norm_factors

		## Includes extrusion operation label (add=0, cut=1, intersect=3)
		self.op = op
		if op:
			self.operations = extrusion_operations

		self.center = center
		if center:
			self.extrusion_centers = extrusion_centers

		self.extent = extent
		if extent:
			self.extrusion_extents = extrusion_extents

		print(np.max(self.n_instances))
		if (np.max(self.n_instances) != self.K):
			print("WARNING. K= " + str(self.K) + ", max_instance in data= " + str(np.max(self.n_instances)))

		print("Number of samples: "+str(self.n_samples))


	def __getitem__(self, index):
		### Shuffle and sample point cloud
		idx = torch.randperm(self.pcs.shape[1])
		#idx = torch.arange(self.pcs.shape[1])
		if (self.pcs.shape[1] < self.npoints):
			print("ERROR. Sampling more points than point cloud resolution.")

		selected_idx = idx[:self.npoints]
		# print(selected_idx)

		sampled_pcs = self.pcs[index][selected_idx,:]
		sampled_normals = self.normals[index][selected_idx,:]
		sampled_extrusion_labels = self.extrusion_labels[index][selected_idx]
		sampled_bb_labels = self.bb_labels[index][selected_idx]

		per_point_extrusion_axes = self.extrusion_axes[index][sampled_extrusion_labels]
		per_point_extrusion_distances = self.extrusion_distances[index][sampled_extrusion_labels]

		extrusion_axes = self.extrusion_axes[index][:self.K]
		extrusion_distances = self.extrusion_distances[index][:self.K]

		## Sketch
		sk_idx = torch.randperm(self.sketches.shape[2])
		#sk_idx = torch.arange(self.sketches.shape[2])
		selected_idx = sk_idx[:self.num_sk_points]
		sampled_sketch = self.sketches[index][:,selected_idx,:]

		if not self.extent:
			if not self.with_scale:
				if not self.op and not self.center:
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_sketch
				elif not self.op and self.center:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers, sampled_sketch
				elif self.op and not self.center:
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_sketch					
				else:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers, sampled_sketch

			else:
				sampled_norm_factors = self.sk_norm_factors[index]
				if not self.op and not self.center:
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_sketch, sampled_norm_factors
				elif not self.op and self.center:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers, sampled_sketch, sampled_norm_factors
				elif self.op and not self.center:
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_sketch, sampled_norm_factors					
				else:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers, sampled_sketch, sampled_norm_factors
		else:
			sampled_extrusion_extents = self.extrusion_extents[index][:self.K]			

			if not self.with_scale:
				if not self.op and not self.center:
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_sketch, sampled_extrusion_extents
				elif not self.op and self.center:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers, sampled_sketch, sampled_extrusion_extents
				elif self.op and not self.center:
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_sketch, sampled_extrusion_extents					
				else:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers, sampled_sketch, sampled_extrusion_extents

			else:
				sampled_norm_factors = self.sk_norm_factors[index]
				if not self.op and not self.center:
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_sketch, sampled_norm_factors, sampled_extrusion_extents
				elif not self.op and self.center:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_centers, sampled_sketch, sampled_norm_factors, sampled_extrusion_extents
				elif self.op and not self.center:
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_sketch, sampled_norm_factors, sampled_extrusion_extents					
				else:
					sampled_extrusion_centers = self.extrusion_centers[index][:self.K]			
					sampled_extrusion_op = self.operations[index][selected_idx]
					return sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
							per_point_extrusion_distances, extrusion_axes, extrusion_distances, sampled_extrusion_op, sampled_extrusion_centers, sampled_sketch, sampled_norm_factors, sampled_extrusion_extents

	def __len__(self):
		return self.n_samples


if __name__ == "__main__":
	fname = "../../autodesk_data_extrusions/data_ss_dense_ext2_8_filtersmallext/train.h5"
	num_points = 4096
	max_instances = 8
	
	dataset = AutodeskDataset_h5(fname, num_points, max_instances)
	print(len(dataset))
	sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
				per_point_extrusion_distances, extrusion_axes, extrusion_distances = dataset[0]

	print(sampled_pcs)	
	print(sampled_normals)	
	print(sampled_extrusion_labels)	
	print(sampled_bb_labels)	
	print(per_point_extrusion_axes)	
	print(per_point_extrusion_distances)	
	print(extrusion_axes)	
	print(extrusion_distances)
	print()
	print(sampled_pcs.shape)	
	print(sampled_normals.shape)	
	print(sampled_extrusion_labels.shape)	
	print(sampled_bb_labels.shape)	
	print(per_point_extrusion_axes.shape)	
	print(per_point_extrusion_distances.shape)	
	print(extrusion_axes.shape)	
	print(extrusion_distances.shape)		



