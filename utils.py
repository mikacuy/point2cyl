# Mikaela Uy (mikacuy@cs.stanford.edu)

import json
import numpy as np
import math
import sys
import h5py

from global_variables import *

import trimesh
from PIL import Image

print("Imported utils.")

######### JSON Parsing Autodesk Data #############

def parse_files(model_id):
	filename = os.path.join(RAW_ROOT_DIR, model_id+".json")
	with open(filename) as f:
		json_data = json.load(f)
	json_sequence = json_data["sequence"]
	json_timeline = json_data["timeline"]
	json_entities = json_data["entities"]
	obj_list, entities_list = collect_objs(json_sequence)
	
	return obj_list, entities_list, json_sequence, json_timeline, json_entities 

# Get all OBJs created in the design
def collect_objs(json_sequence):
	### Returns ordered objs 
	ordered_objs = []
	ordered_entities = []
	for entry in json_sequence:
		if "obj" in entry:
			assert entry["type"] == "ExtrudeFeature", "Error in extracting obj_list from json sequence."
			ordered_objs.append(entry["obj"])
			ordered_entities.append(entry["entity"])
	return ordered_objs, ordered_entities

def direction_from_sketch(extrude_sketch):
    normal = extrude_sketch["reference_plane"]["plane"]["normal"]
    normal = np.array([float(normal["x"]), float(normal["y"]), float(normal["z"])])
    return normal

def get_extrude_infos(ordered_entities, json_entities, filter_two_extents=False, filter_tapered=True, index = None):
	
	if index is not None:
		ordered_entities = ordered_entities[:index+1]
	
	### Returns a dictionary mapping extrude entities to axis and distance
	extrude_info = {}
	for entity in ordered_entities:
		entity_info = json_entities[entity]

		## Check if it extrudes into two directions
		if filter_two_extents:
			if "extent_two" in entity_info:
				return None

		## Check if tapered
		if filter_tapered:
			if entity_info["extent_one"]["taper_angle"]["value"] > g_zero_tol:
				return None
			if "extent_two" in entity_info and entity_info["extent_two"]["taper_angle"]["value"] > g_zero_tol:
				return None

		extrude_operation = entity_info["operation"]

		extrude_distance = entity_info["extent_one"]["distance"]["value"]
		extrude_sketch = entity_info["profiles"][0]["sketch"]
		normal = direction_from_sketch(json_entities[extrude_sketch])

		if (1.0 - np.sum(np.abs(normal)**2,axis=-1)**(1./2)) > g_zero_tol:
			 print("Extrusion axis not unit vector")
			 normal /= np.sum(np.abs(normal)**2,axis=-1)**(1./2)

		extrude_info[entity] = {
			"distance": extrude_distance,
			"axis": normal,
			"operation": extrude_operation
		}

		### Get extrude faces, these are group ids
		extrude_info[entity]["all_faces"] = entity_info["extrude_faces"]
		extrude_info[entity]["side_faces"] = entity_info["extrude_side_faces"]
		extrude_info[entity]["start_faces"] = entity_info["extrude_start_faces"]
		extrude_info[entity]["end_faces"] = entity_info["extrude_end_faces"]

	return extrude_info

####### This function is problematic. Fix!
####### New : Added split face recovery, and filtering of merge faces
####### The problem with group_to_id is when split or merge happens (are there other edge cases?)
def face_groups_to_extrusion_id(ordered_entities, json_entities):

	## Keep track which groups are created by an entity
	entity_to_group = {}

	## For a group, give the extrusion step that created it, ordered by ordered entities
	group_to_id = {}

	num_new_groups = []
	num_deleted_group = []

	for i in range(len(ordered_entities)):
		entity = ordered_entities[i]
		entity_info = json_entities[entity]
		operation = entity_info["operation"]
		
		group_ids = entity_info["extrude_faces"]
		
		new_group = []
		for group_id in group_ids:
			if group_id not in group_to_id:
				group_to_id[group_id] = i
				new_group.append(group_id)

		### Check the num groups that were deleted from this extrusion
		### For group_delta_check()
		# Get all faces
		body_faces = []
		bodies = entity_info["bodies"]
		for body in bodies:
			body_faces += entity_info["bodies"][body]["faces"]

		# Check whether all groups are in the current body, otherwise a group was deleted
		num_deleted = 0
		for group_id in group_to_id:
			if group_id not in body_faces:
				num_deleted += 1

		if len(num_deleted_group)==0:
			num_deleted_group.append(num_deleted)
		else:
			num_deleted_group.append(num_deleted - num_deleted_group[-1])
		#########

		entity_to_group[entity] = new_group

		# print(len(new_group))
		# print(num_deleted)
		# print()
		num_new_groups.append(len(new_group))

	return group_to_id, entity_to_group, num_new_groups, num_deleted_group


### Attempt to handle split faces
def collect_split_faces(ordered_entities, json_entities, index=None):

	if index is not None:
		ordered_entities = ordered_entities[:index+1]

	## Keep track of current created faces
	created_faces = []

	## key is the group id for which split face occured
	## value is the timestep i (index in ordered_entities) for which it appeared
	## need to check operations in timesteps 0, 1, 2, ..., i-1, where the face belonged to and was created
	## get parent face
	split_faces = {}

	for i in range(len(ordered_entities)):
		entity = ordered_entities[i]
		entity_info = json_entities[entity]
		
		group_ids = entity_info["extrude_faces"]
		
		new_group = []
		for group_id in group_ids:
			if group_id not in created_faces:
				created_faces.append(group_id)

		### Check faces that were not previously created
		### Indicates split face has happened
		bodies = entity_info["bodies"]
		for body in bodies:
			curr_faces = entity_info["bodies"][body]["faces"]

			for f in curr_faces:
				if f not in created_faces:
					### Split face detected
					split_faces[f] = i

	return split_faces


def get_groups_ordered_objs(ordered_objs):
	'''
	Return:	[list] number of groups of each obj in list
			[list of dictionaries] each dictionary has keys which are groups, and value is the total surface area of the group
	'''
	num_groups = []
	group_areas = []

	for j in range(len(ordered_objs)):
		obj_fname = os.path.join(RAW_ROOT_DIR, ordered_objs[j])
		vertices, faces, face_normals, groups, _ = load_obj(obj_fname, True)

		# print(j, len(groups))

		##Get face areas
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
		face_areas = mesh.area_faces

		group_surface_areas = {}
		for group in groups:
			g_faces = groups[group]
			g_face_areas = face_areas[g_faces]
			g_surface_area = np.sum(g_face_areas)

			group_surface_areas[group] = g_surface_area

		num_groups.append(len(groups))
		group_areas.append(group_surface_areas)

	return num_groups, group_areas

def get_split_face_assignments(ordered_objs, split_faces, group_to_id, index = None):
	'''
	Returns : original groupid of the splitted face
	'''
	if index is not None:
		ordered_objs = ordered_objs[:index+1]

	all_meshes_and_groups = []

	## key is the split face groupid, value is the groupid reassignment
	split_face_groupid = {}

	for j in range(len(ordered_objs)):
		obj_fname = os.path.join(RAW_ROOT_DIR, ordered_objs[j])
		vertices, faces, face_normals, groups, _ = load_obj(obj_fname, True)

		mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
		
		all_meshes_and_groups.append((mesh, groups))	

	for f in split_faces:
		idx = split_faces[f]
		
		## Get faces in the split face
		curr_mesh = all_meshes_and_groups[idx][0]
		curr_group = all_meshes_and_groups[idx][1]

		idx_faces_of_interest = curr_group[f]
		# print(idx_faces_of_interest)
		# exit()

		points_on_faces = []
		for f_id in idx_faces_of_interest:
			vertices = np.array(curr_mesh.vertices[curr_mesh.faces[f_id]])
			c = np.mean(vertices, axis=0)
			points_on_faces.append(c)

		points_on_faces = np.array(points_on_faces)
		# print(points_on_faces)

		## Get face reassignments
		f_groupid_reassignment = []

		found = False
		curr_mesh_idx = idx-1

		## Need while loop to handle double split face
		while (not found and curr_mesh_idx>=0):	
			prev_mesh = all_meshes_and_groups[curr_mesh_idx][0]
			prev_groups = all_meshes_and_groups[curr_mesh_idx][1]
			_, dist, cf_id = trimesh.proximity.ProximityQuery(prev_mesh).on_surface(points_on_faces)

			for i in range(len(dist)):
				d = dist[i]
				if (d < g_zero_tol):
					### Find the group reassignment
					for group_id in prev_groups:
						if cf_id[i] in prev_groups[group_id] and group_id in group_to_id:
							## face found to reassign
							f_groupid_reassignment.append(group_id)

				else:
					curr_mesh_idx = curr_mesh_idx -1
					continue

			if len(f_groupid_reassignment) == len(points_on_faces):
				found = True
			else:
				### multiple splits
				curr_mesh_idx = curr_mesh_idx -1

		if not found:
			#### Check if this happens
			print("Wasn't able to recover split face....")
			return None

		## Chack that split face all belong to the same group
		if (len(set(f_groupid_reassignment)) != 1):  
			print("ERROR. Split face belongs to multiple groups")
			return None

		split_face_groupid[f] = f_groupid_reassignment[0]

	return split_face_groupid

def update_grouptoid_from_splitface(group_to_id, split_face_groupid):
	## Update group ids for the split faces using the relabel assignments

	for f in split_face_groupid:
		orig_group_assignment = split_face_groupid[f]
		split_face_id = group_to_id[orig_group_assignment]

		## Add to groupid
		group_to_id[f] = split_face_id

	return group_to_id


def group_surface_areas_check(group_areas, index=None):
	'''
	Checks whether the surface area of a groupid DID NOT increase as you progress through the design sequence
	When new extrusions are made, a new group should be created, or faces are cut out from the existing group
	'''

	if index is not None:
		group_areas = group_areas[:index+1]

	current_group_areas = {}
	for group_surface_areas in group_areas:
		for group in group_surface_areas:
			## New group in the sequence
			if group not in current_group_areas:
				current_group_areas[group] = group_surface_areas[group]

			## Check if area is non-increasing, update current group area
			else:
				prev_g_area = current_group_areas[group]
				curr_g_area = group_surface_areas[group]

				if curr_g_area > prev_g_area + g_zero_tol:
					#Area increased
					return False

				current_group_areas[group] = curr_g_area

	return True

def group_delta_check(num_groups_objs, num_newgroups_json, num_deleted_group_json, index=None):
	'''
	Checks the number of new groups in the json file in equal to 
	'''
	if index is not None:
		num_groups_objs = num_groups_objs[:index+1]
		num_newgroups_json = num_newgroups_json[:index+1]
		num_deleted_group_json = num_deleted_group_json[:index+1]

	if (num_groups_objs[0] != num_newgroups_json[0]):
		return False

	for i in range(1, len(num_groups_objs)):
		delta = num_groups_objs[i] - num_groups_objs[i-1]
		if num_newgroups_json[i] - num_deleted_group_json[i] !=  delta:
			return False

	return True

def normals_extrusions_check(normals, extrusion_labels, extrusion_axes):
	num_points = normals.shape[0]

	for i in range(num_points):
		n = normals[i]
		ext = extrusion_axes[extrusion_labels[i]]

		if (np.abs(np.dot(n, ext)) > g_zero_tol) and (1 - np.abs(np.dot(n, ext)) > g_zero_tol):
			return False
	return True

def get_base_barrel_label(normals, extrusion_labels, extrusion_axes):
	num_points = normals.shape[0]

	bb_labels = []
	for i in range(num_points):
		n = normals[i]
		ext = extrusion_axes[extrusion_labels[i]]

		if (np.abs(np.dot(n, ext)) <= g_zero_tol):
			bb_labels.append(0)
		elif ( (1 - np.abs(np.dot(n, ext))) < g_zero_tol ):
			bb_labels.append(1)
		else:
			print("Error in base barrel labeling. This should not happen")
			return None
	bb_labels = np.array(bb_labels)

	return bb_labels

def get_base_barrel_label_faces(face_normals, extrude_info, ordered_entities, face_to_ids, index = None):

	if index is not None:
		ordered_entities = ordered_entities[:index+1]
	
	face_bb_label = []

	for i in range(face_normals.shape[0]):
		fn = face_normals[i]
		extrude_id = face_to_ids[i]

		extrude_axis = extrude_info[ordered_entities[extrude_id]]["axis"]

		if (np.abs(np.dot(fn, extrude_axis)) <= g_zero_tol):
			face_bb_label.append(0)
		elif ( (1 - np.abs(np.dot(fn, extrude_axis))) < g_zero_tol ):
			face_bb_label.append(1)
		else:
			return None

	face_bb_label = np.array(face_bb_label)

	return face_bb_label


def get_operation_label(extrusion_labels, operation):
	num_points = extrusion_labels.shape[0]

	op_labels = []
	for i in range(num_points):
		op_label = operation[extrusion_labels[i]]
		op_labels.append(op_label)
		
		if (op_label==2):
			print("Intersection operation found.")

	op_labels = np.array(op_labels)

	return op_labels


def entity_to_extrusion_id(ordered_entities):
	## IDs are ordered
	## Returns a dictionary where key: entity, value: extrusion_id
	extrusion_labels = {}

	for i in range(len(ordered_entities)):
		entity = ordered_entities[i]
		extrusion_labels[entity] = i

	return extrusion_labels

### For multi-loop extraction
# Check for connected components for each set of faces for extrusion segments
def check_and_relabel_multiloop(vertices, faces, face_bb_labels, face_to_ids):
	## Remove duplicate vertices for accurate connected component calculation
	mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
	# Overwrite faces because trimesh reorders the vertices
	faces = mesh.faces	

	##############################################	
	# Collect base/barrel faces of the same extrusion segment
	# Separate base/barrel
	# ids_to_faces = {}
	ids_to_faces_barrel = {}
	ids_to_faces_base = {}

	unique_eids = []
	## Helper dictionaries construction
	for f_id in range(len(face_to_ids)):
		e_id = face_to_ids[f_id]

		if e_id not in unique_eids:
			unique_eids.append(e_id)

		# if e_id not in ids_to_faces:
		# 	ids_to_faces[e_id] = [f_id]
		# else:
		# 	ids_to_faces[e_id].append(f_id)

		# barrel
		if face_bb_labels[f_id] == 0:

			if e_id not in ids_to_faces_barrel:
				ids_to_faces_barrel[e_id] = [f_id]
			else:
				ids_to_faces_barrel[e_id].append(f_id)

		# base
		else:
			if e_id not in ids_to_faces_base:
				ids_to_faces_base[e_id] = [f_id]
			else:
				ids_to_faces_base[e_id].append(f_id)

	ids_to_faces = {}
	# Combine base and barrel components
	for e_id in unique_eids:
		curr_faces = []
		### Need to append base first
		if e_id in ids_to_faces_base.keys():
			curr_faces += ids_to_faces_base[e_id]
		if e_id in ids_to_faces_barrel.keys():
			curr_faces += ids_to_faces_barrel[e_id]	
			
		ids_to_faces[e_id] = np.array(curr_faces)
	## 			

	##############################################	
	# Keep track of parent label of splitted loops (for extrude_info)		
	splitted_labels = {}
	curr_max_label = max(ids_to_faces.keys())

	# Get barrel face connectivities to extract multiple loops, update face_to_ids
	for e_id in ids_to_faces_barrel:
		curr_segment_barrel_fid = np.array(ids_to_faces_barrel[e_id])
		curr_segment_barrel_faces = faces[curr_segment_barrel_fid]

		face_edges = trimesh.graph.face_adjacency(curr_segment_barrel_faces)
		connected_components = trimesh.graph.connected_component_labels(face_edges, node_count=len(curr_segment_barrel_faces))

		## Check if multiloop
		components = np.unique(connected_components)
		print(components)
		print(len(components))

		## To keep track of the splitted labels
		segment_labels = [e_id]

		## Relabel
		if len(components) > 1:
			## Update face_to_ids
			for i in range(len(connected_components)):
				curr_component = connected_components[i]

				if curr_component != 0 :
					new_label = curr_max_label + curr_component
					face_to_ids[curr_segment_barrel_fid[i]] = new_label

					if new_label not in segment_labels:
						segment_labels.append(new_label)

			curr_max_label += (len(components)-1)
			
		## Keep track of splitted face
		splitted_labels[e_id] = segment_labels

	# Keep track of parent label of splitted loops (for extrude_info)
	splitted_label_mapping = {}
	for k in splitted_labels.keys():
		for l in splitted_labels[k]:
			splitted_label_mapping[l] = k
	##############################################	
	
	# Assign base faces to relabeled multi loops, update face_to_id
	for e_id in ids_to_faces_base:
		## Base connected to components
		curr_segment_base_fid = np.array(ids_to_faces_base[e_id])
		curr_segment_base_faces = faces[curr_segment_base_fid]

		face_edges_base = trimesh.graph.face_adjacency(curr_segment_base_faces)
		connected_components_base = trimesh.graph.connected_component_labels(face_edges_base, node_count=len(curr_segment_base_faces))

		## Barrel connected to components (recomputed from prev, inefficient...)
		if e_id not in ids_to_faces_barrel.keys():
			return None, None

		curr_segment_barrel_fid = np.array(ids_to_faces_barrel[e_id])
		curr_segment_barrel_faces = faces[curr_segment_barrel_fid]

		face_edges_barrel = trimesh.graph.face_adjacency(curr_segment_barrel_faces)
		connected_components_barrel = trimesh.graph.connected_component_labels(face_edges_barrel, node_count=len(curr_segment_barrel_faces))

		## Check if multiloop
		components_barrel = np.unique(connected_components_barrel)
		print(components_barrel)
		print(len(components_barrel))

		## Combined connected component
		curr_segment_fid = np.array(ids_to_faces[e_id])
		curr_segment_faces = faces[curr_segment_fid]

		face_edges = trimesh.graph.face_adjacency(curr_segment_faces)
		connected_components = trimesh.graph.connected_component_labels(face_edges, node_count=len(curr_segment_faces))

		## Relabel done if multiple loops in barrel
		if len(components_barrel) > 1:
			base_comp_to_id = {}
			## Update face_to_ids
			for i in range(len(connected_components_base)):
				curr_component_id = connected_components_base[i]

				if curr_component_id in base_comp_to_id.keys():
					continue

				### Base faces come first in the whole mesh
				whole_component_id = connected_components[i]
				label_candidates = []
				# Find a barrel labels belonging to the same component
				for w_id in range(len(connected_components)):
					##Check if barrel
					if face_bb_labels[curr_segment_fid[w_id]] == 0 and connected_components[w_id] == whole_component_id:
						label = face_to_ids[curr_segment_fid[w_id]]
						label_candidates.append(label)

				label_candidates = np.unique(np.array(label_candidates))
				base_comp_to_id[curr_component_id] = label_candidates

			## Get centroid of each base for donut relabeling
			print("Handling base labeling in donut case")
			unique_base_labels = np.unique(connected_components_base)
			base_label_centroid = {}

			base_comp_relabel = {}
			for label in unique_base_labels:	
				curr_idx = np.where(connected_components_base==label)[0]
				curr_faces = faces[curr_idx]
				pc_base, _ = sample_point_cloud_partial(mesh.vertices, curr_faces)
				centroid = np.mean(pc_base, axis=0)

				## Get candidate barrel labels
				barrel_labels = base_comp_to_id[label]

				candidate_dists = []
				candidate_fid = []
				for barrel_label in barrel_labels:
					curr_idx = np.where(np.logical_and(face_to_ids==barrel_label, face_bb_labels ==0))[0]
					curr_faces = faces[curr_idx]
					print(curr_faces.shape)
					curr_pc_barrel, sampled_faces = sample_point_cloud_partial(mesh.vertices, curr_faces)

					## Get furthest point from the centroid in the barrel
					dist_2 = np.sum((curr_pc_barrel - centroid)**2, axis=1)
					cmax_dist =  np.max(dist_2)
					cmax_dist_idx =  np.argmax(dist_2)
					cmax_dist_face_id = curr_idx[sampled_faces[cmax_dist_idx]]
					
					candidate_dists.append(cmax_dist)
					candidate_fid.append(cmax_dist_face_id)

				# Get furthest face
				max_idx = np.argmax(np.array(candidate_dists))
				max_fid = candidate_fid[max_idx]

				# Get label
				base_comp_relabel[label] = face_to_ids[max_fid]


			## Relabel
			for i in range(len(connected_components_base)):
				curr_component_id = connected_components_base[i]
				base_face_id = curr_segment_base_fid[i]

				### Updated
				# new_label = base_comp_to_id[curr_component_id][0]
				new_label = base_comp_relabel[curr_component_id]
				###

				face_to_ids[base_face_id] = new_label

	return face_to_ids, splitted_label_mapping
###########################################

######## Loading Autodesk OBJ #############
## For normal calculation
def normalize_normals(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def load_obj(filename, get_groups=True, group_to_id=None):
	fin = open(filename, 'r')
	lines = [line.rstrip() for line in fin]
	fin.close()

	vertices = []
	faces = []

	### Get mesh info
	for line in lines:
		if line.startswith('v '):
			vertices.append(np.float32(line.split()[1:4]))
		elif line.startswith('f '):
			## index of faces offset by 1
			faces.append(np.array([int(item.split('/')[0])-1 for item in line.split()[1:4]]))
		# elif line.startswith('vn '):
		# 	normals.append(np.float32(line.split()[1:4]))

	vertices = np.array(vertices)
	faces = np.array(faces)

	## Calcualte face normals
	triangles = vertices[faces]
	face_normals = np.cross( triangles[::,0 ] - triangles[::,1]  , triangles[::,0 ] - triangles[::,2] )
	face_normals = normalize_normals(face_normals)
	face_normals = np.array(face_normals)

	# ### Debug
	# print(vertices.shape)
	# print(faces.shape)
	# print(face_normals.shape)
	# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)
	# out_mesh_file = "test.obj"  
	# mesh.export(out_mesh_file)
	# print("Saved '{}'.".format(out_mesh_file))
	# exit()	

	group_dict = {}
	face_to_ids = []

	if get_groups:

		new_group = []
		
		### Dictionary from entity to a list of face_ids
		groups = {}
		
		group_id = ""
		reading_group = False
		f_counter = 0
		group_counter = 0

		for line in lines:
			if line.startswith('g '):
				reading_group = True
				group_counter += 1
				if(len(new_group) > 0):
					new_group = np.array(new_group)
					groups[group_id] = new_group
				group_id = line.split()[1]
				new_group = []
				continue

			if(reading_group == True and line.startswith('f ')):
				items = line.split()
				
				## Append face_id to the group
				new_group.append(f_counter)

				##### DEBUG!!!
				## Map face_id to extrusion label
				if group_to_id is None:
					face_to_ids.append(0)
				else:
					face_to_ids.append(group_to_id[group_id])
				############

			if line.startswith('f '):
				f_counter += 1

		if(len(new_group) > 0):
			new_group = np.array(new_group)
			groups[group_id] = new_group  

		# print(group_counter)
		face_to_ids = np.array(face_to_ids)

		return vertices, faces, face_normals, groups, face_to_ids

	return vertices, faces, face_normals
#################################################

####### For filtering small faces ######
def face_areas(V, F):
	M = np.shape(F)[0]

	# Compute face areas.
	tri_verts = np.empty((M, 3, 3))
	for i in range(M):
		for j in range(3):
			tri_verts[i,j] = V[F[i,j]-1]

	areas = trimesh.triangles.area(tri_verts)

	return areas

def get_extrusion_segment_areas(face_to_ids, areas):
	extrusions_dict = {}

	for i in range(areas.shape[0]):
		curr_id = face_to_ids[i]

		if curr_id in extrusions_dict:
			extrusions_dict[curr_id] += areas[i]
		else:
			extrusions_dict[curr_id] = areas[i]

	return extrusions_dict

def get_area_distribution(extrusions_dict):
	extrusion_areas = []
	for key in extrusions_dict.keys():
		extrusion_areas.append(extrusions_dict[key])

	extrusion_areas = np.array(extrusion_areas)
	area_distribution = extrusion_areas / np.sum(extrusion_areas)
	return area_distribution

####### For filtering small barrel extents ######
def get_barrel_extents(point_cloud, bb_labels, extrusion_labels, extrusion_axes, with_extents=False):
	# Get the extents/distance of each barrel segment
	# Given the center point of the barrel c, a point p on the barrel with axis x
	# Project (p-c) onto x and take the range of the projections

	num_segments = np.max(extrusion_labels) + 1
	barrel_points_idx = np.squeeze(np.argwhere(bb_labels==0))

	ext_dists = []
	num_barrel_points = []
	extents = []
	for i in range(num_segments):
		## Get barrel points in current segment
		curr_ext_idx = np.squeeze(np.argwhere(extrusion_labels==i))
		selected_idx = np.intersect1d(barrel_points_idx, curr_ext_idx)
		
		curr_barrel_pc = point_cloud[selected_idx]

		# print()
		# print(selected_idx)
		# print(selected_idx.shape)

		if (len(selected_idx) == 0):
			ext_dists.append(0)
			num_barrel_points.append(len(selected_idx))	
			continue		

		## Get center point
		c = np.mean(curr_barrel_pc, axis=0)

		# print(curr_barrel_pc.shape)
		# print(c)
		# print(extrusion_axes[i])

		dot = np.dot(curr_barrel_pc - c, extrusion_axes[i])

		min_extent = np.min(dot)
		max_extent = np.max(dot)

		curr_extent = np.array([min_extent, max_extent])
		extents.append(curr_extent)

		curr_ext_dist = np.ptp(dot)
		ext_dists.append(curr_ext_dist)

		num_barrel_points.append(len(selected_idx))

	ext_dists = np.array(ext_dists)
	num_barrel_points = np.array(num_barrel_points)
	extents = np.array(extents)

	if not with_extents:
		return ext_dists, num_barrel_points
	else:
		return ext_dists, num_barrel_points, extents


## Get extrusion centers
def get_extrusion_centers(point_cloud, extrusion_labels):
	num_segments = np.max(extrusion_labels) + 1

	ext_centers = []
	for i in range(num_segments):
		## Get barrel points in current segment
		curr_ext_idx = np.squeeze(np.argwhere(extrusion_labels==i))	
		curr_pc = point_cloud[curr_ext_idx]
		curr_center = np.mean(curr_pc, axis=0)

		ext_centers.append(curr_center)

	ext_centers = np.array(ext_centers)

	return ext_centers



####### Point cloud preprocessing #########
def sample_point_cloud(vertices, faces, face_normals, num_points, face_to_ids, sample_even=True):
	#### Output point cloud, normals per point, and extrusion label per point

	mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

	if sample_even:
		point_cloud, sampled_faces = trimesh.sample.sample_surface_even(mesh, num_points)
	else:
		point_cloud, sampled_faces = trimesh.sample.sample_surface(mesh, num_points)


	# print(point_cloud.shape)
	# print(sampled_faces.shape)

	normals =  []
	extrusion_labels = []

	for i in range(point_cloud.shape[0]):
		curr_face_id = sampled_faces[i]
		curr_extrusion_label = face_to_ids[curr_face_id]
		curr_normal = face_normals[curr_face_id]

		extrusion_labels.append(curr_extrusion_label)
		normals.append(curr_normal)

	normals = np.array(normals)
	extrusion_labels = np.array(extrusion_labels)

	# ### Debug
	# print(normals.shape)
	# print(extrusion_labels.shape)

	return point_cloud, normals, extrusion_labels

def sample_point_cloud_partial(vertices, faces, num_points=4096, sample_even=False):
	#### Output point cloud, normals per point, and extrusion label per point

	mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

	if sample_even:
		point_cloud, sampled_faces = trimesh.sample.sample_surface_even(mesh, num_points)
	else:
		point_cloud, sampled_faces = trimesh.sample.sample_surface(mesh, num_points)


	return point_cloud, sampled_faces

def center_data(pcs):
    for pc in pcs:
        centroid = np.mean(pc, axis=0)
        pc[:,0]-=centroid[0]
        pc[:,1]-=centroid[1]
        pc[:,2]-=centroid[2]
    return pcs

def normalize_data(pcs):
    for pc in pcs:
        #get furthest point distance then normalize
        d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
        pc /= d

    return pcs, norm_factor

def center_data_single(pc):
    centroid = np.mean(pc, axis=0)
    pc[:,0]-=centroid[0]
    pc[:,1]-=centroid[1]
    pc[:,2]-=centroid[2]
    return pc

def normalize_data_single(pc):
    #get furthest point distance then normalize
    d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
    pc /= d

    return pc, d
################################

########## Renderer ###########
def render_point_cloud(point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None, default_angle=0):
	g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'

	if (default_angle == 0):
		g_azimuth_deg = -70
		g_elevation_deg = 20
		g_theta_deg = 0

	elif (default_angle == 1):
		g_azimuth_deg = 20
		g_elevation_deg = -70
		g_theta_deg = 0		

	elif (default_angle == 2):
		g_azimuth_deg = -70
		g_elevation_deg = 20
		g_theta_deg = 0		

	if (default_angle == 1):
		snapshot_file += "_angle1"
	elif (default_angle == 2):
		snapshot_file += "_angle2"

	if not outfile:
		cmd = g_renderer + ' \\\n'
		cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
		cmd += ' --point_labels=' + point_labels_file + ' \\\n'
		cmd += ' --snapshot=' + snapshot_file + ' \\\n'
		cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
		cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
		cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
		cmd += ' >/dev/null 2>&1'
	else:
		cmd = g_renderer + ' --point_cloud=' + point_cloud_file + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
				+ ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) + ' --theta_deg=' + str(g_theta_deg) \
				+ ' --theta_deg=' + str(g_theta_deg) + ' >/dev/null 2>&1'

	if outfile:
		filehandle_.write(cmd+'\n')
	else:		
		os.system(cmd)
	snapshot_file += '.png'
	print("Saved '{}'.".format(snapshot_file))


def render_sketch(point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None, adjust_camera=True):
	g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'

	g_azimuth_deg = -70
	g_elevation_deg = 20
	g_theta_deg = 0

	if not outfile:
		cmd = g_renderer + ' \\\n'
		cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
		cmd += ' --point_labels=' + point_labels_file + ' \\\n'
		cmd += ' --snapshot=' + snapshot_file + ' \\\n'
		cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
		cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
		cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
		cmd += ' >/dev/null 2>&1'
	else:
		if adjust_camera:
			cmd = g_renderer + ' --point_cloud=' + point_cloud_file + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
					+ ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) + ' --theta_deg=' + str(g_theta_deg) \
					+ ' --theta_deg=' + str(g_theta_deg) + ' >/dev/null 2>&1'
		else:
			cmd = g_renderer + ' --point_cloud=' + point_cloud_file + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
					+ ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) + ' --theta_deg=' + str(g_theta_deg) \
					+ ' --theta_deg=' + str(g_theta_deg) + ' --auto_adjust_camera=false >/dev/null 2>&1'			

	if outfile:
		filehandle_.write(cmd+'\n')
	else:		
		os.system(cmd)
	snapshot_file += '.png'
	print("Saved '{}'.".format(snapshot_file))


def render_mesh(mesh_file, face_labels_file, snapshot_file):
	g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'
	g_azimuth_deg = -70
	g_elevation_deg = 20
	g_theta_deg = 0

	cmd = g_renderer + ' \\\n'
	cmd += ' --mesh=' + mesh_file + ' \\\n'
	cmd += ' --face_labels=' + face_labels_file + ' \\\n'
	cmd += ' --snapshot=' + snapshot_file + ' \\\n'
	cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
	cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
	cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
	cmd += ' >/dev/null 2>&1'
	os.system(cmd)
	snapshot_file += '.png'
	print("Saved '{}'.".format(snapshot_file))


def render_autodesk_mesh(model_id, output_folder, vertices, faces, face_labels=None):
	# Save mesh.
	out_mesh_file = os.path.join(output_folder, 'mesh', model_id+'_mesh.obj')

	mesh = trimesh.Trimesh(vertices=vertices, faces=faces)  
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(output_folder, 'mesh', model_id + '_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, mesh.vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Save face ids.
	out_face_ids_file = os.path.join(output_folder, 'mesh', model_id + '_face_ids.txt')
	if face_labels is None:
		face_labels = np.zeros(faces.shape[0])
	np.savetxt(out_face_ids_file, face_labels, fmt='%d')
	print("Saved '{}'.".format(out_face_ids_file))

	# Render mesh.
	mesh_snapshot_file = os.path.join(output_folder, 'rendering_mesh', model_id+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, mesh_snapshot_file)

def render_autodesk_pointcloud(model_id, output_folder, pc, label, outfile=False, filehandle_=None):
	### Point Cloud ###
	# Save point cloud.
	out_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_points.xyz')
	np.savetxt(out_point_cloud_file, pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save point ids.
	out_point_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_point_ids.txt')
	np.savetxt(out_point_ids_file, label, fmt='%d')
	print("Saved '{}'.".format(out_point_ids_file))

	# Render point_cloud.
	points_snapshot_file = os.path.join(output_folder, 'rendering_point_cloud', model_id+'_points')
	render_point_cloud(out_point_cloud_file, out_point_ids_file,
			points_snapshot_file, outfile=outfile, filehandle_=filehandle_)

def combine_both_renders(model_id, output_folder, label):
	mesh_snapshot_file = os.path.join(output_folder, 'rendering_mesh', model_id+'_mesh')
	points_snapshot_file = os.path.join(output_folder, 'rendering_point_cloud', model_id+'_points')

	#Output to a single image
	height = 1080
	width = 1920
	new_im = Image.new('RGBA', (width*2, height))
	im1 = Image.open(mesh_snapshot_file+".png")
	im2 = Image.open(points_snapshot_file+".png")
	images = [im1, im2]
	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset,0))
		x_offset += width

	output_image_filename = os.path.join(output_folder, "rendering_both", model_id+'_both.png')
	new_im.save(output_image_filename)    
	print("Saved '{}'.".format(output_image_filename))	
###################################

##### Single model h5 file loader ########
def get_model(h5_file, mesh_info=False, operation=False):
	with h5py.File(h5_file, 'r') as f:
		# Normalized
		point_cloud = f["point_cloud"][:]
		normals = f["normals"][:]
		extrusion_labels = f["extrusion_labels"][:]
		extrusion_axes = f["extrusion_axes"][:]
		extrusion_distances = f["extrusion_distances"][:]
		n_instances = f["n_instances"][:]

		# NOT normalized
		vertices = f["vertices"][:] 
		faces = f["faces"][:]
		face_normals = f["face_normals"][:] 
		face_to_ids = f["face_extrusion_labels"][:]
		norm_factor = f["norm_factor"][:] 

		# # Sanity check
		# print(f.keys())
		# print(point_cloud.shape)
		# print(normals.shape)
		# print(extrusion_labels.shape)
		# print(extrusion_axes.shape)
		# print(extrusion_distances.shape)
		# print(n_instances)
		# print()
		# print(vertices.shape)
		# print(faces.shape)
		# print(face_normals.shape)
		# print(face_to_ids.shape)
		# print(norm_factor)
		if operation:
			curr_op = f["operation"][:] 
		else:
			curr_op = None

		if not mesh_info:
			return point_cloud, normals, extrusion_labels, extrusion_axes, extrusion_distances, n_instances, curr_op
		else:
			return point_cloud, normals, extrusion_labels, extrusion_axes, extrusion_distances, n_instances, vertices, faces, face_normals, face_to_ids, norm_factor, curr_op

###########################################

##### For h5 files ######
def save_dataset(fname, point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_operations=None, centers=None, extents=None):
	cloud = np.stack([pc for pc in point_cloud])
	cloud_normals = np.stack([normal for normal in normals])
	cloud_extrusion_labels = np.stack([extrusion_label for extrusion_label in extrusion_labels])
	cloud_bb_labels = np.stack([bb_label for bb_label in bb_labels])
	cloud_n_instances = np.stack([n_instance for n_instance in n_instances])
	cloud_extrusion_axes = np.stack([ex for ex in extrusion_axes])
	cloud_extrusion_distances = np.stack([dist for dist in extrusion_distances])
	if extrusion_operations is not None: 
		cloud_extrusion_operations = np.stack([op for op in extrusion_operations])
	if centers is not None: 
		cloud_centers = np.stack([c for c in centers])
	if extents is not None: 
		cloud_extents = np.stack([e for e in extents])

	fout = h5py.File(fname)
	fout.create_dataset('point_cloud', data=cloud, compression='gzip', dtype='float32')
	fout.create_dataset('normals', data=cloud_normals, compression='gzip', dtype='float32')
	fout.create_dataset('extrusion_labels', data=cloud_extrusion_labels, compression='gzip', dtype='int')
	fout.create_dataset('base_barrel_labels', data=cloud_bb_labels, compression='gzip', dtype='int')
	fout.create_dataset('n_instances', data=cloud_n_instances, compression='gzip', dtype='int')
	fout.create_dataset('extrusion_axes', data=cloud_extrusion_axes, compression='gzip', dtype='float32')
	fout.create_dataset('extrusion_distances', data=cloud_extrusion_distances, compression='gzip', dtype='float32')
	if extrusion_operations is not None:
		fout.create_dataset('extrusion_operation', data=cloud_extrusion_operations, compression='gzip', dtype='int')

	if centers is not None:
		fout.create_dataset('extrusion_centers', data=cloud_centers, compression='gzip', dtype='float32')
	if extents is not None:
		fout.create_dataset('extrusion_extents', data=cloud_extents, compression='gzip', dtype='float32')

	fout.close()

	print("Saved " + fname + ".")
	return

def load_h5(h5_filename, op=False, center=False, extent=False):
	with h5py.File(h5_filename, 'r') as f:
		# Normalized
		point_cloud = f["point_cloud"][:]
		normals = f["normals"][:]
		extrusion_labels = f["extrusion_labels"][:]
		bb_labels = f["base_barrel_labels"][:]
		n_instances = f["n_instances"][:]
		extrusion_axes = f["extrusion_axes"][:]
		extrusion_distances = f["extrusion_distances"][:]

		if op:
			extrusion_op = f["extrusion_operation"][:]
		if center:
			extrusion_center = f["extrusion_centers"][:]
		if extent:
			extrusion_extents = f["extrusion_extents"][:]

	if not extent:
		if not op and not center:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances
		if center and not op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_center 
		if center and op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, extrusion_center 		
		else:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op
	else:
		if not op and not center:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_extents
		if center and not op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_center, extrusion_extents 
		if center and op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, extrusion_center, extrusion_extents 		
		else:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, extrusion_extents

## With sketches
def save_dataset_sk(fname, point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, sketches, sketch_norms, extrusion_operations=None, centers=None, extents=None):
	cloud = np.stack([pc for pc in point_cloud])
	cloud_normals = np.stack([normal for normal in normals])
	cloud_extrusion_labels = np.stack([extrusion_label for extrusion_label in extrusion_labels])
	cloud_bb_labels = np.stack([bb_label for bb_label in bb_labels])
	cloud_n_instances = np.stack([n_instance for n_instance in n_instances])
	cloud_extrusion_axes = np.stack([ex for ex in extrusion_axes])
	cloud_extrusion_distances = np.stack([dist for dist in extrusion_distances])
	if extrusion_operations is not None: 
		cloud_extrusion_operations = np.stack([op for op in extrusion_operations])
	if centers is not None: 
		cloud_centers = np.stack([c for c in centers])
	if extents is not None: 
		cloud_extents = np.stack([e for e in extents])

	cloud_sketches = np.stack([sk for sk in sketches])
	cloud_sketches_norm = np.stack([sk_n for sk_n in sketch_norms])

	fout = h5py.File(fname)
	fout.create_dataset('point_cloud', data=cloud, compression='gzip', dtype='float32')
	fout.create_dataset('normals', data=cloud_normals, compression='gzip', dtype='float32')
	fout.create_dataset('extrusion_labels', data=cloud_extrusion_labels, compression='gzip', dtype='int')
	fout.create_dataset('base_barrel_labels', data=cloud_bb_labels, compression='gzip', dtype='int')
	fout.create_dataset('n_instances', data=cloud_n_instances, compression='gzip', dtype='int')
	fout.create_dataset('extrusion_axes', data=cloud_extrusion_axes, compression='gzip', dtype='float32')
	fout.create_dataset('extrusion_distances', data=cloud_extrusion_distances, compression='gzip', dtype='float32')
	if extrusion_operations is not None:
		fout.create_dataset('extrusion_operation', data=cloud_extrusion_operations, compression='gzip', dtype='int')

	if centers is not None:
		fout.create_dataset('extrusion_centers', data=cloud_centers, compression='gzip', dtype='float32')

	if extents is not None:
		fout.create_dataset('extrusion_extents', data=cloud_extents, compression='gzip', dtype='float32')

	fout.create_dataset('sketches', data=cloud_sketches, compression='gzip', dtype='float32')
	fout.create_dataset('sketches_norms', data=cloud_sketches_norm, compression='gzip', dtype='float32')

	fout.close()

	print("Saved " + fname + ".")
	return

def load_h5_sk(h5_filename, op=False, center=False, extent=False):
	with h5py.File(h5_filename, 'r') as f:
		# Normalized
		point_cloud = f["point_cloud"][:]
		normals = f["normals"][:]
		extrusion_labels = f["extrusion_labels"][:]
		bb_labels = f["base_barrel_labels"][:]
		n_instances = f["n_instances"][:]
		extrusion_axes = f["extrusion_axes"][:]
		extrusion_distances = f["extrusion_distances"][:]

		if op:
			extrusion_op = f["extrusion_operation"][:]
		if center:
			extrusion_center = f["extrusion_centers"][:]

		if extent:
			extrusion_extents = f["extrusion_extents"][:]

		sketches = f["sketches"][:]
		sketches_norm_factors = f["sketches_norms"][:]

	if not extent:
		if not op and not center:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, sketches, sketches_norm_factors
		if center and not op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_center, sketches, sketches_norm_factors 
		if center and op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, extrusion_center, sketches, sketches_norm_factors 		
		else:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, sketches, sketches_norm_factors
	else:
		if not op and not center:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, sketches, sketches_norm_factors, extrusion_extents
		if center and not op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_center, sketches, sketches_norm_factors, extrusion_extents 
		if center and op:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, extrusion_center, sketches, sketches_norm_factors, extrusion_extents 		
		else:
			return point_cloud, normals, extrusion_labels, bb_labels, n_instances, extrusion_axes, extrusion_distances, extrusion_op, sketches, sketches_norm_factors, extrusion_extents


