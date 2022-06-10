# Mikaela Uy (mikacuy@cs.stanford.edu)
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import datetime
import time
import sys
import importlib
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))

## For implicit
sys.path.append(os.path.join(BASE_DIR, 'IGR'))
from sampler import *
from network import *
from general import *
from plots import plot_surface_2d

from global_variables import *
from utils import *
from data_utils import *
from dataloader import AutodeskDataset_h5_sketches
from losses import *

import pickle

from thop import profile
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pointnet_extrusion', help='model name')
parser.add_argument('--num_point', type=int,  default=8192, help='Point Number [default: 8192]')
parser.add_argument('--num_sk_point', type=int,  default=2048, help='Point Number [default: 2048]')
parser.add_argument('--K', type=int,  default=8, help='Max number of extrusions')
parser.add_argument('--batch_size', type=int,  default=4, help='batch size')

parser.add_argument("--logdir", default="./results/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="model.pth", help="checkpoint", type=str)

parser.add_argument('--dump_dir', default= "./results/", type=str)

parser.add_argument('--data_dir', type=str, default='data/')

parser.add_argument('--data_split', default= "test", type=str)
parser.add_argument('--visu', action='store_true')

parser.add_argument('--pred_seg', action='store_false')
parser.add_argument('--pred_normal', action='store_false')
parser.add_argument('--pred_bb', action='store_false')
parser.add_argument('--pred_extrusion', action='store_false')
parser.add_argument('--norm_eig', action='store_true')

parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--noise_sigma', type=float, default=0.01, help='Sigma for random noise addition.')

### For extrusion axis prediction
parser.add_argument('--use_gt_normals', action='store_true')
parser.add_argument('--use_gt_segmentation', action='store_true')
parser.add_argument('--use_gt_bb', action='store_true')

### To output gt
parser.add_argument('--use_gt_sketch', action='store_true')
parser.add_argument('--use_gt_im', action='store_true')
parser.add_argument('--use_whole_pc', action='store_true')
parser.add_argument('--use_extrusion_axis_feat', action='store_true')

parser.add_argument("--im_logdir", default="./results/IGR_dense/", help="path to the log directory", type=str)
parser.add_argument("--im_ckpt", default="latest.pth", help="checkpoint", type=str)

FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)
plot_fol = os.path.join(DUMP_DIR, "plot")
if not os.path.exists(plot_fol): os.mkdir(plot_fol)
pickle_fol = os.path.join(DUMP_DIR, "pickle")
if not os.path.exists(pickle_fol): os.mkdir(pickle_fol)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

DATA_SPLIT = FLAGS.data_split
DATA_DIR = FLAGS.data_dir
H5_FILENAME = os.path.join(DATA_DIR, DATA_SPLIT+".h5")

NUM_POINT = FLAGS.num_point
NUM_SK_POINT = FLAGS.num_sk_point
MODEL = FLAGS.model
K = FLAGS.K
BATCH_SIZE = FLAGS.batch_size

PRED_SEG = FLAGS.pred_seg
PRED_NORMAL = FLAGS.pred_normal
PRED_EXT = FLAGS.pred_extrusion
PRED_BB = FLAGS.pred_bb

NORM_EIG = FLAGS.norm_eig

ADD_NOISE = FLAGS.add_noise
NOISE_SIGMA = FLAGS.noise_sigma

USE_GT_NORMALS = FLAGS.use_gt_normals
USE_GT_SEGMENTATION = FLAGS.use_gt_segmentation
USE_GT_BB = FLAGS.use_gt_bb
USE_GT_SKETCH = FLAGS.use_gt_sketch

USE_WHOLE_PC = FLAGS.use_whole_pc
USE_GT_IM = FLAGS.use_gt_im

IS_VISU = FLAGS.visu
USE_GT_IM = FLAGS.use_gt_im
USE_EXTRUSION_AXIS_FEAT = FLAGS.use_extrusion_axis_feat

IM_LOGDIR = FLAGS.im_logdir
IM_CKPT = FLAGS.im_ckpt

LOG_FOUT.write(str(FLAGS)+'\n')

np.random.seed(0)

### For rendering in orionp2
if IS_VISU:
    filename = "render.sh"
    f = open(os.path.join(DUMP_DIR, filename), "w")

    ## To store the output image files
    filename = "image_files.sh"
    g = open(os.path.join(DUMP_DIR, filename), "w")    

    os.makedirs(os.path.join(DUMP_DIR, "point_cloud"), exist_ok=True)
    os.makedirs(os.path.join(DUMP_DIR, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(DUMP_DIR, "rendering_point_cloud"), exist_ok=True)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():

    dataset = AutodeskDataset_h5_sketches(H5_FILENAME, NUM_POINT, NUM_SK_POINT, K, op=False, center=True, with_scale=True)
    if DATA_SPLIT == "test":
        to_shuffle = False
    else:
        to_shuffle = True
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=to_shuffle,
    )

    device = torch.device('cuda')

    MODEL_IMPORTED = importlib.import_module(MODEL)
    shutil.copy('models/%s.py' % MODEL, str(DUMP_DIR))

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

    ##### For IMPLICIT NETWORK
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
    fname = os.path.join(IM_LOGDIR, IM_CKPT)
    implicit_net.load_state_dict(torch.load(fname)["model_state_dict"])
    pn_encoder.load_state_dict(torch.load(fname)["encoder_state_dict"])

    model.to(device)
    implicit_net.to(device)
    pn_encoder.to(device)
    model.eval()
    implicit_net.eval()
    pn_encoder.eval()

    num_evaluated = 0
    total_mIOU = 0.0
    total_normal_difference = 0.0
    total_extrusion_difference = 0.0
    total_centroid_difference = 0.0
    total_pred_fit_cyl_loss = 0.0
    total_pred_fit_glob_loss = 0.0

    total_bb_acc = 0.0

    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
                per_point_extrusion_distances, extrusion_axes, extrusion_distances, extrusion_centers, sampled_sketch, sk_norm_factors  = batch   

            cur_batch_size, _, _ = sampled_pcs.size()
            batch_size = cur_batch_size

            if ADD_NOISE:
                sampled_pcs = add_noise(sampled_pcs, sampled_normals, sigma=NOISE_SIGMA)
                
            ###########
            pcs = [pc.to(device, dtype=torch.float) for pc in sampled_pcs]
            pcs = torch.stack(pcs)

            gt_normals = [n.to(device, dtype=torch.float) for n in sampled_normals]
            gt_normals = torch.stack(gt_normals)

            gt_extrusion_instances = [ex.to(device, dtype=torch.long) for ex in sampled_extrusion_labels]
            gt_extrusion_instances = torch.stack(gt_extrusion_instances)

            gt_bb_labels = [bb.to(device, dtype=torch.float) for bb in sampled_bb_labels]
            gt_bb_labels = torch.stack(gt_bb_labels)

            gt_extrusion_axes = [ax.to(device, dtype=torch.float) for ax in extrusion_axes]
            gt_extrusion_axes = torch.stack(gt_extrusion_axes)

            gt_extrusion_centers = [ax.to(device, dtype=torch.float) for ax in extrusion_centers]
            gt_extrusion_centers = torch.stack(gt_extrusion_centers)

            gt_sketches = [sk.to(device, dtype=torch.float) for sk in sampled_sketch]
            gt_sketches = torch.stack(gt_sketches)         

            gt_sk_norms = [sk_n.to(device, dtype=torch.float) for sk_n in sk_norm_factors]
            gt_sk_norms = torch.stack(gt_sk_norms)                                                     
            ###########

            X, W_raw = model(pcs)

            if PRED_NORMAL:
                X = F.normalize(X, p=2, dim=2, eps=1e-12)
            else:
                #Dummy
                X = torch.zeros((cur_batch_size, NUM_POINT, 3))

            if PRED_SEG and PRED_BB:
                # W : (B, N, K)
                W_2K = torch.softmax(W_raw, dim=2)

                ## 2K classes were predicted, create segmentation pred
                # Barrel
                W_barrel = W_2K[:, :, ::2]
                W_barrel_bb = W_raw[:, :, ::2]
                # Base
                W_base = W_2K[:, :, 1::2]
                W_base_bb = W_raw[:, :, 1::2]

                # For extrusion segmentation loss
                W = W_barrel + W_base
                #W = W_2K[:, :, ::2] + W_2K[:, :, 1::2]

                # Base or barrel segmentation
                '''
                0 for barrel
                1 for base
                ''' 
                BB = torch.zeros(cur_batch_size, NUM_POINT, 2).to(device)                
                for j in range(K):
                    BB[:,:,0] += W_2K[:, :, j*2]
                    BB[:,:,1] += W_2K[:, :, j*2+1]

            elif PRED_SEG:
                W = torch.softmax(W, dim=2)
            else:
                #Dummy
                W = torch.zeros((cur_batch_size, NUM_POINT, K))

            ## For weighted loss
            gt_exlabel_ = gt_extrusion_instances.view(-1)
            weights = F.one_hot(gt_exlabel_, num_classes=K)
            weights = weights.view(cur_batch_size, NUM_POINT, K)
            weights = torch.sum(weights, dim=1).float()

            if PRED_SEG:
                ## Segmentation loss
                W_ = hard_W_encoding(W, to_null_mask=True)

                matching_indices, mask = hungarian_matching(W_, gt_extrusion_instances, with_mask=True)
                mask = mask.float()
                mIoU = compute_segmentation_iou(W_, gt_extrusion_instances, matching_indices, mask)

                ## For visualization
                W_reordered_unmasked = torch.gather(W_, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                W_reordered = torch.where((mask).unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)==1, W_reordered_unmasked, torch.ones_like(W_reordered_unmasked)* -1.)

                label = torch.argmax(W_reordered, dim=-1)
                
            else:
                mIoU = torch.ones(cur_batch_size)


            if PRED_NORMAL:
                ## Normal loss
                normal_difference = compute_normal_difference(X, gt_normals, in_radians=False)

            else:
                normal_difference = torch.zeros(cur_batch_size)


            if PRED_BB:
                pred_bb_label = torch.argmax(BB, dim=-1)

                pred_bb_acc = (pred_bb_label==gt_bb_labels).sum(dim=-1)/float(NUM_POINT)
            else:
                pred_bb_acc = torch.zeros(cur_batch_size)


            if PRED_EXT:
                if USE_GT_NORMALS:
                    EA_X = gt_normals
                else:
                    EA_X = X

                if USE_GT_SEGMENTATION and USE_GT_BB:
                    gt_exlabel_ = gt_extrusion_instances.view(-1)
                    EA_W = F.one_hot(gt_exlabel_, num_classes=K)
                    EA_W = EA_W.view(cur_batch_size, NUM_POINT, K)

                    gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
                    W_barrel_reordered = torch.where(gt_bb_labels_==0, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))
                    W_base_reordered = torch.where(gt_bb_labels_==1, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))


                elif USE_GT_SEGMENTATION:
                    ## GT segmentation, prediction base or barrel
                    gt_exlabel_ = gt_extrusion_instances.view(-1)
                    EA_W = F.one_hot(gt_exlabel_, num_classes=K)
                    EA_W = EA_W.view(cur_batch_size, NUM_POINT, K)

                    pred_bb_labels = torch.argmax(BB, dim=-1)
                    pred_bb_labels = pred_bb_labels.unsqueeze(-1).repeat(1,1,K)
                    W_barrel_reordered = torch.where(pred_bb_labels==0, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))
                    W_base_reordered = torch.where(pred_bb_labels==1, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))

                elif USE_GT_BB:
                    ## GT base/barrel, prediction for segmentation
                    W_ = hard_W_encoding(W, to_null_mask=True)
                    matching_indices, mask = hungarian_matching(W_, gt_extrusion_instances, with_mask=True)
                    W_reordered = torch.gather(W_, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                    EA_W = W_reordered

                    gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1,1,K)
                    W_barrel_reordered = torch.where(gt_bb_labels_==0, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))
                    W_base_reordered = torch.where(gt_bb_labels_==1, EA_W.float(), torch.tensor([0.0]).to(EA_W.device))                    

                else:
                    ## Prediction for all
                    W_ = hard_W_encoding(W, to_null_mask=True)
                    matching_indices, mask = hungarian_matching(W_, gt_extrusion_instances, with_mask=True)

                    EA_W = W_reordered

                    W_barrel_reordered = torch.gather(W_barrel, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                    W_base_reordered = torch.gather(W_base, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK


                E_AX = estimate_extrusion_axis(EA_X, W_barrel_reordered, W_base_reordered, gt_bb_labels, gt_extrusion_instances, normalize=NORM_EIG)
                extrusion_difference = compute_normal_difference(E_AX, gt_extrusion_axes, in_radians=False, collapse=False)

                # Only calculate difference for existing 
                mask_gt = get_mask_gt(gt_extrusion_instances, K)

                extrusion_difference_uncollapsed = torch.where(mask_gt, extrusion_difference, torch.zeros_like(extrusion_difference))

                extrusion_difference = reduce_mean_masked_instance(extrusion_difference, mask_gt)

                ## Extrusion centers
                ## For center prediction
                predicted_centroids = torch.zeros((cur_batch_size, K, 3)).to(gt_extrusion_centers.device)
                found_centers_mask = torch.zeros((cur_batch_size, K)).to(gt_extrusion_centers.device)
                
                ## Calculate centroids of each segment
                for j in range(K):
                    ### Get points on the segment
                    curr_segment_W = EA_W[:, :, j]
                    indices_pred = curr_segment_W==1
                    indices_pred = indices_pred.nonzero()

                    for b in range(cur_batch_size):
                        ## get indices in current point cloud
                        curr_batch_idx = indices_pred[:,0]==b
                        
                        ## No points found in segment (1 point found is considered no points to handle .squeeze() function)
                        if (curr_batch_idx.nonzero().shape[0]<=1):
                            found_centers_mask[b,j] = 0.0
                            continue

                        curr_batch_idx = curr_batch_idx.nonzero().squeeze()
                        curr_batch_pt_idx = indices_pred[:,1][curr_batch_idx]
                        curr_segment_pc = torch.gather(pcs[b,:,:], 0, curr_batch_pt_idx.unsqueeze(-1).repeat(1,3))

                        ## Get center
                        pred_centroid = torch.mean(curr_segment_pc, axis=0)

                        predicted_centroids[b, j, :] = pred_centroid
                        found_centers_mask[b,j] = 1.0


                centroid_diff = torch.square(predicted_centroids - gt_extrusion_centers).sum(dim=-1)

                ## Take mean if found both in ground truth and in prediction
                centroid_difference_uncollapsed = found_centers_mask * centroid_diff
                centroid_difference = torch.mean(found_centers_mask * centroid_diff, dim=-1)

                centroid_difference_uncollapsed = torch.where(mask_gt, centroid_diff, torch.zeros_like(centroid_diff))
                centroid_difference = reduce_mean_masked_instance(centroid_diff, mask_gt)

            else:
                extrusion_difference = torch.zeros(cur_batch_size)
                centroid_difference = torch.zeros(cur_batch_size)
                extrusion_difference_uncollapsed = torch.zeros(cur_batch_size, K)
                centroid_difference_uncollapsed = torch.zeros(cur_batch_size, K)


            ###### Get extrusion extent (along the axis and center)
            extents, _ = get_extrusion_extents(pcs, gt_extrusion_instances, gt_bb_labels, gt_extrusion_axes, gt_extrusion_centers, num_points_to_sample=NUM_SK_POINT) ## Change to predicted
            extents = extents.permute(1,0,2)

            ###### Run implicit
            sk_pnts = gt_sketches[:, :, :, :2].view(cur_batch_size*K, NUM_SK_POINT, 2)
            sk_normals = gt_sketches[:, :, :, -2:].view(cur_batch_size*K, NUM_SK_POINT, 2)

            if not USE_GT_IM:

                W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                W_reordered = torch.where((mask).unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)==1, W_reordered, torch.zeros_like(W_reordered))
                
                if USE_WHOLE_PC:
                    # print("Using whole pc in encoding")

                    pcs_repreated = pcs.unsqueeze(1).repeat(1,K,1,1)

                    W_reordered_p = W_reordered.permute(0,2,1)
                    W_reordered_p = W_reordered_p.unsqueeze(-1)

                    if USE_EXTRUSION_AXIS_FEAT:
                        extrusion_axis_repeated = E_AX.unsqueeze(-2).repeat(1,1,NUM_POINT,1)
                        global_pc = torch.cat((pcs_repreated, W_reordered_p, extrusion_axis_repeated), dim=-1)
                        out_dim = 7
                    
                    else:
                        global_pc = torch.cat((pcs_repreated, W_reordered_p), dim=-1)
                        out_dim = 4

                    global_pc = global_pc.reshape(cur_batch_size*K, -1, out_dim)              
                    latent_codes = pn_encoder(global_pc)

                else:           
                    label = torch.argmax(W_reordered, dim=-1)

                    ## Use prediction base/barrel
                    BB = torch.zeros(cur_batch_size, NUM_POINT, 2).to(device)                
                    for j in range(K):
                        BB[:,:,0] += W_2K[:, :, j*2]
                        BB[:,:,1] += W_2K[:, :, j*2+1]
                    pred_bb_label = torch.argmax(BB, dim=-1)

                    pred_projected_pc, pred_projected_normal, pred_scales = sketch_implicit_projection(pcs, X, label, pred_bb_label, E_AX, predicted_centroids, num_points_to_sample=NUM_SK_POINT) # Use all predictions for projection
                    pred_projected_pc /= pred_scales.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_projected_pc.shape[-2], pred_projected_pc.shape[-1])

                    pred_projected_pc = pred_projected_pc.reshape(cur_batch_size*K, NUM_SK_POINT, 2)
                    pred_projected_normal = pred_projected_normal.reshape(cur_batch_size*K, NUM_SK_POINT, 2)

                    global_pc = torch.cat((pred_projected_pc, pred_projected_normal), dim=-1)
                    latent_codes = pn_encoder(global_pc)

            else:
                ## Use GT labels
                ## gt_extrusion_instances, gt_bb_labels

                if USE_WHOLE_PC:
                    pcs_repreated = pcs.unsqueeze(1).repeat(1,K,1,1)

                    exlabel_ = gt_extrusion_instances.view(-1)
                    gt_EA_W = F.one_hot(exlabel_, num_classes=K)
                    gt_EA_W = gt_EA_W.view(cur_batch_size, -1, K).float()

                    gt_EA_W = gt_EA_W.permute(0,2,1)
                    gt_EA_W = gt_EA_W.unsqueeze(-1)

                    ## Append extrusion axis
                    if USE_EXTRUSION_AXIS_FEAT:
                        extrusion_axis_repeated = gt_extrusion_axes.unsqueeze(-2).repeat(1,1,NUM_POINT,1)
                        global_pc = torch.cat((pcs_repreated, gt_EA_W, extrusion_axis_repeated), dim=-1)
                        out_dim = 7
                    else:
                        global_pc = torch.cat((pcs_repreated, gt_EA_W), dim=-1)
                        out_dim = 4

                    global_pc = global_pc.reshape(cur_batch_size*K, -1, out_dim)              
                    latent_codes = pn_encoder(global_pc)

                else:
                    pred_projected_pc, pred_projected_normal, pred_scales = sketch_implicit_projection(pcs, gt_normals, gt_extrusion_instances, gt_bb_labels, gt_extrusion_axes, gt_extrusion_centers, num_points_to_sample=NUM_SK_POINT)

                    pred_scales = pred_scales.unsqueeze(-1).unsqueeze(-1).repeat(1,1, pred_projected_pc.shape[-2], pred_projected_pc.shape[-1])
                    pred_projected_pc /= pred_scales

                    pred_projected_pc = pred_projected_pc.reshape(cur_batch_size*K, NUM_SK_POINT, 2)
                    pred_projected_normal = pred_projected_normal.reshape(cur_batch_size*K, NUM_SK_POINT, 2)

                    global_pc = torch.cat((pred_projected_pc, pred_projected_normal), dim=-1) 
                    latent_codes = pn_encoder(global_pc)


            sk_pnts = sk_pnts.reshape(cur_batch_size, K , NUM_SK_POINT, 2)

            ## Get mask which sketches to predict as part of an extrusion
            mask_gt = get_mask_gt(gt_extrusion_instances, K)
            mask_gt = mask_gt.to("cpu").detach().numpy()
            ###################
 
            pred_projected_pc, pred_projected_normal, _, pred_found_mask  = sketch_implicit_projection2(pcs, gt_normals, gt_extrusion_instances, gt_bb_labels, E_AX, predicted_centroids, num_points_to_sample=NUM_SK_POINT)

            pred_projected_pc /= pred_scales.unsqueeze(-1).unsqueeze(-1)

            pred_projected_pc = pred_projected_pc.view(cur_batch_size*K, NUM_SK_POINT, 2)

            pred_net_input = add_latent(pred_projected_pc, latent_codes)

            pred_sk_out = implicit_net(pred_net_input).reshape(K, cur_batch_size, NUM_SK_POINT)

            pred_weighted_sk_out = pred_sk_out * pred_scales.unsqueeze(-1)

            pred_mask = mask.T * pred_found_mask.T

            pred_mask = pred_mask.unsqueeze(-1).repeat(1, 1, NUM_SK_POINT)

            num_gt_extrusion_instances = torch.max(gt_extrusion_instances, 1)[0] + 1
            pred_sk_out_for_im = pred_sk_out * pred_mask

            pred_sk_out_for_im = pred_sk_out_for_im.abs().permute(1,0,2).mean(-1).reshape(batch_size, -1).sum(1)
            pred_fit_cyl_loss = pred_sk_out_for_im / num_gt_extrusion_instances
            pred_fit_cyl_loss = pred_fit_cyl_loss.to("cpu").detach().numpy()

            pred_projected_pc, pred_projected_normal, _, pred_found_mask = sketch_implicit_projection3(pcs, gt_normals, gt_extrusion_instances, gt_bb_labels, E_AX, predicted_centroids)
            pred_projected_pc /= pred_scales.unsqueeze(-1).unsqueeze(-1)
            pred_projected_pc = pred_projected_pc.reshape(batch_size*K, 8192, 2)
            pred_projected_normal = pred_projected_normal.reshape(batch_size*K, 8192, 2)
            pred_net_input = add_latent(pred_projected_pc, latent_codes)
            pred_sk_out = implicit_net(pred_net_input).reshape(K, batch_size, 8192)

            pred_mask = mask.T * pred_found_mask.T
            pred_mask = pred_mask.unsqueeze(-1).repeat(1, 1, 8192)

            pred_sk_out = torch.where(pred_mask==1, pred_sk_out.abs().float(), torch.tensor([10000.0]).to(pred_sk_out.device))
            pred_fit_glob_loss, _ = torch.min(pred_sk_out, axis=0)
            weight_mask = torch.ones_like(gt_bb_labels).to(gt_bb_labels.device) - gt_bb_labels
            pred_fit_glob_loss *= weight_mask
            pred_fit_glob_loss = pred_fit_glob_loss.sum(1) / (8192 - gt_bb_labels.sum(1))

            ################

            latent_codes = latent_codes.reshape(cur_batch_size, K , -1)

            ## Aggregate losses
            mIoU = mIoU.to("cpu")
            mIoU = mIoU.detach().numpy()

            normal_difference = normal_difference.to("cpu")
            normal_difference = normal_difference.detach().numpy()
            extrusion_difference = extrusion_difference.to("cpu")
            extrusion_difference = extrusion_difference.detach().numpy()

            centroid_difference = centroid_difference.to("cpu")
            centroid_difference = centroid_difference.detach().numpy()

            pred_fit_glob_loss = pred_fit_glob_loss.to("cpu")
            pred_fit_glob_loss = pred_fit_glob_loss.detach().numpy()  


            pred_bb_acc = pred_bb_acc.to("cpu")
            pred_bb_acc = pred_bb_acc.detach().numpy() 

            gt_bb_labels = gt_bb_labels.to("cpu")
            gt_bb_labels = gt_bb_labels.detach().numpy() 

            ### for base-barrel visualization
            if PRED_BB:
                pred_bb_label = pred_bb_label.to("cpu")
                pred_bb_label = pred_bb_label.detach().numpy()

            ## To debug extrusion axis prediction from gt
            extrusion_difference_uncollapsed = extrusion_difference_uncollapsed.to("cpu")
            extrusion_difference_uncollapsed = extrusion_difference_uncollapsed.detach().numpy()            

            ## For visualization
            pcs = pcs.to("cpu")
            pcs = pcs.detach().numpy()            
            label = label.to("cpu")
            label = label.detach().numpy()
            gt_extrusion_instances = gt_extrusion_instances.to("cpu")
            gt_extrusion_instances = gt_extrusion_instances.detach().numpy()

            ### To output to pickle
            predicted_centroids = predicted_centroids.to("cpu").detach().numpy()
            E_AX = E_AX.to("cpu").detach().numpy()
            extents = extents.to("cpu").detach().numpy()
            gt_sk_norms = gt_sk_norms.to("cpu").detach().numpy()
            gt_sketches = gt_sketches[:, :, :, :2].to("cpu").detach().numpy()
      


            for j in range(mIoU.shape[0]):
                num_evaluated += 1

                total_mIOU += mIoU[j]
                total_normal_difference += normal_difference[j]
                total_extrusion_difference += extrusion_difference[j]
                total_bb_acc += pred_bb_acc[j]

                ## For centroid
                total_centroid_difference += centroid_difference[j]

                total_pred_fit_glob_loss += pred_fit_glob_loss[j]

                total_pred_fit_cyl_loss += pred_fit_cyl_loss[j]

                if IS_VISU:
                    if PRED_BB:
                        visualize_segmentation_pc_bb_v2(str(i)+"_"+str(j)+"_"+str(mIoU[j]), DUMP_DIR, pcs[j], label[j], gt_extrusion_instances[j], pred_bb_label[j], gt_bb_labels[j], f, g)

                    else:
                        visualize_segmentation_pc(str(i)+"_"+str(j), DUMP_DIR, pcs[j], label[j], gt_extrusion_instances[j], f, g)


                    ## Write image filename for implicits
                    imagefile_line = ""

                    # Implcit visu
                    for k in range(mask_gt.shape[1]):
                        ## check if exist or not
                        if not mask_gt[j,k]:
                            continue

                        filename = '{0}/igr_{1}_{2}'.format(plot_fol, str(i)+"_"+str(j), str(k)) + " "
                        imagefile_line += filename

                        pnts = sk_pnts[j, k].to("cpu").detach().numpy()

                        curr_latent = latent_codes[j,k]

                        plot_surface_2d(decoder=implicit_net,
                                     path=plot_fol,
                                     epoch=str(i)+"_"+str(j),
                                     shapename=str(k),
                                     points=pnts,
                                     latent=curr_latent,
                                     resolution=512,mc_value=0.0,is_uniform_grid=True,verbose=False,save_html=False,save_ply=False,overwrite=True)

                    imagefile_line += "\n"
                    g.write(imagefile_line)

            if (i%20==0):
                print("Time elapsed: "+str(time.time()-start_time)+" sec for batch "+str(i)+ "/"+ str(len(loader))+".")                

        mean_mIOU = total_mIOU/float(num_evaluated)
        mean_normal_difference = total_normal_difference/float(num_evaluated)
        mean_extrusion_difference = total_extrusion_difference/float(num_evaluated)
        mean_centroid_difference = total_centroid_difference/float(num_evaluated)
        mean_bb_acc = total_bb_acc/float(num_evaluated)
        mean_pred_fit_glob_loss = total_pred_fit_glob_loss/float(num_evaluated)
        mean_pred_fit_cyl_loss = total_pred_fit_cyl_loss/float(num_evaluated)

        log_string("=" * 20)
        log_string("")
        log_string("Num evaluated= "+str(num_evaluated))
        log_string("")
        log_string("Mean mIOU= "+str(mean_mIOU))
        log_string("")             
        log_string("Mean normal angle error (degrees) = "+str(mean_normal_difference))
        log_string("")     
        log_string("Mean base/barrel accuracy= "+str(mean_bb_acc))
        log_string("")                
        log_string("Mean extrusion angle error (degrees) = "+str(mean_extrusion_difference))
        log_string("") 
        log_string("Mean centroid difference = "+str(mean_centroid_difference))
        log_string("")           
        log_string("Mean per-extrusion cylinder fitting loss= "+str(mean_pred_fit_cyl_loss))
        log_string("") 
        log_string("Mean global fitting loss= "+str(mean_pred_fit_glob_loss))
        log_string("") 

        if IS_VISU:
            f.close()
            g.close()
        LOG_FOUT.close()

if __name__ == '__main__':
    main()

