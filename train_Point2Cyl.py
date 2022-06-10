# Mikaela Uy (mikacuy@cs.stanford.edu)
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import datetime
import sys
import importlib
import shutil
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))

## For implicit
sys.path.append(os.path.join(BASE_DIR, 'IGR'))
from sampler import *
from network import *
from general import *
from plots import plot_surface_2d

from utils import * 
from data_utils import *
from dataloader import AutodeskDataset_h5_sketches
from losses import *

### For tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pointnet_extrusion', help='model name')
parser.add_argument('--num_point', type=int,  default=8192, help='Point Number [default: 8192]')
parser.add_argument('--num_sk_point', type=int,  default=2048, help='Point Number [default: 2048]')
parser.add_argument('--K', type=int,  default=8, help='Max number of extrusions')
parser.add_argument('--batch_size', type=int,  default=4, help='batch size')

parser.add_argument("--logdir", default="Point2Cyl", help="path to the log directory", type=str)
parser.add_argument('--data_dir', type=str, default='data/')

parser.add_argument('--data_split', default= "train", type=str)

parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--bn_decay_step', type=int, default=200000, help='Decay step for bn decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')

parser.add_argument('--pred_seg', action='store_true')
parser.add_argument('--pred_normal', action='store_true')
parser.add_argument('--pred_bb', action='store_true')
parser.add_argument('--pred_extrusion', action='store_true')
parser.add_argument('--pred_center', action='store_true')
parser.add_argument('--norm_eig', action='store_true')

parser.add_argument('--weight_seg', type=float, default=1.0, help='Weight for extrusion segmentation loss.')
parser.add_argument('--weight_normal', type=float, default=1.0, help='Weight for normal loss')
parser.add_argument('--weight_bb', type=float, default=1.0, help='Weight for base/barrel loss')
parser.add_argument('--weight_extrusion', type=float, default=1.0, help='Weight for extrusion axis loss')
parser.add_argument('--weight_center', type=float, default=1.0, help='Weight for center loss.')

parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--noise_sigma', type=float, default=0.01, help='Sigma for random noise addition.')
parser.add_argument('--sald', action='store_true', help='sald for normal loss')

## Load ckpt
parser.add_argument('--is_pc_init', action='store_true')
parser.add_argument('--is_im_init', action='store_true')

parser.add_argument('--is_pc_train', action='store_true')
parser.add_argument('--is_im_train', action='store_true')
parser.add_argument('--is_implicitnet_train', action='store_true')

parser.add_argument("--pc_logdir", default="Point2Cyl_without_sketch", help="path to the log directory", type=str)
parser.add_argument("--pc_ckpt", default="model.pth", help="checkpoint", type=str)

parser.add_argument("--im_logdir", default="./results/IGR_dense/", help="path to the log directory", type=str)
parser.add_argument("--im_ckpt", default="latest.pth", help="checkpoint", type=str)

parser.add_argument('--is_L2', action='store_true')
parser.add_argument('--with_im_loss', action='store_true')
parser.add_argument('--use_whole_pc', action='store_true')
parser.add_argument('--use_gt_im', action='store_true')
parser.add_argument('--use_extrusion_axis_feat', action='store_true')

##

FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

DATA_SPLIT = FLAGS.data_split
DATA_DIR = FLAGS.data_dir

H5_FILENAME = os.path.join(DATA_DIR, DATA_SPLIT + ".h5")

NUM_POINT = FLAGS.num_point
NUM_SK_POINT = FLAGS.num_sk_point
MODEL = FLAGS.model
K = FLAGS.K
BATCH_SIZE = FLAGS.batch_size

PRED_SEG = FLAGS.pred_seg
PRED_NORMAL = FLAGS.pred_normal
PRED_BB = FLAGS.pred_bb
PRED_EXT = FLAGS.pred_extrusion
PRED_CENTER = FLAGS.pred_center
SALD = FLAGS.sald

NORM_EIG = FLAGS.norm_eig

NUM_EPOCHS = FLAGS.num_epochs
DECAY_STEP = FLAGS.decay_step
BN_DECAY_STEP = FLAGS.bn_decay_step
DECAY_RATE = FLAGS.decay_rate
LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum

ADD_NOISE = FLAGS.add_noise
NOISE_SIGMA = FLAGS.noise_sigma

WEIGHT_SEG = FLAGS.weight_seg
WEIGHT_NORMAL = FLAGS.weight_normal
WEIGHT_BB = FLAGS.weight_bb
WEIGHT_EXTRUSION = FLAGS.weight_extrusion
WEIGHT_CENTER = FLAGS.weight_center

### Load pre-trained models ###
PC_LOGDIR = FLAGS.pc_logdir
PC_CKPT = FLAGS.pc_ckpt
IS_PC_INIT = FLAGS.is_pc_init
IS_PC_TRAIN = FLAGS.is_pc_train

IM_LOGDIR = FLAGS.im_logdir
IM_CKPT = FLAGS.im_ckpt
IS_IM_INIT = FLAGS.is_im_init
IS_IM_TRAIN = FLAGS.is_im_train
IS_IMPLICITNET_TRAIN = FLAGS.is_implicitnet_train
######

IS_L2 = FLAGS.is_L2
USE_WHOLE_PC = FLAGS.use_whole_pc
WITH_IM_LOSS = FLAGS.with_im_loss
USE_GT_IM = FLAGS.use_gt_im
USE_EXTRUSION_AXIS_FEAT = FLAGS.use_extrusion_axis_feat

LOG_FOUT.write(str(FLAGS)+'\n')

if PRED_NORMAL:
	normal_loss_multiplier = WEIGHT_NORMAL
else:
	normal_loss_multiplier = 0.0

if PRED_SEG:
	miou_loss_multiplier = WEIGHT_SEG
else:
	miou_loss_multiplier = 0.0

if PRED_EXT:
	extrusion_loss_multiplier = WEIGHT_EXTRUSION
else:
	extrusion_loss_multiplier = 0.0	

if PRED_BB:
	bb_loss_multiplier = WEIGHT_BB
else:	
	bb_loss_multiplier = 0.0

if PRED_CENTER:
    center_loss_multiplier = WEIGHT_CENTER
else:
    center_loss_multiplier = 0.0

## For summary writer
writer = SummaryWriter("runs/"+LOG_DIR)

np.random.seed(0)

def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

# BN Decay
def get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True):
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    p = global_step * batch_size / bn_decay_step
    if staircase:
        p = int(np.floor(p))
    bn_momentum = max(BN_INIT_DECAY * (BN_DECAY_RATE ** p), 1-BN_DECAY_CLIP)
    return bn_momentum

def update_momentum(module, bn_momentum):
    for name, module_ in module.named_modules():
        if 'bn' in name:
            module_.momentum = bn_momentum

# LR Decay
def get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True):
    p = global_step * batch_size / decay_step
    if staircase:
        p = int(np.floor(p))
    learning_rate = init_learning_rate * (decay_rate ** p)
    return learning_rate

def main():
    dataset = AutodeskDataset_h5_sketches(H5_FILENAME, NUM_POINT, NUM_SK_POINT, K, op=False, center=True, extent=False)
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
    shutil.copy('models/%s.py' % MODEL, str(LOG_DIR))

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

    if not USE_GT_IM:
        model.to(device)

    # Optimizer
    init_learning_rate = LEARNING_RATE

    ##### Switch optimizer, adagrad
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    ##### For IMPLICIT NETWORK
    GLOBAL_SIGMA = 1.8
    LOCAL_SIGMA = 0.01
    D_IN = 2
    LATENT_SIZE = 256

    sampler = NormalPerPoint(GLOBAL_SIGMA, LOCAL_SIGMA)
    # ## Implicit
    implicit_net = ImplicitNet(d_in=D_IN+LATENT_SIZE, dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ], skip_in = [4], geometric_init= True, radius_init = 1, beta=100)
    implicit_net.to(device)	

    ## PointNet
    if not USE_WHOLE_PC:
        pn_encoder = PointNetEncoder(LATENT_SIZE, D_IN, with_normals=True)
    else:
        if USE_EXTRUSION_AXIS_FEAT:
            print("Using extrusion axis feat")
            pn_encoder = PointNetEncoder(LATENT_SIZE, 7, with_normals=False) ## 3d pc plus confidence mask, plus extrusion axis
        else:
            print("Using seg label feat only")
            pn_encoder = PointNetEncoder(LATENT_SIZE, 4, with_normals=False) ## 3d pc plus confidence mask, plus extrusion axis

    pn_encoder.to(device)

    loaded_pn_encoder = PointNetEncoder(LATENT_SIZE, D_IN, with_normals=True)
    loaded_pn_encoder.to(device)

    im_lr_schedules = get_learning_rate_schedules([
        {
            "Type" : "Step",
            "Initial" : 0.001,
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

    if IS_PC_TRAIN and IS_IM_TRAIN:
        optimizer = torch.optim.Adam([
            {
                "params": model.parameters(), 
                "lr": init_learning_rate
            },
            {
                "params": pn_encoder.parameters(),
                "lr": im_lr_schedules[1].get_learning_rate(0)
            }])
    elif IS_PC_TRAIN:
        print("Only pc net.")
        optimizer = torch.optim.Adam([
            {
                "params": model.parameters(),
                "lr": init_learning_rate
            }])
    else:
        print("Only implicit net.")
        optimizer = torch.optim.Adam([ 
            {
                "params": pn_encoder.parameters(),
                "lr": im_lr_schedules[1].get_learning_rate(0)
            }])			
    #######################

    global_step = 0
    old_learning_rate = init_learning_rate
    old_bn_momentum = MOMENTUM

    ### Load models
    if IS_PC_INIT:
        fname = os.path.join(PC_LOGDIR, PC_CKPT)
        model.load_state_dict(torch.load(fname)["model"])
        print("3D model loaded.")
	
    if IS_IM_INIT:
        fname = os.path.join(IM_LOGDIR, IM_CKPT)	
        pn_encoder.load_state_dict(torch.load(fname)["encoder_state_dict"])	
        print("Implicit model loaded.")
    #######

    ### Load pre-trained model
    fname = os.path.join(IM_LOGDIR, IM_CKPT)
    implicit_net.load_state_dict(torch.load(fname)["model_state_dict"])	
    loaded_pn_encoder.load_state_dict(torch.load(fname)["encoder_state_dict"])	
    print("Pre-trained fixed implicit model loaded.")

    ## Save initial combined model
    fname = os.path.join(LOG_DIR, "model.pth")
    print("> Saving model to {}...".format(fname))
    model_to_save = {"model": model.state_dict(), "implicit_net": implicit_net.state_dict(), "pn_encoder": pn_encoder.state_dict()}
    torch.save(model_to_save, fname)
    # exit()	

    if not USE_GT_IM:
        if IS_PC_TRAIN:
            model.train()
        else:
            model.eval()

    if IS_IM_TRAIN:
        pn_encoder.train()
    else:
        pn_encoder.eval()
	
    implicit_net.eval()
    loaded_pn_encoder.eval()

    best_loss = np.Inf

    for epoch in range(1, NUM_EPOCHS+1):
        start = datetime.datetime.now()
        scalars = defaultdict(list)
        for i, batch in enumerate(loader):
            sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
                per_point_extrusion_distances, extrusion_axes, extrusion_distances, extrusion_centers, sampled_sketch = batch

            batch_size, _, _ = sampled_pcs.size()

            if ADD_NOISE:
                sampled_pcs = add_noise(sampled_pcs, sampled_normals, sigma=NOISE_SIGMA)

            ###########
            pcs = [pc.to(device, dtype=torch.float) for pc in sampled_pcs]
            pcs = torch.stack(pcs)

            gt_normals = [n.to(device, dtype=torch.float) for n in sampled_normals]
            gt_normals = torch.stack(gt_normals)

            gt_extrusion_instances = [ex.to(device, dtype=torch.long) for ex in sampled_extrusion_labels]
            gt_extrusion_instances = torch.stack(gt_extrusion_instances)

            gt_bb_labels = [bb.to(device, dtype=torch.long) for bb in sampled_bb_labels]
            gt_bb_labels = torch.stack(gt_bb_labels)

            gt_extrusion_axes = [ax.to(device, dtype=torch.float) for ax in extrusion_axes]
            gt_extrusion_axes = torch.stack(gt_extrusion_axes)

            gt_extrusion_centers = [c.to(device, dtype=torch.float) for c in extrusion_centers]
            gt_extrusion_centers = torch.stack(gt_extrusion_centers)

            gt_sketches = [sk.to(device, dtype=torch.float) for sk in sampled_sketch]
            gt_sketches = torch.stack(gt_sketches)

            mask_gt = get_mask_gt(gt_extrusion_instances, K)

            if not USE_GT_IM:
                #X, W_raw, O, _, _ = model(pcs)
                X, W_raw = model(pcs)

                if PRED_NORMAL:
                    X = F.normalize(X, p=2, dim=2, eps=1e-12)
                else:
                    #Dummy
                    X = torch.zeros((batch_size, NUM_POINT, 3))

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

                    ## Base and barrel loss as mIOU
                    ## Create base-barrel as separate classes
                    gt_extbb_instances = gt_extrusion_instances + gt_bb_labels*K

                elif PRED_SEG:
                    W = torch.softmax(W_raw, dim=2)
                else:
                    #Dummy
                    W = torch.zeros((batch_size, NUM_POINT, K))

                #### Compute segmentation and normal losses
                total_loss, total_normal_loss, total_miou_loss, matching_indices, mask = compute_all_losses(pcs, W, gt_extrusion_instances, X, gt_normals, normal_loss_multiplier, miou_loss_multiplier, return_match_indices=True)			

                # To compute for base and barrel loss
                if (PRED_BB):
                    #### Compute base-barrel segmentation loss
                    #### mIOU from W_2K and gt_extbb_instances
                    cur_batch_size, _, _ = sampled_pcs.size()
                    W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                    mask = mask.float()
                    W_reordered = torch.where((mask).unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)==1, W_reordered, torch.zeros_like(W_reordered))

                    W_reordered = torch.softmax(W_reordered, dim=-1)

                    W_sorted, label = torch.sort(W_reordered, dim=-1)

                    segment_barrel_confidence = torch.gather(W_barrel_bb, 2, label) # BxNx1
                    segment_base_confidence = torch.gather(W_base_bb, 2, label) # BxNx1

                    BB_segment = torch.cat((segment_barrel_confidence.unsqueeze(-1), segment_base_confidence.unsqueeze(-1)), dim=-1)

                    gt_bb_labels_ = gt_bb_labels.unsqueeze(-1).repeat(1, 1, K)

                    total_bb_loss = F.cross_entropy(BB_segment.contiguous().view(batch_size*NUM_POINT*K, -1), gt_bb_labels_.view(batch_size*NUM_POINT*K), reduction='none')
                    total_bb_loss = total_bb_loss.view(batch_size, NUM_POINT, K)
                    total_bb_loss = torch.sum(total_bb_loss * W_sorted, dim=-1)

                    total_bb_loss = torch.mean(torch.mean(total_bb_loss, dim=-1))

                else:
                    total_bb_loss = torch.zeros([batch_size]).to(gt_extrusion_axes.device)

                total_bb_loss = torch.mean(total_bb_loss)
                total_loss += bb_loss_multiplier * total_bb_loss

                ###### Calculate extrusion axis loss using joint base/barrel formulation
                if (PRED_NORMAL and PRED_BB and PRED_EXT):
                # if epoch>100 and (PRED_NORMAL and PRED_BB and PRED_EXT):
                    # Calculate extrusion axis with normals, pred_seg and pred_bb

                    # matching_indices, mask = hungarian_matching(W, gt_extrusion_instances, with_mask=True)

                    W_barrel_reordered = torch.gather(W_barrel, 2, matching_indices.unsqueeze(1).expand(batch_size, NUM_POINT, K)) # BxNxK
                    W_base_reordered = torch.gather(W_base, 2, matching_indices.unsqueeze(1).expand(batch_size, NUM_POINT, K)) # BxNxK

                    E_AX = estimate_extrusion_axis(X, W_barrel_reordered, W_base_reordered, gt_bb_labels, gt_extrusion_instances, normalize=NORM_EIG)

                    ### Use angle loss with ground truth extrusion
                    extrusion_loss = compute_normal_loss(E_AX, gt_extrusion_axes, angle_diff=False, collapse=False)
                    # Only calculate loss for existing 
                    avg_extrusion_loss = reduce_mean_masked_instance(extrusion_loss, mask_gt)

                else:
                    # Zero loss
                    avg_extrusion_loss = torch.zeros([batch_size, K]).to(gt_extrusion_axes.device)

                total_extrusion_loss = torch.mean(avg_extrusion_loss)*extrusion_loss_multiplier
                total_loss += total_extrusion_loss

                ### Center loss
                if PRED_CENTER:
                    W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K))
                    predicted_centroids = estimate_extrusion_centers(W_reordered, pcs)

                    centroid_diff = torch.square(predicted_centroids - gt_extrusion_centers).sum(dim=-1)
                    avg_center_loss = reduce_mean_masked_instance(centroid_diff, mask_gt)

                else:
                    avg_center_loss = torch.zeros([batch_size]).to(gt_extrusion_axes.device)

                total_center_loss = torch.mean(avg_center_loss) * center_loss_multiplier
                total_loss += total_center_loss				


            ###### Implicit network
            ## Get latent code

            if not USE_GT_IM:
                W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)) # BxNxK
                W_reordered = torch.where((mask).unsqueeze(1).expand(cur_batch_size, NUM_POINT, K)==1, W_reordered, torch.zeros_like(W_reordered))

                if USE_WHOLE_PC:
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

                    global_pc = global_pc.reshape(batch_size*K, -1, out_dim)				
                    latent_codes = pn_encoder(global_pc)

                else:
                    label = torch.argmax(W_reordered, dim=-1)

                    ## Use prediction base/barrel
                    BB = torch.zeros(cur_batch_size, NUM_POINT, 2).to(device)                
                    for j in range(K):
                        BB[:,:,0] += W_2K[:, :, j*2]
                        BB[:,:,1] += W_2K[:, :, j*2+1]
                    pred_bb_label = torch.argmax(BB, dim=-1)

                    pred_projected_pc, pred_projected_normal, pred_scales = sketch_implicit_projection(pcs, X, label, pred_bb_label, gt_extrusion_axes, gt_extrusion_centers, num_points_to_sample=NUM_SK_POINT)
                    gt_projected_pc, gt_projected_normal, gt_scales = sketch_implicit_projection(pcs, gt_normals, gt_extrusion_instances, gt_bb_labels, gt_extrusion_axes, gt_extrusion_centers, num_points_to_sample=NUM_SK_POINT)

                    gt_scales = gt_scales.unsqueeze(-1).unsqueeze(-1).repeat(1,1, pred_projected_pc.shape[-2], pred_projected_pc.shape[-1])
                    pred_projected_pc /= gt_scales

                    pred_projected_pc = pred_projected_pc.reshape(batch_size*K, NUM_SK_POINT, 2)
                    pred_projected_normal = pred_projected_normal.reshape(batch_size*K, NUM_SK_POINT, 2)

                    global_pc = torch.cat((pred_projected_pc, pred_projected_normal), dim=-1) 
                    latent_codes = pn_encoder(global_pc)

            else:
                ## Use GT labels
                ## gt_extrusion_instances, gt_bb_labels

                if USE_WHOLE_PC:
                    pcs_repreated = pcs.unsqueeze(1).repeat(1,K,1,1)

                    exlabel_ = gt_extrusion_instances.view(-1)
                    gt_EA_W = F.one_hot(exlabel_, num_classes=K)
                    gt_EA_W = gt_EA_W.view(batch_size, -1, K).float()

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

                    global_pc = global_pc.reshape(batch_size*K, -1, out_dim)
                    latent_codes = pn_encoder(global_pc)

                else:
                    pred_projected_pc, pred_projected_normal, pred_scales = sketch_implicit_projection(pcs, gt_normals, gt_extrusion_instances, gt_bb_labels, gt_extrusion_axes, gt_extrusion_centers, num_points_to_sample=NUM_SK_POINT)

                    pred_scales = pred_scales.unsqueeze(-1).unsqueeze(-1).repeat(1,1, pred_projected_pc.shape[-2], pred_projected_pc.shape[-1])
                    pred_projected_pc /= pred_scales

                    pred_projected_pc = pred_projected_pc.reshape(batch_size*K, NUM_SK_POINT, 2)
                    pred_projected_normal = pred_projected_normal.reshape(batch_size*K, NUM_SK_POINT, 2)

                    global_pc = torch.cat((pred_projected_pc, pred_projected_normal), dim=-1)
                    latent_codes = pn_encoder(global_pc)


            sk_pnts = gt_sketches[:, :, :, :2].view(batch_size*K, NUM_SK_POINT, 2)
            sk_normals = gt_sketches[:, :, :, -2:].view(batch_size*K, NUM_SK_POINT, 2)
            global_pc_gt = torch.cat((sk_pnts, sk_normals), dim=-1) ### Change this to encode segmentation prediction
            latent_codes_gt = loaded_pn_encoder(global_pc_gt)
		

            if WITH_IM_LOSS:
                nonmnfld_pnts = sampler.get_points(sk_pnts)

                ### Sketch fitting loss
                sk_pnts = add_latent(sk_pnts, latent_codes)		
                nonmnfld_pnts = add_latent(nonmnfld_pnts, latent_codes)

                # forward pass
                sk_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()

                sk_pred = implicit_net(sk_pnts)
                nonmnfld_pred = implicit_net(nonmnfld_pnts)

                mnfld_grad = gradient(sk_pnts, sk_pred)
                nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

                sk_pred = sk_pred.reshape(batch_size, K, -1, 1)
                nonmnfld_grad = nonmnfld_grad.reshape(batch_size, K, -1, 2)
                mnfld_grad = mnfld_grad.reshape(batch_size, K, -1, 2)
                sk_normals = sk_normals.reshape(batch_size, K, -1, 2)

                mnfld_loss = (sk_pred.abs()).mean(dim=-1).mean(dim=-1)
                mnfld_loss = reduce_mean_masked_instance(mnfld_loss, mask_gt).mean()
                # print(mnfld_loss.shape)

                # eikonal loss
                grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean(dim=-1)
                grad_loss = reduce_mean_masked_instance(grad_loss, mask_gt).mean()

                # normals loss --> SALD
                norm_sub = (mnfld_grad - sk_normals).norm(2, dim=-1)
                norm_add = (mnfld_grad + sk_normals).norm(2, dim=-1)

                values = torch.cat((norm_sub.unsqueeze(-1), norm_add.unsqueeze(-1)), dim=-1)
                normals_loss = torch.min(values, dim=-1)[0]
                normals_loss = normals_loss.mean(dim=-1)
                normals_loss = reduce_mean_masked_instance(normals_loss, mask_gt).mean()			

                im_loss = mnfld_loss + 0.1 * grad_loss
                im_loss = im_loss + 1.0 * normals_loss

            else:
                mnfld_loss = torch.zeros(1).to(pcs.device).mean()
                grad_loss = torch.zeros(1).to(pcs.device).mean()
                normals_loss = torch.zeros(1).to(pcs.device).mean()
                im_loss = torch.zeros(1).to(pcs.device).mean()

            ##L2 loss or angle loss for the two latent codes
            latent_codes = latent_codes.reshape(batch_size, K, -1)
            latent_codes_gt = latent_codes_gt.reshape(batch_size, K, -1)
            # print(latent_codes.shape)
            # print(latent_codes_gt.shape)

            if IS_L2:
                latent_loss = torch.square(latent_codes - latent_codes_gt).sum(dim=-1)
                latent_loss = reduce_mean_masked_instance(latent_loss, mask_gt).mean()

            else:
                ## Angle
                dot_abs = torch.sum(latent_codes * latent_codes_gt, dim=-1) # BxN
                dot_abs = 1.0 - dot_abs
                latent_loss = reduce_mean_masked_instance(dot_abs, mask_gt).mean()

            im_loss += latent_loss
            log = "Epoch: {} | Batch [{:04d}/{:04d}] | total loss: {:.4f} | latent loss: {:.4f} | manifold loss: {:.4f} | eikonal loss: {:.4f} | normal loss: {:.4f}"
            log = log.format(str(epoch)+'/'+str(NUM_EPOCHS), i, len(loader), im_loss.item(), latent_loss.item(), \
                mnfld_loss.item(), grad_loss.item(), normals_loss.item())
            log_string(log)

            scalars["IM_total_loss"].append(im_loss)
            scalars["IM_latent_loss"].append(latent_loss)
            scalars["IM_manifold_loss"].append(mnfld_loss)
            scalars["IM_eikonal_loss"].append(grad_loss)
            scalars["IM_normal_loss"].append(normals_loss)

            ### For tensorboard
            writer.add_scalar("Loss/IM_total_loss", im_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/IM_latent_loss", latent_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/IM_manifold_loss", mnfld_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/IM_eikonal_loss", grad_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/IM_normal_loss", normals_loss, epoch*len(loader)+i)
            ###########

            if IS_PC_TRAIN :
                total_loss += im_loss
            else:
                total_loss = im_loss

            optimizer.zero_grad()

            # Updating the BN decay
            bn_momentum = get_batch_norm_decay(global_step, batch_size, BN_DECAY_STEP, staircase=True)
            if old_bn_momentum != bn_momentum:
                update_momentum(model, bn_momentum)
                old_bn_momentum = bn_momentum
            # Updating the LR decay
            learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, DECAY_STEP, DECAY_RATE, staircase=True)
            if old_learning_rate != learning_rate:
                optimizer.param_groups[0]['lr'] = learning_rate
                old_learning_rate = learning_rate

            total_loss.backward()

            optimizer.step()
            global_step += 1

            now = datetime.datetime.now()

            if IS_PC_TRAIN:
                log = "Epoch: {} | Batch [{:04d}/{:04d}] | total loss: {:.4f} | normal loss: {:.4f} | mIOU loss: {:.4f} | ext loss: {:.4f} | bb loss: {:.4f} | center loss: {:.4f}"
                log = log.format(str(epoch)+'/'+str(NUM_EPOCHS), i, len(loader), total_loss.item(), total_normal_loss.item(), \
                    total_miou_loss.item(), total_extrusion_loss.item(), total_bb_loss.item(), total_center_loss.item())
                log_string(log)
                log_string("")

                scalars["normal_loss"].append(total_normal_loss)
                scalars["mIOU_loss"].append(total_miou_loss)
                scalars["ext_loss"].append(total_extrusion_loss)
                scalars["bb_loss"].append(total_bb_loss)
                scalars["center"].append(total_center_loss)

                ### For tensorboard
                writer.add_scalar("Loss/normal", total_normal_loss, epoch*len(loader)+i)
                writer.add_scalar("Loss/segmentation_mIOU", total_miou_loss, epoch*len(loader)+i)
                writer.add_scalar("Loss/bb_CE", total_bb_loss, epoch*len(loader)+i)
                writer.add_scalar("Loss/ext_angle", total_extrusion_loss, epoch*len(loader)+i)
                writer.add_scalar("Loss/center", total_center_loss, epoch*len(loader)+i)

            scalars["total_loss"].append(total_loss)
            writer.add_scalar("Loss/total", total_loss, epoch*len(loader)+i)

        writer.flush()

        if ((epoch) %10 == 0):
            # Summary after each epoch
            summary = {}
            now = datetime.datetime.now()
            duration = (now - start).total_seconds()
            log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
            log = log.format(now.strftime("%c"), epoch, NUM_EPOCHS, duration)
            for m, v in scalars.items():
                summary[m] = torch.stack(v).mean()
                log += " {}: {:.4f} |".format(m, summary[m].item())

            fname = os.path.join(LOG_DIR, "checkpoint_{:04d}.pth".format(epoch))
            print("> Saving model to {}...".format(fname))
            model_to_save = {"model": model.state_dict(), "implicit_net": implicit_net.state_dict(), "pn_encoder": pn_encoder.state_dict()}
            torch.save(model_to_save, fname)

            if epoch >20 and summary["total_loss"] < best_loss:
                best_loss = summary["total_loss"]
                fname = os.path.join(LOG_DIR, "best_model.pth")
                print("> Saving model to {}...".format(fname))
                model_to_save = {"model": model.state_dict(), "implicit_net": implicit_net.state_dict(), "pn_encoder": pn_encoder.state_dict()}
                torch.save(model_to_save, fname)

                log += " best: {:.4f} |".format(best_loss)

                fname = os.path.join(LOG_DIR, "train.log")
                with open(fname, "a") as fp:
                    fp.write(log + "\n")

                log_string(log)
                print("--------------------------------------------------------------------------")
			
            fname = os.path.join(LOG_DIR, "model.pth")
            print("> Saving model to {}...".format(fname))
            model_to_save = {"model": model.state_dict(), "implicit_net": implicit_net.state_dict(), "pn_encoder": pn_encoder.state_dict()}
            torch.save(model_to_save, fname)				


if __name__ == '__main__':
    main()
