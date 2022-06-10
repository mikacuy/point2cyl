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


# from global_variables import *
from utils import * 
from data_utils import *
from dataloader import AutodeskDataset_h5
from losses import *

### For tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pointnet_extrusion', help='model name')
parser.add_argument('--num_point', type=int,  default=8192, help='Point Number [default: 8192]')
parser.add_argument('--K', type=int,  default=8, help='Max number of extrusions')
parser.add_argument('--batch_size', type=int,  default=4, help='batch size')

parser.add_argument("--logdir", default="Point2Cyl_without_sketch", help="path to the log directory", type=str)
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

FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

DATA_SPLIT = FLAGS.data_split
DATA_DIR = FLAGS.data_dir

H5_FILENAME = os.path.join(DATA_DIR, DATA_SPLIT + ".h5")

NUM_POINT = FLAGS.num_point
MODEL = FLAGS.model
K = FLAGS.K
BATCH_SIZE = FLAGS.batch_size

PRED_SEG = FLAGS.pred_seg
PRED_NORMAL = FLAGS.pred_normal
PRED_BB = FLAGS.pred_bb
PRED_EXT = FLAGS.pred_extrusion
PRED_CENTER = FLAGS.pred_center

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
    dataset = AutodeskDataset_h5(H5_FILENAME, NUM_POINT, K, op=False, center=True, extent=False)
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
    model.to(device)

    # Optimizer
    init_learning_rate = LEARNING_RATE

    ### Switch optimizer, adagrad
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    global_step = 0
    old_learning_rate = init_learning_rate
    old_bn_momentum = MOMENTUM

    model.train()
    best_loss = np.Inf

    for epoch in range(1, NUM_EPOCHS+1):
        start = datetime.datetime.now()
        scalars = defaultdict(list)
        for i, batch in enumerate(loader):
            sampled_pcs, sampled_normals, sampled_extrusion_labels, sampled_bb_labels, per_point_extrusion_axes, \
                per_point_extrusion_distances, extrusion_axes, extrusion_distances, extrusion_centers = batch
								
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

            gt_extrusion_centers = [ax.to(device, dtype=torch.float) for ax in extrusion_centers]
            gt_extrusion_centers = torch.stack(gt_extrusion_centers)

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
			
            mask_gt = get_mask_gt(gt_extrusion_instances, K)
			
            ###### Calculate extrusion axis loss using joint base/barrel formulation
            if (PRED_NORMAL and PRED_BB and PRED_EXT):
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

            ###### Calculate center loss ######
            if PRED_CENTER:
                W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(cur_batch_size, NUM_POINT, K))
                predicted_centroids = estimate_extrusion_centers(W_reordered, pcs)

                centroid_diff = torch.square(predicted_centroids - gt_extrusion_centers).sum(dim=-1)
                avg_center_loss = reduce_mean_masked_instance(centroid_diff, mask_gt)

            else:
                avg_center_loss = torch.zeros([batch_size]).to(gt_extrusion_axes.device)

            total_center_loss = torch.mean(avg_center_loss) * center_loss_multiplier
            total_loss += total_center_loss				

            optimizer.zero_grad()
            # Updating the BN decay
            bn_momentum = get_batch_norm_decay(global_step, batch_size, BN_DECAY_STEP, staircase=True)
            if old_bn_momentum != bn_momentum:
                update_momentum(model, bn_momentum)
                old_bn_momentum = bn_momentum
            # Updating the LR decay
            learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, DECAY_STEP, DECAY_RATE, staircase=True)
            if old_learning_rate != learning_rate:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                old_learning_rate = learning_rate

            total_loss.backward()
            optimizer.step()
            global_step += 1

            now = datetime.datetime.now()
            log = "Epoch: {} | Batch [{:04d}/{:04d}] | total loss: {:.4f} | normal loss: {:.4f} | mIOU loss: {:.4f} | bb loss: {:.4f} | ext loss: {:.4f} | center loss: {:.4f}"
            log = log.format(str(epoch)+'/'+str(NUM_EPOCHS), i, len(loader), total_loss.item(), total_normal_loss.item(), \
                total_miou_loss.item(), total_bb_loss.item(), total_extrusion_loss.item(), total_center_loss.item())
            log_string(log)

            scalars["total_loss"].append(total_loss)
            scalars["normal_loss"].append(total_normal_loss)
            scalars["mIOU_loss"].append(total_miou_loss)
            scalars["ext_loss"].append(total_extrusion_loss)
            scalars["bb_loss"].append(total_bb_loss)
            scalars["center"].append(total_center_loss)

            ### For tensorboard
            writer.add_scalar("Loss/total", total_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/normal", total_normal_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/segmentation_mIOU", total_miou_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/bb_CE", total_bb_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/ext_angle", total_extrusion_loss, epoch*len(loader)+i)
            writer.add_scalar("Loss/center", total_center_loss, epoch*len(loader)+i)

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
            model_to_save = {"model": model.state_dict()}
            torch.save(model_to_save, fname)

            if epoch >20 and summary["total_loss"] < best_loss:
                best_loss = summary["total_loss"]
                fname = os.path.join(LOG_DIR, "best_model.pth")
                print("> Saving model to {}...".format(fname))
                model_to_save = {"model": model.state_dict()}
                torch.save(model_to_save, fname)

                log += " best: {:.4f} |".format(best_loss)

                fname = os.path.join(LOG_DIR, "train.log")
                with open(fname, "a") as fp:
                   fp.write(log + "\n")

                log_string(log)
                print("--------------------------------------------------------------------------")
			
            fname = os.path.join(LOG_DIR, "model.pth")
            print("> Saving model to {}...".format(fname))
            model_to_save = {"model": model.state_dict()}
            torch.save(model_to_save, fname)				


if __name__ == '__main__':
    main()
