# Mikaela Uy (mikacuy@cs.stanford.edu)
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class backbone(nn.Module):
    def __init__(self, normal_channel=False, output_sizes = [3]):
        super(backbone, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.dim_pos = 3

        ## Dimension of position (x,y,z)
        dim_pos = self.dim_pos

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=dim_pos+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + dim_pos, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + dim_pos, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+additional_channel, mlp=[128, 128, 128])

        # FC stage
        self.fc1 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.ModuleList()
        for output_size in output_sizes:
            self.fc2.append(torch.nn.Conv1d(128, output_size, 1))


    def forward(self, x):
        x = x.transpose(2,1)
        batch_size = x.shape[0]

        input_pos = x[:,:self.dim_pos,:]
        if x.shape[1] > self.dim_pos:
            input_feats = x[:,self.dim_pos:,:]
        else:
            input_feats = None

        # Set Abstraction layers
        l1_xyz, l1_feats = self.sa1(input_pos, input_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)

        # Feature Propagation layers
        l4_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)
        l5_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l4_feats)
        l6_feats = self.fp1(input_pos, l1_xyz, input_feats, l5_feats)

        # FC stage
        output_feat = self.fc1(l6_feats)
        output_feat = F.relu(self.bn1(output_feat))
        output_feat = F.dropout(output_feat, p=0.5)
        results = []
        for fc2_layer in self.fc2:
            result = fc2_layer(output_feat)
            result = result.transpose(1,2)
            results.append(result)
        return results
