"""
Pixel-level alignment between student and teacher
"""

import torch
from torch import nn
import torch.nn.functional as F

class TeacherStudentAlignHead(nn.Module):
    def __init__(self, cfg, student_dim, teacher_dim, normalize_feature=True):
        super(TeacherStudentAlignHead, self).__init__()
        head_type = cfg.SEMISUPNET.ALIGN_HEAD_TYPE
        self.proj_dim = cfg.SEMISUPNET.ALIGN_HEAD_PROJ_DIM
        self.normalize_feature = normalize_feature
        if cfg.SEMISUPNET.ALIGN_PROJ_GELU:
            nl_layer = nn.GELU()
        else:
            nl_layer = nn.ReLU()
        if head_type=='attention':
            self.projection_layer = MHALayer(student_dim, teacher_dim)
        elif head_type=='MLP':
            self.projection_layer = nn.Sequential(nn.Conv2d(student_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, teacher_dim, 1, 1))
        elif head_type=='MLP3':
            self.projection_layer = nn.Sequential(nn.Conv2d(student_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, teacher_dim, 1, 1))
        elif head_type=='linear':
            self.projection_layer = nn.Conv2d(student_dim, teacher_dim, 1, 1)
        else:
            raise NotImplementedError("{} align head not supported.".format(head_type))
        # print('head type', head_type, self.projection_layer)

    def forward(self, feat_cnn, teacher_feat_shape):
        return self.project_student_feat(feat_cnn, teacher_feat_shape)
    
    def project_student_feat(self, feat_cnn, teacher_feat_shape):
        h, w = teacher_feat_shape
        feat_cnn = self.projection_layer(feat_cnn)
        feat_cnn = F.interpolate(feat_cnn, (h,w), mode='bilinear')
        if self.normalize_feature:
            feat_cnn = F.normalize(feat_cnn, p=2, dim=1)
        return feat_cnn
            
    def align_loss(self, feat_student, feat_teacher, return_sim=False):
        if self.normalize_feature:
            feat_student = feat_student.permute((0,2,3,1))
            feat_teacher = feat_teacher.permute((0,2,3,1))
            sim = torch.matmul(feat_student.unsqueeze(-2), feat_teacher.unsqueeze(-1))
            loss = (1-sim).mean()

        else:
            sim = torch.linalg.norm(feat_student-feat_teacher, dim=1, ord=2)
            loss = sim.mean() / 100

        if return_sim:
            return loss, sim
        else:
            return loss
        
class MHALayer(nn.Module):
    def __init__(self, cnn_dim, dino_dim):
        super(MHALayer, self).__init__()

        self.attn_layer = nn.MultiheadAttention(cnn_dim, num_heads=4, batch_first=True)
        self.projection = nn.Conv2d(cnn_dim, dino_dim, 1, 1)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b,c,h*w).transpose(1,2)
        x, _ = self.attn_layer(x, x, x, need_weights=False)
        x = self.projection(x.transpose(1,2).reshape(b,c,h,w))
        return x 