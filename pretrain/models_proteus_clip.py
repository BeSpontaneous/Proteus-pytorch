# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed.nn
import torch.distributed as dist
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import models_clip



class MetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        import_student = getattr(models_clip, cfg.target_model)
        student = import_student()
        
        embed_dim = student.embed_dim
        
        import_teacher = getattr(models_clip, cfg.teacher_model)
        teacher_backbone = import_teacher(teacher_path=cfg.teacher_path)
        teacher_backbone.eval()

        student_model_dict['backbone'] = student
        teacher_model_dict['backbone'] = teacher_backbone
        
        self.embed_dim = embed_dim

        # initialize parameters and checks
        self.total_n_global_crops = cfg.batch_size

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        teacher_embed_dim = teacher_backbone.embed_dim
        
        self.token_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))

        self.soft_criterion = torch.nn.MSELoss()

        for param in self.teacher.backbone.parameters():
            param.requires_grad = False

    ## we explicitly remove the patch and feature learning objectives for CLIP training following the original design
    def forward(self, inputs):
        global_crops = inputs["collated_global_crops"]

        # compute teacher output
        # @torch.no_grad()
        def compute_teacher_output():
            with torch.no_grad():
                teacher_backbone_output_dict = self.teacher.backbone(global_crops)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]

            return teacher_cls_tokens

        # get the teacher outputs
        teacher_cls_tokens = compute_teacher_output()

        student_backbone_output_dict_unmask = self.student.backbone(global_crops)

        student_cls_token_unmask = student_backbone_output_dict_unmask["x_norm_clstoken"]

        ## projection head
        student_cls_token_unmask = self.token_head(student_cls_token_unmask)

        ## token objective
        distillation_loss_token = self.soft_criterion(student_cls_token_unmask, teacher_cls_tokens)
        
        # coefficient
        token_loss = self.cfg.lambda_token * distillation_loss_token

        # compute the total loss
        total_loss = token_loss

        # return the final loss dict
        loss_dict = {"patch_loss": torch.tensor(0.0), "fea_loss": torch.tensor(0.0), "token_loss": token_loss, "loss": total_loss}
        
        return loss_dict