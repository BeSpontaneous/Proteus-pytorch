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
import models_synclr



class MetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        import_student = getattr(models_synclr, cfg.target_model)
        student = import_student(patch_size=cfg.patch_size, num_classes=0, mask_style='ibot')
        
        embed_dim = student.embed_dim
        
        import_teacher = getattr(models_synclr, cfg.teacher_model)
        teacher_backbone = import_teacher(patch_size=cfg.patch_size, teacher_path=cfg.teacher_path, num_classes=0, mask_style='ibot')
        teacher_backbone.eval()

        student_model_dict['backbone'] = student
        teacher_model_dict['backbone'] = teacher_backbone
        
        self.embed_dim = embed_dim

        # initialize parameters and checks
        self.total_n_global_crops = cfg.batch_size

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        teacher_embed_dim = teacher_backbone.embed_dim
        self.patch_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))
        
        self.token_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))

        self.fea_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))

        self.soft_criterion = torch.nn.MSELoss()

        for param in self.teacher.backbone.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        global_crops = inputs["collated_global_crops"]
        
        masks = inputs["collated_masks"]
        mask_indices_list = inputs["mask_indices_list"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = inputs["upperbound"]

        n_global_crops = 1

        # compute teacher output
        # @torch.no_grad()
        def compute_teacher_output():
            with torch.no_grad():
                teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = teacher_patch_tokens.shape[-1]

            # mask teacher patch tokens
            buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound, _dim)
            torch.index_select(
                teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            teacher_patch_tokens_masked = buffer_tensor_teacher[:n_masked_patches]

            return teacher_cls_tokens, teacher_patch_tokens, teacher_patch_tokens_masked

        # get the teacher outputs
        (
            teacher_cls_tokens,
            teacher_patch_tokens,
            teacher_patch_tokens_masked
        ) = compute_teacher_output()
        
        cur_masks = masks if self.cfg.mask_probability > 0 else None

        student_backbone_output_dict, student_backbone_output_dict_unmask = self.student.backbone(
            [global_crops, global_crops], masks=[cur_masks, None], is_training=True
        )

        student_cls_token_unmask = student_backbone_output_dict_unmask["x_norm_clstoken"]
        student_patch_tokens_unmask = student_backbone_output_dict_unmask["x_norm_patchtokens"]
        student_patch_tokens = student_backbone_output_dict["x_norm_patchtokens"]

        # mask student patch tokens
        _dim = student_patch_tokens.shape[-1]
        
        buffer_tensor_student = student_patch_tokens.new_zeros(upperbound, _dim)
        buffer_tensor_student[:n_masked_patches].copy_(
            torch.index_select(student_patch_tokens.flatten(0, 1),
                                dim=0,
                                index=mask_indices_list)
        )

        ## projection head
        student_patch_tokens_unmask = self.fea_head(student_patch_tokens_unmask)
        
        student_cls_token_unmask = self.token_head(student_cls_token_unmask)
        
        tokens_after_head = self.patch_head(buffer_tensor_student)
        student_patch_tokens_masked = tokens_after_head[:n_masked_patches]

        ## token objective
        distillation_loss_token = self.soft_criterion(student_cls_token_unmask, teacher_cls_tokens)

        ## fea objective
        student_whole_fea = torch.cat((student_cls_token_unmask.unsqueeze(1),student_patch_tokens_unmask),dim=1)
        teacher_whole_fea = torch.cat((teacher_cls_tokens.unsqueeze(1),teacher_patch_tokens),dim=1)
        distillation_loss_fea = self.soft_criterion(student_whole_fea, teacher_whole_fea)

        ## patch objective
        patch_loss = self.soft_criterion(student_patch_tokens_masked, teacher_patch_tokens_masked)
        
        # coefficient
        token_loss = self.cfg.lambda_token * distillation_loss_token
        fea_loss = self.cfg.lambda_fea * distillation_loss_fea
        patch_loss = self.cfg.lambda_patch * patch_loss

        # compute the total loss
        total_loss = patch_loss + fea_loss + token_loss

        # return the final loss dict
        loss_dict = {"patch_loss": patch_loss, "fea_loss": fea_loss, "token_loss": token_loss, "loss": total_loss}
        
        return loss_dict