#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch

from ..bbox_utils import decode, nms
from torch.autograd import Function


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data,  cls_conf_data,lr_conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        cls_preds = cls_conf_data.view(num, num_priors, 2).transpose(2, 1)
        lr_preds = lr_conf_data.view(num,num_priors,3)[:,:,1:]

        batch_priors = prior_data.view(-1, num_priors,4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, 2, self.top_k, 7)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            cls_scores = cls_preds[i].clone()
            lr_scores = lr_preds[i].clone()

            for cl in range(1, 2):
                c_mask = cls_scores[cl].gt(self.conf_thresh)
                scores = cls_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)

                lr_mask = c_mask.unsqueeze(1).expand_as(lr_scores)
                lr_score = lr_scores[lr_mask].view(-1,2)
                if boxes_.numel() == 0:
                    continue 
                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),boxes_[ids[:count]],lr_score[ids[:count]]), 1)
        return output
