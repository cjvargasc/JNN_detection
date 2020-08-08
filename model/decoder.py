""" https://github.com/tztztztztz/yolov2.pytorch """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from utils.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, xxyy2xywh, box_ious
from config import Config as cfg


def filter_boxes(boxes_pred, conf_pred, confidence_threshold=0.6):
    """
    Filter boxes whose confidence is lower than a given threshold

    Arguments:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4) (x1, y1, x2, y2) predicted boxes
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    threshold -- float, threshold used to filter boxes

    Returns:
    filtered_boxes -- tensor of shape (num_positive, 4)
    filtered_conf -- tensor of shape (num_positive, 1)
    """
    pos_inds = (conf_pred > confidence_threshold).view(-1)

    filtered_boxes = boxes_pred[pos_inds, :]

    filtered_conf = conf_pred[pos_inds, :]

    return filtered_boxes, filtered_conf


def nms(boxes, scores, threshold):
    """
    Apply Non-Maximum-Suppression on boxes according to their scores

    Arguments:
    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    scores -- tensor of shape (N) confidence
    threshold -- float. NMS threshold

    Returns:
    keep -- tensor of shape (None), index of boxes which should be retain.
    """

    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:

        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)

        ious = box_ious(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def generate_prediction_boxes(deltas_pred):
    """
    Apply deltas prediction to pre-defined anchors

    Arguments:
    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)

    Returns:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4)  (x1, y1, x2, y2)
    """

    H = int(cfg.im_w / cfg.strides)
    W = int(cfg.im_h / cfg.strides)

    anchors = torch.FloatTensor(cfg.anchors)
    all_anchors_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)

    all_anchors_xywh = deltas_pred.new(*all_anchors_xywh.size()).copy_(all_anchors_xywh)

    boxes_pred = box_transform_inv(all_anchors_xywh, deltas_pred)

    return boxes_pred


def scale_boxes(boxes, im_info):
    """
    scale predicted boxes

    Arguments:
    boxes -- tensor of shape (N, 4) xxyy format
    im_info -- dictionary {width:, height:}

    Returns:
    scaled_boxes -- tensor of shape (N, 4) xxyy format

    """

    h = im_info['height']
    w = im_info['width']

    input_h, input_w = cfg.im_h, cfg.im_h
    scale_h, scale_w = input_h / float(h), input_w / float(w)

    # scale the boxes
    boxes *= cfg.strides

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h

    boxes = xywh2xxyy(boxes)

    # clamp boxes
    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes


def decode(model_output, im_info, conf_threshold=0.6, nms_threshold=0.4):
    """
    Evaluates the model output and generates the final predicted boxes

    Arguments:
    model_output -- list of tensors (deltas_pred, conf_pred, classes_pred)

    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)

    im_info -- dictionary {w:, h:}

    threshold -- float, threshold used to filter boxes


    Returns:
    detections -- tensor of shape (None, 7) (x1, y1, x2, y2, conf)
    """

    deltas = model_output[0].cpu()
    conf = model_output[1].cpu()

    # apply deltas to anchors
    boxes = generate_prediction_boxes(deltas)

    # filter boxes on confidence score
    boxes, conf = filter_boxes(boxes, conf, conf_threshold)

    # no detection !
    if boxes.size(0) == 0:
        return []

    # scale boxes
    boxes = scale_boxes(boxes, im_info)

    # apply nms
    keep = nms(boxes, conf.view(-1), nms_threshold)
    boxes_keep = boxes[keep, :]
    conf_keep = conf[keep, :]

    #
    seq = [boxes_keep, conf_keep]

    return torch.cat(seq, dim=1)
    """ 
    detections = []

    # apply NMS classwise
    for cls in range(num_classes):
        cls_mask = cls_max_id == cls
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_pred_class = boxes[inds, :].view(-1, 4)
        conf_pred_class = conf[inds, :].view(-1, 1)
        cls_max_conf_class = cls_max_conf[inds].view(-1, 1)

        nms_keep = yolo_nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep, cls_max_conf_class_keep]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)
    """
