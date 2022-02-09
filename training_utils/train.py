import sys

import training_utils.utils as utils
import numpy as np
import numpy as np
import torch
import torch.utils.data
import math

from postprocessing import get_NonMaxSup_boxes, intersection_over_union


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, img_name in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def eval_intrain(model, data_loader, device):
    model.eval()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i, data in enumerate(data_loader):

        images, targets, img_name = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        TP_temp = 0
        FP_temp = 0
        TN_temp = 0
        FN_temp = 0

        with torch.no_grad():
            prediction = model(images)
            boxes = targets[0]['boxes'].cpu().numpy()

            predicted_boxes = get_NonMaxSup_boxes(prediction[0])['boxes']
            predicted_scores = get_NonMaxSup_boxes(prediction[0])['scores']

            if boxes.shape[0] > 0:

                for j in range(len(predicted_boxes)):
                    np_predicted_boxes = predicted_boxes[j].cpu().numpy()
                    np_predicted_scores = predicted_scores[j].cpu().numpy()

                    truth = 0
                    predicted_result = 0

                    for k in range(boxes.shape[0]):
                        if intersection_over_union(np_predicted_boxes, boxes[k, :]) > 0.2:
                            truth = truth + 1

                        if np_predicted_scores > 0.25:
                            predicted_result = predicted_result + 1

                    if truth > 0 and predicted_result > 0:
                        TP_temp = TP_temp + 1
                    if truth > 0 and predicted_result == 0:
                        FN_temp = FN_temp + 1
                    if truth == 0 and predicted_result > 0:
                        FP_temp = FP_temp + 1
                    if truth == 0 and predicted_result == 0:
                        TN_temp = TN_temp + 1

                if len(predicted_boxes)==0:
                    length = 1
                else:
                    length = len(predicted_boxes)

                TP = TP + TP_temp / length
                FP = FP + FP_temp / length
                TN = TN + TN_temp / length
                FN = FN + FN_temp / length


            else:

                for j in range(len(predicted_boxes)):

                    np_predicted_scores = predicted_scores[j].cpu().numpy()
                    predicted_result = 0

                    if np_predicted_scores > 0.25:
                        FP_temp = FP_temp + 1
                    else:
                        TN_temp = TN_temp + 1

                if len(predicted_boxes)==0:
                    length = 1
                else:
                    length = len(predicted_boxes)

                FP = FP + FP_temp / length
                TN = TN + TN_temp / length

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    print('Sensitivity: {:.6f}, Specificity {:.6f}'.format(Sensitivity, Specificity))
