import math
import os
import random
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.datasets_gen.hico_eval_triplet import HICOEvaluator as HICOEvaluator_gen
from datasets.datasets_gen.vcoco_eval import VCOCOEvaluator as VCOCOEvaluator_gen
import json
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import time
import gc

# change SS
from pathlib import Path

# visualization
# from gradcam import gradcam_output
# from pytorch_grad_cam import GradCAM
# from myVisualization import gradCAMAveragePooling,gradCAMfo,gradCAMSum
# from models.models_hoiclip.hoiclip import hwArray,adaptiveAverageActivations,sumActivations,foActivations,adaptiveAverageGradients,sumGradients,foGradients

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, lr_scheduler=None,
                    gradient_accumulation_steps=1, enable_amp=False, no_training=False, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels') and False:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if enable_amp:
        print('\nEnable half precision training\n')

    # scaler = GradScaler()
    # debug
    debug_count = 0
    step = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if no_training:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
            clip_img = torch.stack([v['clip_inputs'] for v in targets])
            # with autocast():
            obj_feature, hoi_feature, verb_feature = model(samples, clip_input=clip_img, targets=targets)

            metric_logger.update(loss=0)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=0)
            elif hasattr(criterion, 'loss_hoi_labels'):
                metric_logger.update(hoi_class_error=0)
            else:
                metric_logger.update(obj_class_error=0)
            metric_logger.update(lr=0)
            continue

        samples = samples.to(device)


        file_names = [{'filename': i['filename']} for i in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
        for t, f in zip(targets, file_names):
            t.update(f)
        clip_img = torch.stack([v['clip_inputs'] for v in targets])
        # print(clip_img[0])
        # print(targets[0]["filename"])
        # exit(0)
        # print("from engine.py")
        # print(targets[0]["clip_inputs"].shape)
        # print(f"clip_img shape: {clip_img.shape}")
        outputs = model(samples, clip_input=clip_img, targets=targets)

        # change SS
        # exit()
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.detach().item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        delay_unscale = (step + 1) % gradient_accumulation_steps != 0
        losses = losses / gradient_accumulation_steps
        if enable_amp:
            raise NotImplementedError
            # with amp.scale_loss(losses, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            #     scaled_loss.backward()
        else:
            losses.backward()
        # print(f"{adaptiveAverageActivations[0].shape} {adaptiveAverageGradients[0][0].shape}")
        # print(f"{sumActivations[0].shape} {sumGradients[0][0].shape}")
        # print(f"{foActivations[0].shape} {foGradients[0][0].shape}")
        # gradCAMAveragePooling(adaptiveAverageActivations[0],adaptiveAverageGradients[0][0],hwArray[0][0],hwArray[0][1])
        # gradCAMSum(sumActivations[0],sumGradients[0][0],hwArray[0][0],hwArray[0][1])
        # gradCAMfo(foActivations[0],foGradients[0][0],hwArray[0][0],hwArray[0][1])
        # exit(0)

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_norm > 0:
                if enable_amp:
                    pass
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler:
            lr_scheduler.iter_step()

        step += 1

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels') and False:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            if 'hoi_class_error' in loss_dict_reduced:
                metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
            else:
                metric_logger.update(hoi_class_error=-1)
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # to free memory
        samples=None
        targets=None
        clip_img=None
        del losses, outputs, loss_dict, loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled
        gc.collect()
        torch.cuda.empty_cache()
        # # outputs=None
    # gc.collect()
    # torch.cuda.empty_cache()
    

    # trick for generate verb
    if no_training:
        from datasets.static_hico import HOI_IDX_TO_ACT_IDX, HOI_IDX_TO_OBJ_IDX
        hoi_feature = hoi_feature / hoi_feature.norm(dim=1, keepdim=True)
        obj_feature = obj_feature / obj_feature.norm(dim=1, keepdim=True)

        y_verb = [HOI_IDX_TO_ACT_IDX[i] for i in range(600)]
        y_obj = [HOI_IDX_TO_OBJ_IDX[i] for i in range(600)]

        # composite image feature verb + text feature object
        obj_human = []
        for i in range(600):
            obj_human.append(obj_feature[y_obj[i]])
        obj_human = torch.stack(obj_human)
        verb_human = hoi_feature - obj_human

        verb_feature = torch.zeros(117, 512)
        for idx, v in zip(y_verb, verb_human):
            verb_feature[idx] += v

        for i in range(117):
            verb_feature[i] /= y_verb.count(i)

        v_feature = verb_feature / verb_feature.norm(dim=-1, keepdim=True)
        torch.save(v_feature, f'./verb_{args.dataset_file}.pth')
        exit()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    counter = 0
    
    # print(len(data_loader))
    # for tsne
    # selectedInteractionFeatures=[]
    # selectedLabels=[]
    # desiredInteractions={(111, 4),(10, 15),(111, 3),(111, 18),(111, 6),(12, 59),(96, 50),(111, 49),(15, 48),(0, 30)}

    # to calculate the accuracy of calculation of toplo,toplv
    # total_matched_o=0
    # total_matched_v=0
    # total_interactive_images=0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        
        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        
        # to calculate the accuracy of calculation of toplo,toplv
        # for target in targets:
        #     total_matched_o+=target["numMatches_o"]
        #     total_matched_v+=target["numMatches_v"]
        #     total_interactive_images+=target["numinteractive_images"]
        if(not args.viz_only): # change SS: for TETCI
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['hoi'](outputs, orig_target_sizes)

            # for tsne
            # interactionFeature = outputs['hoi_feature']
            # interactionFeature = torch.mean(interactionFeature,dim=1)
            # for i in range(len(targets)):
            #     target = targets[i]
            #     labels = target['labels']
            #     hois = target['hois']
            #     numMatches = 0
            #     prevMatch = None 
            #     for hoi in hois:
            #         currInteraction = (int(hoi[2]),int(labels[hoi[1]]))
            #         if(currInteraction in desiredInteractions):
            #             if(currInteraction != prevMatch):
            #                 numMatches+=1
            #                 prevMatch=currInteraction
            #     if(numMatches == 1):
            #         selectedInteractionFeatures.append(interactionFeature[i].unsqueeze(0))
            #         selectedLabels.append(prevMatch)
            
            
            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
            # For avoiding a runtime error, the copy is used
            gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        counter += 1
        if counter >= 20 and args.no_training:
            break
        
        # to free memory
        samples=None
        targets=None
        clip_img=None
        # outputs=None
    gc.collect()
    torch.cuda.empty_cache()
    
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if(args.viz_only): # change SS: for TETCI
        exit(0)

    # for tsne
    # interactionFeatures=torch.cat(selectedInteractionFeatures,dim=0)
    # print(f"interactionFeatures shape:{interactionFeatures.shape}")
    # torch.save(interactionFeatures,'../tSNEPlots/features_NF_UC_fo_fv_clip100percent_myloss.pt')
    # torch.save(selectedLabels,'../tSNEPlots/labels_NF_UC_fo_fv_clip100percent_myloss.pt')
    # exit(0)
    
    # to calculate the accuracy of calculation of toplo,toplv
    # print(f"topk objects accuracy: {(total_matched_o*100)/total_interactive_images},topk verbs accuracy: {(total_matched_v*100)/total_interactive_images}")
    # exit(0)
    
    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]


    """
    For zero-shot enhancement
    args.training_free_enhancement_path is the path to store performance for different hyper-parameter
    """
    root = os.path.join(args.output_dir, args.training_free_enhancement_path)
    # change SS: make the directory here also if not exists
    Path(root).mkdir(parents=True, exist_ok=True)
    if args.training_free_enhancement_path:

        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========The great hyperparameter tuning begins============\n'
            print(log)
            f.write(log)

        test_pred = copy.deepcopy(preds)

        # testing
        if dataset_file == 'hico':
            evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets,
                                          data_loader.dataset.correct_mat, args=args)
        else:
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
        stats = evaluator.evaluate()

        text_hoi_feature = model.transformer.hoi_cls
        spatial_feature = torch.cat([i['clip_visual'].unsqueeze(0) for i in preds])
        spatial_feature /= spatial_feature.norm(dim=-1, keepdim=True)
        spatial_cls = spatial_feature[:, 0, :]  # M, c
        
        #mine
        # print(spatial_cls.device)
        # print(spatial_cls.shape)
        # print(text_hoi_feature.device)
        # print(text_hoi_feature.shape)
        spatial_cls=spatial_cls.to(text_hoi_feature.device)
        # here we add prompt learning on text_hoi_feature
        cls_scores = spatial_cls @ text_hoi_feature.T
        # mine
        cls_scores = cls_scores.detach().cpu()
        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========Baseline Performance============\n{stats}\n============================\n'
            print(log)
            f.write(log)

        best_performance_1 = 0
        for a in [1]:
            for co in [1.0]:
                for topk in [10, 20, 30, 40, 50]:
                    print(f'current at topk: {topk} as: {a}')
                    test_pred = copy.deepcopy(preds)
                    clip_hoi_score = cls_scores
                    # clip_hoi_score /= (1 + alpha + beta)
                    clip_hoi_score_ori = clip_hoi_score.clone()

                    ignore_idx = clip_hoi_score.sort(descending=True).indices[:, topk:]
                    for idx, igx in enumerate(ignore_idx):
                        clip_hoi_score[idx][igx] *= 0
                    clip_hoi_score = clip_hoi_score.unsqueeze(1)

                    # update logits
                    print(clip_hoi_score.device)
                    for i in range(len(test_pred)):
                        test_pred[i]['hoi_scores'] += clip_hoi_score[i].sigmoid() * co
                    # testing
                    if dataset_file == 'hico':
                        evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                                      data_loader.dataset.non_rare_triplets,
                                                      data_loader.dataset.correct_mat, args=args)

                    else:
                        evaluator = VCOCOEvaluator_gen(test_pred, gts, data_loader.dataset.correct_mat,
                                                       use_nms_filter=args.use_nms_filter)
                    stats = evaluator.evaluate()
                    if dataset_file == 'hico':
                        re_map = stats['mAP']
                    elif dataset_file == 'vcoco':
                        re_map = stats['mAP_all']
                    elif dataset_file == 'hoia':
                        re_map = stats['mAP']
                    else:
                        raise NotImplementedError

                    if best_performance_1 < re_map:
                        best_performance_1 = re_map

                        with open(os.path.join(root, 'log.txt'), 'a') as f:
                            log = f'sigmoid after topk: {topk} as: {a} co: {co}' \
                                  f'\n performance: {stats}\n'
                            print(log)
                            f.write(log)

    if dataset_file == 'hico':
        if args.dataset_root == 'GEN':
            evaluator = HICOEvaluator_gen(preds, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat,
                                          args=args)
    elif dataset_file == 'vcoco':
        if args.dataset_root == 'GEN':
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
    else:
        raise NotImplementedError
    start_time = time.time()
    stats = evaluator.evaluate()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time computing mAP: {}'.format(total_time_str))

    return stats
