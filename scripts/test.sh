#!/bin/bash

# official testing for V-COCO in supervised setting (Funnel-HOI)
# EXP_DIR=../output_dir_no_wtsharing/Supervised_vcoco_fo_fv_clip100percent_myloss

# python ../generate_vcoco_official.py --pretrained ${EXP_DIR}/checkpoint_best.pth --save_path ${EXP_DIR}/vcoco.pickle --hoi_path ../../data/v-coco --dataset_file vcoco --num_queries 64 --use_nms_filter --with_clip_label --with_obj_clip_label --verb_pth ../tmp/vcoco_verb.pth --verb_weight 0.1 --conceptnet_path ../conceptNet/vcoco_similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1

# python ../../data/v-coco/vsrl_eval.py ${EXP_DIR}/vcoco.pickle ../../data/v-coco/



# for RF-UC
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/RF_UC_fo_clip100percent/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type rare_first   --del_unseen --training_free_enhancement_path ../training_free_testing/RFUC/RF_UC_fo_clip100percent     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --onlyFo --rel_alpha 1 

# for UO
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/UO_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type unseen_object   --del_unseen --training_free_enhancement_path ../training_free_testing/UO/UO_fo_fv_clip100percent_myloss     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1 

# for UV
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/UV_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type unseen_verb   --del_unseen --training_free_enhancement_path ../training_free_testing/UV/UV_fo_fv_clip100percent_myloss     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1

# for UA
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/UA_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type unseen_action   --del_unseen --training_free_enhancement_path ../training_free_testing/UA/UA_fo_fv_clip100percent_myloss     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1


# for NF-UC
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/NF_UC_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type non_rare_first   --del_unseen --training_free_enhancement_path ../training_free_testing/junkTests     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1


# for UC
# python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/UC4_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type uc4   --del_unseen --training_free_enhancement_path ../training_free_testing/UC/UC4_fo_fv_clip100percent_myloss     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1 

# for default (fully-supervised) setting
python ../main.py   --model_name HOICLIP      --pretrained ../output_dir_no_wtsharing/Supervised_fo_fv_clip100percent_myloss/checkpoint_best.pth      --dataset_file hico    --verb_pth ../tmp/verb.pth     --hoi_path ../../data/hico_20160224_det         --num_obj_classes 80         --num_verb_classes 117         --backbone resnet50         --num_queries 64         --dec_layers 3         --eval         --with_clip_label         --with_obj_clip_label         --use_nms_filter         --zero_shot_type default  --training_free_enhancement_path ../training_free_testing/Supervised/Supervised_fo_fv_clip100percent_myloss     --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt --bothFoFv --proposed_loss --rel_alpha 1  --json_file  ../training_free_testing/Supervised/Supervised_fo_fv_clip100percent_myloss/results.json     
