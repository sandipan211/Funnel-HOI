ulimit -n 4096
set -x
# EXP_DIR=../exps/hico/hoiclip  # agneys
# removed # --fs_num -1 \
# --resume ${EXP_DIR}/checkpoint_last.pth \


# for latest verbing+myloss+fo+fv+clip100percent
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/RF_UC_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  

# for latest nfuc verbing+myloss+fo+fv+clip100percent
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/NF_UC_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type non_rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  

# for verbing+myloss
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/RF_UC_fo_fv_myloss_lr_1e_minus4/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --proposed_loss 

# for verbing + no loss
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/RF_UC_noCrossAttn_add_no768proj_verbing/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt 

# for nfuc verbing + fo,fv + clip 100 percent + no loss
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/NF_UC_fo_fv_clip100percent/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type non_rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --rel_alpha 1  

# for only fo + no loss
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/RF_UC_onlyFo_noLoss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt 

# for nfuc only fo + no loss
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/NF_UC_fo_clip100percent/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type non_rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --onlyFo  \
#             --rel_alpha 1  


# for original HOICLIP
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../checkpoint_rf_uc.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt 

# for nfuc original HOICLIP
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../checkpoint_nrf_uc.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type non_rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt 

#################### IEEE TETCI submission visualization run #######################

# EXP_DIR=../exps/hico/hoiclip/funnel-hoi/UO
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/UO_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type unseen_object \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  \
#             --viz_only

# EXP_DIR=../exps/hico/hoiclip/funnel-hoi/UV
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/UV_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type unseen_verb \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  \
#             --viz_only


# EXP_DIR=../exps/hico/hoiclip/funnel-hoi/UA
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/UA_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type unseen_action \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  \
#             --viz_only

# EXP_DIR=../exps/hico/hoiclip/funnel-hoi/NF-UC
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/NF_UC_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type non_rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  \
#             --viz_only

# EXP_DIR=../exps/hico/hoiclip/funnel-hoi/RF-UC
# python ../main.py \
#             --output_dir ${EXP_DIR} \
#             --dataset_file hico \
#             --hoi_path ../../data/hico_20160224_det \
#             --num_obj_classes 80 \
#             --num_verb_classes 117 \
#             --backbone resnet50 \
#             --num_queries 64 \
#             --dec_layers 3 \
#             --epochs 120 \
#             --lr_drop 60 \
#             --use_nms_filter \
#             --fix_clip \
#             --batch_size 8 \
#             --pretrained ../output_dir_no_wtsharing/RF_UC_fo_fv_clip100percent_myloss/checkpoint_best.pth \
#             --with_clip_label \
#             --with_obj_clip_label \
#             --gradient_accumulation_steps 1 \
#             --num_workers 8 \
#             --opt_sched "multiStep" \
#             --dataset_root GEN \
#             --model_name VISUALIZATION \
#             --zero_shot_type rare_first \
#             --del_unseen \
#             --verb_pth ../tmp/verb.pth \
#             --eval \
#             --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
#             --bothFoFv  \
#             --proposed_loss \
#             --rel_alpha 1  \
#             --viz_only

EXP_DIR=../exps/hico/hoiclip/funnel-hoi/Supervised
python ../main.py \
            --output_dir ${EXP_DIR} \
            --dataset_file hico \
            --hoi_path ../../data/hico_20160224_det \
            --num_obj_classes 80 \
            --num_verb_classes 117 \
            --backbone resnet50 \
            --num_queries 64 \
            --dec_layers 3 \
            --epochs 120 \
            --lr_drop 60 \
            --use_nms_filter \
            --fix_clip \
            --batch_size 8 \
            --pretrained ../output_dir_no_wtsharing/Supervised_fo_fv_clip100percent_myloss/checkpoint_best.pth \
            --with_clip_label \
            --with_obj_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 8 \
            --opt_sched "multiStep" \
            --dataset_root GEN \
            --model_name VISUALIZATION \
            --zero_shot_type default \
            --del_unseen \
            --verb_pth ../tmp/verb.pth \
            --eval \
            --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
            --bothFoFv  \
            --proposed_loss \
            --rel_alpha 1  \
            --viz_only



