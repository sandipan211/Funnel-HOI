ulimit -n 4096
set -x
EXP_DIR=../output_dir_no_wtsharing/RF_UC_fo_fv_clip100percent_myloss
python  ../main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file hico \
    --hoi_path ../../data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --num_queries 64 \
    --dec_layers 3 \
    --epochs 90 \
    --lr_drop_list 30 60 \
    --use_nms_filter \
    --fix_clip \
    --batch_size 8 \
    --pretrained ../params/detr-r50-pre-2branch-hico.pth \
    --with_clip_label \
    --with_obj_clip_label \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --opt_sched "multiStep" \
    --dataset_root GEN \
    --model_name HOICLIP \
    --del_unseen \
    --zero_shot_type rare_first \
    --resume ${EXP_DIR}/checkpoint_last.pth \
    --verb_pth ../tmp/verb.pth \
    --conceptnet_path ../conceptNet/similarityMatrix_verb_ing.pt \
    --bothFoFv  \
    --proposed_loss \
    --rel_alpha 1  
