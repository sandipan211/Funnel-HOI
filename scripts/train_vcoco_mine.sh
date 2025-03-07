ulimit -n 4096
set -x
EXP_DIR=../output_dir_no_wtsharing/Supervised_vcoco_fo_fv_clip100percent_myloss

python ../main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file vcoco \
    --hoi_path ../../data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --backbone resnet50 \
    --num_queries 64 \
    --dec_layers 3 \
    --epochs 90 \
    --use_nms_filter \
    --fix_clip \
    --batch_size 8 \
    --pretrained ../params/detr-r50-pre-2branch-vcoco.pth \
    --with_clip_label \
    --with_obj_clip_label \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --opt_sched "multiStep" \
    --dataset_root GEN \
    --model_name HOICLIP \
    --zero_shot_type default \
    --resume ${EXP_DIR}/checkpoint_last.pth \
    --verb_pth ../tmp/vcoco_verb.pth \
    --verb_weight 0.1 \
    --lr_drop_list 30 60 \
    --conceptnet_path ../conceptNet/vcoco_similarityMatrix_verb_ing.pt \
    --bothFoFv  \
    --proposed_loss \
    --rel_alpha 1


