ulimit -n 4096
set -x


#################### IEEE TETCI submission visualization run #######################

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



