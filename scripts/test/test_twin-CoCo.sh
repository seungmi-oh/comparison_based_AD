mp='./../models'

# twin-CoCo
run=1 # top 1, bot 4, not 3

fn_list="-1"
nf_aug_ratio_list="0"
is_full_list="yes" # not: no, top,bot: yes
for is_full in $is_full_list
do
        for fn in $fn_list
        do
                # python3 main.py -run $run -nl 1 2 3 -al 1 2  --att_type CoCo -ds 6HN-top_6HN-topv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 2 --freeze_enc_epochs 8 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_fe_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn 
                # python3 main.py -run $run -nl 1 2 3 -al 1 2  --att_type CoCo -ds 6HN-top_6HN-botv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 2 --freeze_enc_epochs 8 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_fe_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn 
        for nf_aug_ratio in $nf_aug_ratio_list
        do
                # python3 main.py -run $run -nl 1 2 3 -al 1 2  --att_type CoCo -ds 6HN-top_6HN-topv1 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 2 --lr_warm_epochs 2 --lr_warm yes --lr 2e-5 -nf_inp diff --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_nf_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn
                python3 main.py -run $run -nl 0 2 3 -al 2  --att_type CoCo -ds 6HN-top_6HN-topv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 2 --lr_warm_epochs 2 --lr_warm yes --lr 2e-5 -nf_inp diff --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_joint --viz yes --th_pix 4 --pro no --is_open no --is_close yes --w_fe 0.8 --fn $fn 
                # python3 main.py -run $run -nl 0 2 3 -al 2  --att_type CoCo -ds 2nd-not_2nd-notv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 2 --lr_warm_epochs 2 --lr_warm yes --lr 2e-5 -nf_inp diff --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_joint --viz yes --th_pix 4 --pro no --is_open no --is_close yes --w_fe 0.8 --fn $fn --data_path /home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/WAFER/cropped_data/overlapped_30 --aug_data_path /home/dspl/nyla//Anomaly_with_NF/Dataset/aug_data/ATI/WAFER/cropped_data/overlapped_30
                # python3 main.py -run $run -nl 1 2 3 -al 1 2  --att_type CoCo -ds 6HN-top_6HN-botv1 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 2 --lr_warm_epochs 2 --lr_warm yes --lr 2e-5 -nf_inp diff --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_nf_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn
                # python3 main.py -run $run -nl 1 2 3 -al 1 2  --att_type CoCo -ds 6HN-top_6HN-botv1 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 2 --lr_warm_epochs 2 --lr_warm yes --lr 2e-5 -nf_inp diff --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_joint --viz yes --th_pix 4 --pro no --is_open no --is_close yes --w_fe 0.8 --fn $fn
                done
        done
done
