mp='./../models'

# twin-Comb
run=0

fn_list="-1"
nf_aug_ratio_list="0 0.2"
is_full_list="yes"
for is_full in $is_full_list
do
        for fn in $fn_list
        do
                python3 main.py -run $run -nl 1 2 3 -al 1 2 -inp 256 -bs 16 --aug_ratio_train 0.8 --att_type Comb -arch resnet18-skip123-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 2 --freeze_enc_epochs 8 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_fe_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn 
        for nf_aug_ratio in $nf_aug_ratio_list
        do
                python3 main.py -run $run -nl 1 2 3 -al 1 2 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 --att_type Comb -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 6 --lr_warm_epochs 2 --lr_warm yes -nf_inp diff --lr 2e-5 --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_nf_only --viz yes --th_pix 4 --pro no --is_open no --is_close yes --fn $fn
                python3 main.py -run $run -nl 1 2 3 -al 1 2 --aug_ratio_train 0.8 -arch resnet18-skip123-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 --att_type Comb -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 6 --lr_warm_epochs 2 --lr_warm yes -nf_inp diff --lr 2e-5 --eval_epoch 1 --is_train no --k_size 1 --is_full $is_full --save_all_tn_sample no --infer_type twin_joint --viz yes --th_pix 4 --pro no --is_open no --is_close yes --w_fe 0.5 --fn $fn
                done
        done
done
