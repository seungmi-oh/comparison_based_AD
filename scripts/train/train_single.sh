mp='./../models'

# single
run=0
nf_aug_ratio_list="0"

#python3 main.py -run $run -ds 6HN_6HNv1 -inp 256 -bs 32 --aug_ratio_train 0.8 -arch visformer_tiny-skip012-dec8_glow --loss_type smooth_cls --model_path $mp --train_type single_fe_only --pretrained yes --finetuning yes --meta_epochs 2 --freeze_enc_epochs 8 --repeat_num 2 --sub_epochs 8 --lr 2e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 

for nf_aug_ratio in $nf_aug_ratio_list
do
        python3 main.py -run $run -ds 6HN-top_6HN-topv1 -nl 0 2 3 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio --loss_type smooth_cls --model_path $mp --train_type single_nf_only --pretrained yes --finetuning no --meta_epochs 8 --sub_epochs 6 --lr_warm_epochs 2 --repeat_num 1 --lr 1e-5 --eval_epoch 1 
done
