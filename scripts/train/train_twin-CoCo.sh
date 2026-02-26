mp='./../models'
#
# twin-CoCo
run=0
nf_aug_ratio_list="0"

python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type CoCo -ds 6HN-top_6HN-topv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 3 --freeze_enc_epochs 7 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 

for nf_aug_ratio in $nf_aug_ratio_list
do
       python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type CoCo -ds 6HN-top_6HN-topv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 1 --lr_warm_epochs 2 --lr_warm yes --lr 5e-5 -nf_inp transformed --eval_epoch 1 
done


python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type Comb -ds 6HN-top_6HN-topv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 3 --freeze_enc_epochs 7 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 

for nf_aug_ratio in $nf_aug_ratio_list
do
       python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type Comb -ds 6HN-top_6HN-topv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 1 --lr_warm_epochs 2 --lr_warm yes --lr 5e-5 -nf_inp transformed --eval_epoch 1 
done

python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type CoCo -ds 6HN-bot_6HN-botv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 3 --freeze_enc_epochs 7 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 

for nf_aug_ratio in $nf_aug_ratio_list
do
       python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type CoCo -ds 6HN-bot_6HN-botv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 1 --lr_warm_epochs 2 --lr_warm yes --lr 5e-5 -nf_inp transformed --eval_epoch 1 
done


python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type Comb -ds 6HN-bot_6HN-botv1 -inp 256 -bs 16 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --loss_type smooth_cls --model_path $mp --train_type twin_fe_only --pretrained yes --finetuning yes --meta_epochs 3 --freeze_enc_epochs 7 --repeat_num 2 --sub_epochs 8 --lr 1e-4 --lr_warm_epochs 0 --lr_warm no --eval_epoch 2 

for nf_aug_ratio in $nf_aug_ratio_list
do
       python3 main.py -run $run -nl 0 2 3 -al 2 3 --att_type Comb -ds 6HN-bot_6HN-botv1 --aug_ratio_train 0.8 -arch resnet18-skip023-dec8_glow --nf_aug_ratio_train $nf_aug_ratio -bs 32 -inp 256 --loss_type smooth_cls --model_path $mp --train_type twin_nf_only --pretrained yes --finetuning yes --meta_epochs 8 --sub_epochs 3 --repeat_num 1 --lr_warm_epochs 2 --lr_warm yes --lr 5e-5 -nf_inp transformed --eval_epoch 1 
done