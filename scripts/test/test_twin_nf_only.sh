#!/usr/bin/env bash
set -euo pipefail

##### 고정값 (변경 안 함) #####
declare -r MP="./../models"
declare -r NL="0 2 3"
declare -r INP=256
declare -r BS_NF=32
declare -r ARCH="resnet18-skip023-dec8_glow"
declare -r NF_AUG_RATIO_LIST="0"
declare -r K_SIZE=1
declare -r TH_PIX=4
declare -r PRO="no"
declare -r IS_OPEN="no"
declare -r IS_CLOSE="yes"
declare -r SAVE_ALL_TN="no"
declare -r NF_INP="diff"
declare -r IS_FULL="yes"

##### 바꿀 수 있는 값 (환경변수 override 가능) #####
RUN="${RUN:-6}"
DATASETS=${DATASETS:-"6HN-bot_6HN-botv1" "6HN-bot_6HN-topv1"}
FN="${FN:-"-1"}"               # -1: 전체, 그 외 숫자: 특정 fn만
IS_VIS="${IS_VIS:-yes}"        # yes/no
SAVE_FEAT="${SAVE_FEAT:-no}"   # yes/no
VIZ_TOTAL_FEAT="${VIZ_TOTAL:-no}"
VIZ_DIFF_FEAT="${VIZ_DIFF:-no}"

# 데이터 경로 (override 가능)
DATA_PATH="${DATA_PATH:-/home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/PCB/cropped_data/overlapped_30}"
AUG_DATA_PATH="${AUG_DATA_PATH:-/home/dspl/nyla/Anomaly_with_NF/Dataset/aug_data/ATI/PCB/cropped_data/overlapped_30}"

META_EPOCHS_NF="${META_EPOCHS_NF:-4}"
REPEAT_NUM_NF="${REPEAT_NUM_NF:-1}"
SUB_EPOCHS_NF="${SUB_EPOCHS_NF:-6}"
EVAL_EPOCH_NF="${EVAL_EPOCH_NF:-1}"
LR_WARM_EPOCHS_NF="${LR_WARM_EPOCHS_NF:-2}"
LR_WARM_NF="${LR_WARM_NF:-yes}"
LR_NFS="${LR_NFS:-"7e-5 7e-5 7e-5"}"
LR_DECAY_NF="${LR_DECAY_NF:-0.1}"

##### 공통 옵션 #####
common_base() {
  echo 
}

# 공백 구분해 배열로 변환 (견고)
IFS=' ' read -r -a LR_NF_ARR <<< "$LR_NFS"

run_nf() {
  local ds="$1"; local nf_ratio="$2"

  # 값이 비었는지 체크(선택)
  if [[ ${#LR_NF_ARR[@]} -eq 0 ]]; then
    echo "ERROR: LR_NFS is empty. e.g., LR_NFS=\"1e-4 5e-5 2e-5\"" >&2
    exit 1
  fi

  python3 main.py \
    -run "$RUN" -nl $NL \
    -ds "$1" -inp "$INP" \
    -arch "$ARCH" --model_path "$MP" \
    --data_path "$DATA_PATH" --aug_data_path "$AUG_DATA_PATH" \
    --train_type twin_nf_only --pretrained yes --finetuning no \
    --nf_aug_ratio_train "$nf_ratio" -bs "$BS_NF" \
    --meta_epochs "$META_EPOCHS_NF" --sub_epochs "$SUB_EPOCHS_NF" --repeat_num "$REPEAT_NUM_NF" \
    --lr_warm_epochs "$LR_WARM_EPOCHS_NF" --lr_warm "$LR_WARM_NF" --lr_decay_rate "$LR_DECAY_NF" \
    --lr_nf_list "${LR_NF_ARR[@]}" \
    -nf_inp "$NF_INP" --eval_epoch "$EVAL_EPOCH_NF" \
    --is_full "$IS_FULL" --k_size "$K_SIZE" --save_all_tn_sample "$SAVE_ALL_TN" --viz "$IS_VIS" --th_pix "$TH_PIX" --pro "$PRO" --is_open "$IS_OPEN" --is_close "$IS_CLOSE" --fn "$FN" \
    --is_train no --infer_type twin_nf_only
}

##### 메인 #####
for ds in $DATASETS; do
  for nf_ratio in $NF_AUG_RATIO_LIST; do
    run_nf "$ds" "$nf_ratio"
  done
done
