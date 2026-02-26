#!/usr/bin/env bash
set -euo pipefail

##### 고정값 (변경 안 함) #####
declare -r MP="./../models/fixed_datasets"
declare -r NL="0 2 3"
declare -r INP=256
declare -r LOSS_TYPE="smooth_cls"
declare -r BS_FE=16
declare -r BS_NF=32
declare -r AUG_RATIO_TRAIN=1.0
declare -r ARCH="resnet18-skip023-dec4_glow"
declare -r LR_WARM_EPOCHS_FE=0
declare -r LR_WARM_FE="no"
declare -r NF_AUG_RATIO_LIST="0"

declare -r K_SIZE=1
declare -r TH_PIX=4
declare -r PRO="no"
declare -r IS_OPEN="no"
declare -r IS_CLOSE="yes"
declare -r SAVE_ALL_TN="no"
declare -r IS_VIS="yes"
declare -r SAVE_FEAT="no"
declare -r VIZ_DIFF_FEAT="no"
declare -r VIZ_TOTAL_FEAT="yes"
declare -r W_FE=0.2
declare -r TH_MANUAL=0.5
declare -r BEST_W_FE="yes"

##### 바꿀 수 있는 값 (환경변수 override 가능) #####
RUN="${RUN:-3}"
DATASETS=${DATASETS:-"6HN-top_6HN-topv1" "6HN-top_6HN-botv1"}
INFER_WHAT="${INFER_WHAT:-fe}" # FE/NF 실행 모드: fe | nf | joint | everything
FN="${FN:-"-1"}" # -1: 전체, 그 외 숫자: 특정 fn만
RE_LIST="${RE_LIST:-"0 1 2"}" # 반복 실행 인덱스 리스트 (override 가능)

# 데이터 경로 (override 가능)
##### 데이터 경로 (초기 디폴트는 PCB로 설정) #####
DATA_PATH_PCB="/home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/PCB/cropped_data/overlapped_30"
AUG_DATA_PATH_PCB="/home/dspl/nyla/Anomaly_with_NF/Dataset/aug_full_data/ATI/PCB/cropped_data/overlapped_30"
DATA_PATH_WAFER="/home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/WAFER/cropped_data/overlapped_30"
AUG_DATA_PATH_WAFER="/home/dspl/nyla/Anomaly_with_NF/Dataset/aug_full_data/ATI/WAFER/cropped_data/overlapped_30"

META_EPOCHS_FE="${META_EPOCHS_FE:-4}"
FREEZE_ENC_EPOCHS="${FREEZE_ENC_EPOCHS:-8}"
REPEAT_NUM_FE="${REPEAT_NUM_FE:-1}"
SUB_EPOCHS_FE="${SUB_EPOCHS_FE:-4}"
EVAL_EPOCH_FE="${EVAL_EPOCH_FE:-2}"
LR_DECAY_FE="${LR_DECAY_FE:-0.9}"
LR_FE="${LR_FE:-1e-4}"
W_DECAY_FE="${W_DECAY_FE:-1e-5}"

META_EPOCHS_NF="${META_EPOCHS_NF:-4}"
REPEAT_NUM_NF="${REPEAT_NUM_NF:-1}"
SUB_EPOCHS_NF="${SUB_EPOCHS_NF:-3}"
EVAL_EPOCH_NF="${EVAL_EPOCH_NF:-1}"
LR_WARM_EPOCHS_NF="${LR_WARM_EPOCHS_NF:-2}"
LR_WARM_NF="${LR_WARM_NF:-yes}"
LR_NFS="${LR_NFS:-"1e-6 3e-6 3e-6"}"
LR_DECAY_NF="${LR_DECAY_NF:-0.1}"

##### 공통 옵션 #####
common_base() {
  local ds="$1"
  
  echo -run "$RUN" -nl "$NL" \
       -ds "$ds" -inp "$INP" --aug_ratio_train "$AUG_RATIO_TRAIN" \
       -arch "$ARCH" --loss_type "$LOSS_TYPE" --model_path "$MP" \
       --data_path "$DATA_PATH" --aug_data_path "$AUG_DATA_PATH" -re "$RE" \
       --is_full "$IS_FULL" --k_size "$K_SIZE" --save_all_tn_sample "$SAVE_ALL_TN" --viz "$IS_VIS" --th_pix "$TH_PIX" --pro "$PRO" --is_open "$IS_OPEN" --is_close "$IS_CLOSE" --fn "$FN" \
       --th_manual "$TH_MANUAL" --save_feat "$SAVE_FEAT" --viz_total_feat "$VIZ_TOTAL_FEAT" --viz_diff_feat "$VIZ_DIFF_FEAT"
}

##### 실행 함수 #####
run_fe() {
  local ds="$1"
  python3 main.py \
    $(common_base "$ds") \
    --w_decay "$W_DECAY_FE" \
    --train_type single_fe_only --pretrained yes --finetuning yes \
    -bs "$BS_FE" --meta_epochs "$META_EPOCHS_FE" --freeze_enc_epochs "$FREEZE_ENC_EPOCHS" \
    --repeat_num "$REPEAT_NUM_FE" --sub_epochs "$SUB_EPOCHS_FE" \
    --lr "$LR_FE" --lr_warm_epochs "$LR_WARM_EPOCHS_FE" --lr_warm "$LR_WARM_FE" --lr_decay_rate "$LR_DECAY_FE" \
    --eval_epoch "$EVAL_EPOCH_FE" --is_train no --infer_type single_fe_only 
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
    $(common_base "$ds") \
    --train_type single_nf_only --pretrained yes --finetuning yes \
    --nf_aug_ratio_train "$nf_ratio" -bs "$BS_NF" \
    --meta_epochs "$META_EPOCHS_NF" --sub_epochs "$SUB_EPOCHS_NF" --repeat_num "$REPEAT_NUM_NF" \
    --lr_warm_epochs "$LR_WARM_EPOCHS_NF" --lr_warm "$LR_WARM_NF" --lr_decay_rate "$LR_DECAY_NF" \
    --lr_nf_list "${LR_NF_ARR[@]}" --eval_epoch "$EVAL_EPOCH_NF" \
    --is_train no --infer_type single_nf_only
}


run_joint() {
  local ds="$1"; local nf_ratio="$2"

  # 값이 비었는지 체크(선택)
  if [[ ${#LR_NF_ARR[@]} -eq 0 ]]; then
    echo "ERROR: LR_NFS is empty. e.g., LR_NFS=\"1e-4 5e-5 2e-5\"" >&2
    exit 1
  fi

  python3 main.py \
    $(common_base "$ds") \
    --train_type single_nf_only --pretrained yes --finetuning yes \
    --nf_aug_ratio_train "$nf_ratio" -bs "$BS_NF" \
    --meta_epochs "$META_EPOCHS_NF" --sub_epochs "$SUB_EPOCHS_NF" --repeat_num "$REPEAT_NUM_NF" \
    --lr_warm_epochs "$LR_WARM_EPOCHS_NF" --lr_warm "$LR_WARM_NF" --lr_decay_rate "$LR_DECAY_NF" \
    --lr_nf_list "${LR_NF_ARR[@]}" --eval_epoch "$EVAL_EPOCH_NF" \
    --is_train no --infer_type single_joint --w_fe "$W_FE" --get_best_w_fe "$BEST_W_FE"
}

##### 모드 체크 #####
case "$INFER_WHAT" in
  fe|FE)   DO_FE=1; DO_NF=0; DO_JOINT=0 ;;
  nf|NF)   DO_FE=0; DO_NF=1; DO_JOINT=0;;
  joint|JOINT) DO_FE=0; DO_NF=0; DO_JOINT=1 ;;
  everything|EVERYTHING) DO_FE=1; DO_NF=1; DO_JOINT=1 ;;
  *)
    echo "ERROR: INFER_WHAT must be one of: fe | nf | joint |everything (got: $INFER_WHAT)" >&2
    exit 1
    ;;
esac


IFS=' ' read -r -a RES <<< "$RE_LIST"
##### 메인 #####
for ds in $DATASETS; do
  for RUN_IDX in "${!RES[@]}"; do
    RE="${RES[$RUN_IDX]}"
    echo "==============================="
    echo "[Dataset] $ds"
    echo "[Index]   $RUN_IDX"
    echo "[RepeatNum]    $RE"
    echo "[RunTag]  $RUN-$RUN_IDX"
    echo "==============================="

  if [[ "$ds" == *"2nd"* ]]; then
    IS_FULL="no"
    DATA_PATH="${DATA_PATH:-$DATA_PATH_WAFER}"
    AUG_DATA_PATH="${AUG_DATA_PATH:-$AUG_DATA_PATH_WAFER}"
  else
    IS_FULL="yes"
    DATA_PATH="${DATA_PATH:-$DATA_PATH_PCB}"
    AUG_DATA_PATH="${AUG_DATA_PATH:-$AUG_DATA_PATH_PCB}"
  fi
  if [[ $DO_FE -eq 1 ]]; then
    run_fe "$ds"
  fi
  if [[ $DO_NF -eq 1 ]]; then
    for nf_ratio in $NF_AUG_RATIO_LIST; do
      run_nf "$ds" "$nf_ratio"
    done
  fi
  if [[ $DO_JOINT -eq 1 ]]; then
    for nf_ratio in $NF_AUG_RATIO_LIST; do
      run_joint "$ds" "$nf_ratio"
    done
  fi
  done
done
