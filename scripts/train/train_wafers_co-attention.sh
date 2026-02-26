#!/usr/bin/env bash
set -euo pipefail

##### 고정값 (변경 안 함) #####
declare -r MP="./../models/fixed_datasets"
declare -r NL="0 1"
declare -r AL="0 1"
declare -r INP=256
declare -r LOSS_TYPE="smooth_cls"
declare -r BS_FE=16
declare -r BS_NF=32
declare -r AUG_RATIO_TRAIN=1.0
declare -r ARCH="resnet18-skip01-dec4_glow"
declare -r LR_WARM_EPOCHS_FE=0
declare -r LR_WARM_FE="no"
declare -r NF_AUG_RATIO_LIST="0"
declare -r IS_FULL="no"

##### 바꿀 수 있는 값 (환경변수 override 가능) #####
RUN="${RUN:-2}"
DATASETS=${DATASETS:-"2nd-not_2nd-notv1"}
ATT_TYPES=${ATT_TYPES:-"CoCo"}
TRAIN_WHAT="${TRAIN_WHAT:-fe}" # FE/NF 실행 모드: fe | nf | both

RE_LIST="${RE_LIST:-"0 1 2"}" # 반복 실행 인덱스 리스트 (override 가능)
# SEED_LIST="${SEED_LIST:-"1014"}"

# 데이터 경로 (override 가능)
DATA_PATH="${DATA_PATH:-/home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/WAFER/cropped_data/overlapped_30}"
AUG_DATA_PATH="${AUG_DATA_PATH:-/home/dspl/nyla/Anomaly_with_NF/Dataset/aug_full_data/ATI/WAFER/cropped_data/overlapped_30}"

META_EPOCHS_FE="${META_EPOCHS_FE:-8}"
FREEZE_ENC_EPOCHS="${FREEZE_ENC_EPOCHS:-16}"
REPEAT_NUM_FE="${REPEAT_NUM_FE:-1}"
SUB_EPOCHS_FE="${SUB_EPOCHS_FE:-1}"
EVAL_EPOCH_FE="${EVAL_EPOCH_FE:-8}"
LR_DECAY_FE="${LR_DECAY_FE:-0.1}"
LR_FE="${LR_FE:-3e-5}"
W_DECAY_FE="${W_DECAY_FE:-1e-4}"

META_EPOCHS_NF="${META_EPOCHS_NF:-4}"
REPEAT_NUM_NF="${REPEAT_NUM_NF:-1}"
SUB_EPOCHS_NF="${SUB_EPOCHS_NF:-3}"
EVAL_EPOCH_NF="${EVAL_EPOCH_NF:-1}"
LR_WARM_EPOCHS_NF="${LR_WARM_EPOCHS_NF:-2}"
LR_WARM_NF="${LR_WARM_NF:-yes}"
LR_NFS="${LR_NFS:-"6e-6 1e-7 1e-7"}"
LR_DECAY_NF="${LR_DECAY_NF:-0.1}"
NF_INP="${NF_INP:-transformed}"

##### 공통 옵션 ##### 
common_base() {
  echo -run "$RUN" -nl $NL -al $AL \
       -ds "$1" -inp "$INP" --aug_ratio_train "$AUG_RATIO_TRAIN" \
       -arch "$ARCH" --loss_type "$LOSS_TYPE" --model_path "$MP" \
       --data_path "$DATA_PATH" --aug_data_path "$AUG_DATA_PATH"  -re "$RE" --is_full "$IS_FULL" 
}

##### 실행 함수 #####
run_fe() {
  local att="$1"; local ds="$2"
  python3 main.py \
    $(common_base "$ds") \
    --att_type "$att" --w_decay "$W_DECAY_FE" \
    --train_type twin_fe_only --pretrained yes --finetuning yes \
    -bs "$BS_FE" --meta_epochs "$META_EPOCHS_FE" --freeze_enc_epochs "$FREEZE_ENC_EPOCHS" \
    --repeat_num "$REPEAT_NUM_FE" --sub_epochs "$SUB_EPOCHS_FE" \
    --lr "$LR_FE" --lr_warm_epochs "$LR_WARM_EPOCHS_FE" --lr_warm "$LR_WARM_FE" --lr_decay_rate "$LR_DECAY_FE" \
    --eval_epoch "$EVAL_EPOCH_FE"
}

# 공백 구분해 배열로 변환 (견고)
IFS=' ' read -r -a LR_NF_ARR <<< "$LR_NFS"

run_nf() {
  local att="$1"; local ds="$2"; local nf_ratio="$3"

  # 값이 비었는지 체크(선택)
  if [[ ${#LR_NF_ARR[@]} -eq 0 ]]; then
    echo "ERROR: LR_NFS is empty. e.g., LR_NFS=\"1e-4 5e-5 2e-5\"" >&2
    exit 1
  fi

  python3 main.py \
    $(common_base "$ds") \
    --att_type "$att" \
    --train_type twin_nf_only --pretrained yes --finetuning yes \
    --nf_aug_ratio_train "$nf_ratio" -bs "$BS_NF" \
    --meta_epochs "$META_EPOCHS_NF" --sub_epochs "$SUB_EPOCHS_NF" --repeat_num "$REPEAT_NUM_NF" \
    --lr_warm_epochs "$LR_WARM_EPOCHS_NF" --lr_warm "$LR_WARM_NF" --lr_decay_rate "$LR_DECAY_NF" \
    --lr_nf_list "${LR_NF_ARR[@]}" \
    -nf_inp "$NF_INP" --eval_epoch "$EVAL_EPOCH_NF"
}

##### 모드 체크 #####
case "$TRAIN_WHAT" in
  fe|FE)   DO_FE=1; DO_NF=0 ;;
  nf|NF)   DO_FE=0; DO_NF=1 ;;
  both|BOTH) DO_FE=1; DO_NF=1 ;;
  *)
    echo "ERROR: TRAIN_WHAT must be one of: fe | nf | both (got: $TRAIN_WHAT)" >&2
    exit 1
    ;;
esac

##### 메인 #####
# seed 배열화 및 인덱스 루프
IFS=' ' read -r -a RES <<< "$RE_LIST"
for ds in $DATASETS; do
  for att in $ATT_TYPES; do
    echo "==============================="
    echo "[Attention Type] $att"
    echo "==============================="
    for RUN_IDX in "${!RES[@]}"; do
      RE="${RES[$RUN_IDX]}"
      echo "==============================="
      echo "[Dataset] $ds"
      echo "[Index]   $RUN_IDX"
      echo "[RepeatNum]    $RE"
      echo "[RunTag]  $RUN-$RUN_IDX"
      echo "==============================="

      if [[ $DO_FE -eq 1 ]]; then
        run_fe "$att" "$ds"
      fi
      if [[ $DO_NF -eq 1 ]]; then
        for nf_ratio in $NF_AUG_RATIO_LIST; do
          run_nf "$att" "$ds" "$nf_ratio"
        done
      fi
    done
  done
done
