#!/bin/bash

PRE=depwise_conv2d
SUF=nchw
TARGETS=(
    "${PRE}_fwd_${SUF}"
    "${PRE}_bwd_filter_${SUF}"
    "${PRE}_bwd_input_${SUF}"
    "${PRE}_fwd_${SUF}_fast"
    "${PRE}_bwd_filter_${SUF}_fast"
    "${PRE}_bwd_input_${SUF}_fast"
)
LOGS=()

for INDEX in ${!TARGETS[@]}; do
  OUT="$(mktemp out_XXX.txt)"
  BIN=${TARGETS[$INDEX]}.out
  
  ## Varying N
  for (( i = 1 ; i < 20 ; i+=2 ))
  do
    ./$BIN $i 128 128 144 3 3 |& tee -a $OUT
  done
  
  ## Varying H/W
  for (( i = 100 ; i < 2000 ; i+=200 ))
  do
    ./$BIN 1 $i $i 144 3 3 |& tee -a $OUT
  done
  
  ## Varying C
  for (( i = 10 ; i < 200 ; i+=20 ))
  do
    ./$BIN 1 128 128 $i 3 3 |& tee -a $OUT
  done
  
  ## Varying R/S
  for (( i = 1 ; i < 10 ; i+=1 ))
  do
    ./$BIN 1 128 128 144 $i $i |& tee -a $OUT
  done

  grep 'LOG: time' $OUT | awk '{print $4}' > log_tmp_$INDEX.txt
  LOGS+=("log_tmp_$INDEX.txt")
  rm -rf $OUT
done

LOG="$(mktemp log_XXX.txt)"
paste ${LOGS[@]} -d ',' > $LOG

echo "Log is stored in $LOG"
rm -rf $LOGS.txt


