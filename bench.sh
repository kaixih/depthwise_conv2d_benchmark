#!/bin/bash

PRE=depwise_conv2d
SUF=nchw
EXEC_FILES=(
    "${PRE}_fwd_${SUF}"
    "${PRE}_bwd_filter_${SUF}"
    "${PRE}_bwd_input_${SUF}"
    "${PRE}_fwd_${SUF}_fast"
    "${PRE}_bwd_filter_${SUF}_fast"
    "${PRE}_bwd_input_${SUF}_fast"
  )


LOG="$(mktemp log_XXX.txt)"
for EXEC in ${EXEC_FILES[@]}; do
  OUT="$(mktemp out_XXX.txt)"
  BIN=$EXEC.out
  
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
  TMP="$(mktemp tmp_XXX.txt)"
  grep 'LOG: time' $OUT | awk '{print $4}' > $TMP
  LOG_TMP="$(mktemp log_tmp_XXX.txt)"
  paste $LOG $TMP -d ',' > $LOG_TMP
  mv $LOG_TMP $LOG
  rm $TMP $OUT
done

LOG_TMP="$(mktemp log_tmp_XXX.txt)"
sed 's/^,//' $LOG > $LOG_TMP
mv $LOG_TMP $LOG
echo "Log is stored in $LOG"


