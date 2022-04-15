#!/bin/bash

BIN=$1

## Varying N
for (( i = 1 ; i < 20 ; i+=2 ))
do
  ./$BIN $i 128 128 144 3 3
done

## Varying H/W
for (( i = 100 ; i < 2000 ; i+=200 ))
do
  ./$BIN 1 $i $i 144 3 3
done

## Varying C
for (( i = 10 ; i < 200 ; i+=20 ))
do
  ./$BIN 1 128 128 $i 3 3
done

## Varying R/S
for (( i = 1 ; i < 10 ; i+=1 ))
do
  ./$BIN 1 128 128 144 $i $i
done

 

