#!/bin/bash

LABELS_TO_EVAL=('FOOD#QUALITY' 'SERVICE#GENERAL' 'RESTAURANT#GENERAL' 'AMBIENCE#GENERAL' 'FOOD#STYLE_OPTIONS' 'RESTAURANT#MISCELLANEOUS' 'FOOD#PRICES' 'RESTAURANT#PRICES' 'DRINKS#QUALITY' 'DRINKS#STYLE_OPTIONS' 'LOCATION#GENERAL' 'DRINKS#PRICES')
GENRE='restaurant'

for label in ${LABELS_TO_EVAL[@]}
do
    echo 'Processing: ', $label
    frameworkpython ./svm/eval.py -g $GENRE -m ./svm/runs/ -l $label -o ./results -p
done
