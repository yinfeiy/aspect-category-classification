#!/bin/bash

LABELS_TO_EVAL=('RESTAURANT#GENERAL')
GENRE='restaurant'

for label in ${LABELS_TO_EVAL[@]}
do
    echo $label
    frameworkpython ./svm/eval.py -g $GENRE -m ./svm/runs/ -l $label
done
