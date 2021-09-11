#!/bin/sh

# Copyright (c) Columbia University Evan Seitz 2019

svdDir="${PWD}"
svdPath="$svdDir/mrc2svd.py"
#echo $svd
cd ..
cd ..
projName=`basename "${PWD##*s_}"`
#echo $proj
cd ..
mainDir="${PWD}"

python $svdPath $projName $mainDir
