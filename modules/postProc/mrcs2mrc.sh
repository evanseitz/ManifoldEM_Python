# Copyright (c) Columbia University Evan Seitz 2019

scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

cd ..
cd ..
cd bin

for filename in *50.star
do
    echo "input: $filename"
    echo "output: ${scriptDir}/${filename%.*}.mrc"
    relion_reconstruct --i "${filename}" --o "${scriptDir}/${filename%.*}.mrc"
done
