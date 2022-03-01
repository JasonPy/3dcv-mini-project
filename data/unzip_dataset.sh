#!/bin/bash
load_dataset ()  {
    dataset=$1

    unzip -q $dataset
    rm -r $dataset.zip

    zips=$(find $dataset -name "*.zip")

    for i in $zips;
    do 
    echo $i
    zip -d $i \*.db;
    unzip -q $i -d $dataset
    rm -r $i
    done
}

load_dataset $1