#!/bin/bash

# download datasets
wget https://srv-file9.gofile.io/download/QmRlVq/data.zip
unzip data.zip
rm data.zip

# create directories recommendation
mkdir -p ./rec_results/amazon_men/
mkdir -p ./rec_results/amazon_women/
mkdir -p ./rec_results/tradesy/

mkdir -p ./rec_model_weights/amazon_men/
mkdir -p ./rec_model_weights/amazon_women/
mkdir -p ./rec_model_weights/tradesy/

mkdir -p ./chr/amazon_men/
mkdir -p ./chr/amazon_women/
mkdir -p ./chr/tradesy/

mkdir -p ./ncdcg/amazon_men/
mkdir -p ./ncdcg/amazon_women/
mkdir -p ./ncdcg/tradesy/