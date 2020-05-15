#!/bin/bash

wget https://srv-file9.gofile.io/download/QmRlVq/data.zip
unzip data.zip
rm data.zip

mkdir -p ./rec_results/amazon_men/
mkdir -p ./rec_results/amazon_women/
mkdir -p ./rec_results/tradesy/

mkdir -p ./rec_model_weights/amazon_men/
mkdir -p ./rec_model_weights/amazon_women/
mkdir -p ./rec_model_weights/tradesy/
