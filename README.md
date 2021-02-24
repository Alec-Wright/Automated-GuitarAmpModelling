# NeuralGuitarAmpModelling

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory. 

### Training Script
the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. 
