# Automated-Guitar Amplifier Modelling

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory. 

### Cloning this repository

To create a working local copy of this repository, use the following command:

git clone --recurse-submodules https://github.com/Alec-Wright/Automated-GuitarAmpModelling

### Python Environment

Using this repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed. 
Additionally this repository uses the 'CoreAudioML' package, which is included as a submodule. Cloining the repo as described in 'Cloning this repository' ensures the CoreAudioML package is also downloaded.

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. At the top of the file the 'argparser' contains a description of all the training script arguments, as well as their default values. To train a model using the default arguments, simply run the model from the command line as follows:

python dist_model_recnet.py

note that you must run this command from a python environment that has the libraries described in 'Python Environment' installed. To use different arguments in the training script you can change the default arguments directly in 'dist_model_recnet.py', or you can direct the 'dist_model_recnet.py' script to look for a config file that contains different arguments, for example by running the script using the following command:

python dist_model_recnet.py -l "ht11.json"

Where in this case the script will look for the file ht11.json in the the 'Configs' directory. To create custom config files, the ht11.json file provided can be edited in any text editor.

During training, the script will save some data to a folder in the Results directory. These are, the lowest loss achieved on the validation set so far in 'bestvloss.txt', as well as a copy of that model 'model_best.json', and the audio created by that model 'best_val_out.wav'. The neural network at the end of the most recent training epoch is also saved, as 'model.json'. When training is complete the test dataset is processed, and the audio produced and the test loss is also saved to the same directory.

A trained model contained in one of the 'model.json' or 'model_best.json' files can be loaded, see the 'proc_audio.py' script for an example of how this is done.

### Feedback

This repository is still a work in progress, and I welcome your feedback! Either by raising an Issue or in the 'Discussions' tab 
