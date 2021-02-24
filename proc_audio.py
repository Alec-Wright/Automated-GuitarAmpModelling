import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import argparse
from scipy.io.wavfile import write
import torch
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='''This script takes an input .wav file, loads it and processes it with a neural network model of a
                    device, i.e guitar amp/pedal, and saves the output as a .wav file''')

    parser.add_argument('input_file', default='Data/test/ht1-input.wav', nargs='?',
                        help='file name/location of audio file to be processed')
    parser.add_argument('output_file', default='output.wav', nargs='?',
                        help='file name/location where processed audio will be saved')
    parser.add_argument('model_file', default='Results/ht1-ht11/model_best.json', nargs='?',
                        help='file name/location of .json file that contains the neural network model')
    return parser.parse_args()

def proc_audio(args):
    network_data = miscfuncs.json_load(args.model_file)
    network = networks.load_model(network_data)
    data = dataset.DataSet(data_dir='', extensions='')
    data.create_subset('data')
    data.load_file(args.input_file, set_names='data')
    with torch.no_grad():
        output = network(data.subsets['data'].data['data'][0])
    write(args.output_file, data.subsets['data'].fs, output.cpu().numpy()[:, 0, 0])


def main():
    args = parse_args()
    proc_audio(args)

if __name__ == '__main__':
    main()