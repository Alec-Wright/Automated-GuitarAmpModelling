import CoreAudioML as Core
import argparse
import time
import os
import json

prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--device', '-p', default='Nothing', help='This label describes what device is being modelled')
prsr.add_argument('--data_location', '-dl', default='..', help='Location of the "Data" directory')
prsr.add_argument('--file_name', '-fn', default='..',
                  help='The filename of the wav file to be loaded as the input/target data, the script looks for files '
                       'with the filename and the extensions -input.wav and -target.wav ')
prsr.add_argument('--load_config', '-l', type=str, 
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                  , default='figgy1')
prsr.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
prsr.add_argument('--save_location', '-sloc', default='Results', help='Directory where trained models will be saved')

# pre-processing of the training/val/test data
prsr.add_argument('--segment_length', '-slen', type=int, default=22050, help='Training audio segment length in samples')

# number of epochs and validation
prsr.add_argument('--epochs', '-eps', type=int, default=500, help='Max number of training epochs to run')
prsr.add_argument('--validation_f', '-vfr', type=int, default=2, help='Validation Frequency (in epochs)')
prsr.add_argument('--validation_p', '-vp', type=int, default=0,
                  help='How many validations without improvement before stopping training, None for no early stopping')

# settings for the training epoch
prsr.add_argument('--batch_size', '-bs', type=int, default=50, help='Training mini-batch size')
prsr.add_argument('--iter_num', '-it', type=int, default=None,
                  help='Overrides --batch_size and instead sets the batch_size so that a total of --iter_num batches'
                       'are processed in each epoch')
prsr.add_argument('--learn_rate', '-lr', type=float, default=0.0005, help='Initial learning rate')
prsr.add_argument('--init_len', '-il', type=int, default=200,
                  help='Number of sequence samples to process before starting weight updates')
prsr.add_argument('--update_freq', '-uf', type=int, default=1000,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')

# loss function/s
prsr.add_argument('--loss_fcns', '-lf', default={'ESRPre': 0.5, 'DC': 0.5},
                  help='Which loss functions, ESR, ESRPre, DC. Argument is a dictionary with each key representing a'
                       'loss function name and the corresponding value being the multiplication factor applied to that'
                       'loss function, used to control the contribution of each loss function to the overall loss ')

# the validation and test sets are divided into shorter chunks before processing to reduce the amount of GPU memory used
# you can probably ignore this unless during training you get a 'cuda out of memory' error
prsr.add_argument('--val_chunk', '-vs', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
prsr.add_argument('--test_chunk', '-tc', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')

# arguments for the network structure
prsr.add_argument('--num_blocks', '-nb', default=1, type=int, help='Number of recurrent blocks')
prsr.add_argument('--hidden_size', '-hs', default=64, type=int, help='Recurrent unit hidden state size')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU or RNN')

args = prsr.parse_args()
if __name__ == "__main__":
    """The main method creates the recurrent network, trains it and carries out validation/testing """
    start_time = time.time()

    # If a load_config argument was provided, create the correct
    if args.load_config:
        config_path = os.path.join(args.config_location, args.load_config)
        config_path = config_path + '.json' if not config_path.endswith('.json') else config_path
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        with open(config_path, 'r') as f:
            configs = json.load(f)
            for parameters in configs:
                args.__setattr__(parameters, configs[parameters])
        dirPath = 'Results/' + args.pedal + ''.join([s for s in args.load_config if s.isdigit()])

