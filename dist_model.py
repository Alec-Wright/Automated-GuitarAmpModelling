import CoreAudioML as Core
import argparse

prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset''')

# arguments for the training/test data locations and file names
prsr.add_argument('--device', '-p', default='Nothing', help='This label describes what device is being modelled')
prsr.add_argument('--data_location', '-dl', default='..', help='Location of the "Data" directory')
prsr.add_argument('--file_name', '-fn', default='..',
                  help='The filename of the wav file to be loaded as the input/target data, the script looks for files '
                       'with thefilename and the extensions -input.wav and -target.wav ')

# pre-processing of the training/val/test data
prsr.add_argument('--segment_length', '-sl', type=int, default=22050, help='Training audio segment length in samples')

# number of epochs and validation
prsr.add_argument('--epochs', '-eps', type=int, default=500, help='Max number of training epochs to run')
prsr.add_argument('--validation_f', '-vfr', type=int, default=2, help='Validation Frequency (in epochs)')
prsr.add_argument('--validation_p', '-vp', type=int, default=50,
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



prsr.add_argument('--loss_fcns', '-lf', default='ESRPre', help='Which loss function, ESR, ')

prsr.add_argument('--val_chunk', '-vs', type=int, default=200000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
prsr.add_argument('--test_chunk', '-tc', type=int, default=200000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')

# arguments for the network structure
prsr.add_argument('--num_layers', '-nl', default=2, type=int, help='Number of recurrent layers')
prsr.add_argument('--hidden_size', '-hs', default=64, type=int, help='Recurrent hidden state size')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU')

prsr.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                  default=True)

prsr.add_argument('--load_config', '-l', type=str, help="Config file path, to a JSON config file, any arguments"
                                                          "listed in the config file will replace the default values"
                  , default=None)
prsr.add_argument('--printing', '-pr', action="store", dest="prin", type=str, help="set to true to turn on console output",
                  default=True)

