import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import time
import os
import math
from scipy.io.wavfile import write

prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--device', '-p', default='ht1', help='This label describes what device is being modelled')
prsr.add_argument('--data_location', '-dl', default='..', help='Location of the "Data" directory')
prsr.add_argument('--file_name', '-fn', default='ht1',
                  help='The filename of the wav file to be loaded as the input/target data, the script looks for files'
                       'with the filename and the extensions -input.wav and -target.wav ')
prsr.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                  , default='')
prsr.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
prsr.add_argument('--save_location', '-sloc', default='Results', help='Directory where trained models will be saved')
prsr.add_argument('--load_model', '-lm', default=True, help='load a pretrained model if it is found')

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
prsr.add_argument('--up_fr', '-uf', type=int, default=1000,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')

# loss function/s
prsr.add_argument('--loss_fcns', '-lf', default={'ESRPre': 0.5, 'DC': 0.5},
                  help='Which loss functions, ESR, ESRPre, DC. Argument is a dictionary with each key representing a'
                       'loss function name and the corresponding value being the multiplication factor applied to that'
                       'loss function, used to control the contribution of each loss function to the overall loss ')
prsr.add_argument('--pre_filt',   '-pc',   default=[1, -0.85],
                    help='FIR filter coefficients for pre-emphasis filter, can also read in a csv file')

# the validation and test sets are divided into shorter chunks before processing to reduce the amount of GPU memory used
# you can probably ignore this unless during training you get a 'cuda out of memory' error
prsr.add_argument('--val_chunk', '-vs', type=int, default=1000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
prsr.add_argument('--test_chunk', '-tc', type=int, default=1000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')

# arguments for the network structure
prsr.add_argument('--input_size', '-is', default=1, type=int, help='1 for mono input data, 2 for stereo, etc ')
prsr.add_argument('--output_size', '-os', default=1, type=int, help='1 for mono output data, 2 for stereo, etc ')
prsr.add_argument('--num_blocks', '-nb', default=1, type=int, help='Number of recurrent blocks')
prsr.add_argument('--hidden_size', '-hs', default=16, type=int, help='Recurrent unit hidden state size')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU or RNN')

args = prsr.parse_args()
if __name__ == "__main__":
    """The main method creates the recurrent network, trains it and carries out validation/testing """
    start_time = time.time()

    # If a load_config argument was provided, construct the file path to the config file
    if args.load_config:
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        configs = miscfuncs.json_load(args.load_config, args.config_location)
        for parameters in configs:
            args.__setattr__(parameters, configs[parameters])

    # Generate name of directory where results will be saved
    save_path = os.path.join(args.save_location, args.device + args.load_config)

    # Check if an existing saved model exists, and load it, otherwise creates a new model
    network = networks.init_model(save_path, args)

    # Check if a cuda device is available
    if not torch.cuda.is_available():
        print('cuda device not available')
        cuda = 0
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print('cuda device available')
        network = network.cuda()
        cuda = 1

    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate)

    loss_functions = training.LossWrapper(args.loss_fcns, args.pre_filt)

    dataset = dataset.DataSet(data_dir='Data')

    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(os.path.join('train', args.file_name), 'train')

    dataset.create_subset('val')
    dataset.load_file(os.path.join('val', args.file_name), 'val')

    dataset.create_subset('test')
    dataset.load_file(os.path.join('test', args.file_name), 'test')


    bs = args.batch_size
    writer = SummaryWriter()
    for epoch in range(network.training_info['current_epoch'] + 1, args.epochs + 1):
        epoch_start_time = time.time()
        network.reset_hidden()

        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(dataset.subsets['train'].data['input'][0].shape[1])
        # Iterate over the batches
        epoch_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / args.batch_size)):
            # Load batch of shuffled segments
            input_batch = dataset.subsets['train'].data['input'][0][:, shuffle[batch_i*bs:(batch_i+1)*bs], :]
            target_batch = dataset.subsets['train'].data['target'][0][:, shuffle[batch_i*bs:(batch_i+1)*bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            network(input_batch[0:args.init_len, :, :])
            network.zero_grad()

            batch_loss = 0
            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = args.init_len
            # Iterate over the remaining samples in the sequence batch
            for k in range(math.ceil((input_batch.shape[0] - args.init_len) / args.up_fr)):

                # Process input batch with neural network
                output = network(input_batch[start_i:start_i + args.up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_functions(output, target_batch[start_i:start_i + args.up_fr, :, :])
                loss.backward()
                optimiser.step()

                # Set the network hidden state, to detach it from the computation graph
                network.detach_hidden()
                network.zero_grad()
                network.reset_hidden()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += args.up_fr
                batch_loss += loss.item()

            # Add the average batch loss to the epoch loss
            epoch_loss += batch_loss/(k+1)
        # Find the epoch training loss
        network.training_info['training_losses'].append(epoch_loss / (batch_i + 1))
        writer.add_scalar('Loss/train', network.training_info['training_losses'][-1], epoch)

        # Run validation
        if epoch % args.validation_f == 0:
            network.reset_hidden()
            with torch.no_grad():
                val_output = torch.empty_like(dataset.subsets['val'].data['target'][0])
                for l in range(int(val_output.size()[0] / args.val_chunk)):
                    val_output[l * args.val_chunk:(l + 1) * args.val_chunk] = \
                        network(dataset.subsets['val'].data['input'][0][l * args.val_chunk:(l + 1) * args.val_chunk])
                    network.detach_hidden()
                # If the validation set doesn't divide evenly into the validation chunk length, process the remainder
                if not (val_output.size()[0] / args.val_chunk).is_integer():
                    val_output[(l + 1) * args.val_chunk:-1] = \
                        network(dataset.subsets['val'].data['input'][0][(l + 1) * args.val_chunk:-1])
                network.reset_hidden()

                val_loss = loss_functions(val_output, dataset.subsets['val'].data['target'][0])
                network.training_info['validation_losses'].append(val_loss.item())
                writer.add_scalar('Loss/val', network.training_info['validation_losses'][-1], epoch)

                if val_loss < network.training_info['best_val_loss']:
                    network.training_info['best_val_loss'] = val_loss.item()
                    with open(os.path.join(save_path, 'bestvloss.txt'), 'w') as f:
                        f.write(str(val_loss.item()))
                    network.training_info['current_epoch'] = epoch
                    network.save_model('model_best', save_path)
                    write(os.path.join(save_path,"best_val_out.wav"),
                          dataset.subsets['test'].fs, val_output.cpu().numpy()[:, 0, 0])

        network.training_info['epoch_times'].append((time.time() - epoch_start_time)/60)
        network.training_info['current_epoch'] = epoch

        network.save_model('model', save_path)
