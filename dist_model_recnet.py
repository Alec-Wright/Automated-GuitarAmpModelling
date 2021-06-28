import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import os
import csv
from scipy.io.wavfile import write


prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--device', '-p', default='ht1', help='This label describes what device is being modelled')
prsr.add_argument('--data_location', '-dl', default='./Data', help='Location of the "Data" directory')
prsr.add_argument('--file_name', '-fn', default='ht1',
                  help='The filename of the wav file to be loaded as the input/target data, the script looks for files'
                       'with the filename and the extensions -input.wav and -target.wav ')
prsr.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                  , default='RNN3')
prsr.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
prsr.add_argument('--save_location', '-sloc', default='Results', help='Directory where trained models will be saved')
prsr.add_argument('--load_model', '-lm', default=True, help='load a pretrained model if it is found')

# pre-processing of the training/val/test data
prsr.add_argument('--segment_length', '-slen', type=int, default=22050, help='Training audio segment length in samples')

# number of epochs and validation
prsr.add_argument('--epochs', '-eps', type=int, default=2000, help='Max number of training epochs to run')
prsr.add_argument('--validation_f', '-vfr', type=int, default=2, help='Validation Frequency (in epochs)')
# TO DO
prsr.add_argument('--validation_p', '-vp', type=int, default=25,
                  help='How many validations without improvement before stopping training, None for no early stopping')

# settings for the training epoch
prsr.add_argument('--batch_size', '-bs', type=int, default=50, help='Training mini-batch size')
prsr.add_argument('--iter_num', '-it', type=int, default=None,
                  help='Overrides --batch_size and instead sets the batch_size so that a total of --iter_num batches'
                       'are processed in each epoch')
prsr.add_argument('--learn_rate', '-lr', type=float, default=0.005, help='Initial learning rate')
prsr.add_argument('--init_len', '-il', type=int, default=200,
                  help='Number of sequence samples to process before starting weight updates')
prsr.add_argument('--up_fr', '-uf', type=int, default=1000,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')
prsr.add_argument('--cuda', '-cu', default=1, help='Use GPU if available')

# loss function/s
prsr.add_argument('--loss_fcns', '-lf', default={'ESRPre': 0.75, 'DC': 0.25},
                  help='Which loss functions, ESR, ESRPre, DC. Argument is a dictionary with each key representing a'
                       'loss function name and the corresponding value being the multiplication factor applied to that'
                       'loss function, used to control the contribution of each loss function to the overall loss ')
prsr.add_argument('--pre_filt',   '-pf',   default='high_pass',
                    help='FIR filter coefficients for pre-emphasis filter, can also read in a csv file')

# the validation and test sets are divided into shorter chunks before processing to reduce the amount of GPU memory used
# you can probably ignore this unless during training you get a 'cuda out of memory' error
prsr.add_argument('--val_chunk', '-vs', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
prsr.add_argument('--test_chunk', '-tc', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')

# arguments for the network structure
prsr.add_argument('--model', '-m', default='SimpleRNN', type=str, help='model architecture')
prsr.add_argument('--input_size', '-is', default=1, type=int, help='1 for mono input data, 2 for stereo, etc ')
prsr.add_argument('--output_size', '-os', default=1, type=int, help='1 for mono output data, 2 for stereo, etc ')
prsr.add_argument('--num_blocks', '-nb', default=1, type=int, help='Number of recurrent blocks')
prsr.add_argument('--hidden_size', '-hs', default=16, type=int, help='Recurrent unit hidden state size')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU or RNN')
prsr.add_argument('--skip_con', '-sc', default=1, help='is there a skip connection for the input to the output')

args = prsr.parse_args()

# This function takes a directory as argument, looks for an existing model file called 'model.json' and loads a network
# from it, after checking the network in 'model.json' matches the architecture described in args. If no model file is
# found, it creates a network according to the specification in args.
def init_model(save_path, args):
    # Search for an existing model in the save directory
    if miscfuncs.file_check('model.json', save_path) and args.load_model:
        print('existing model file found, loading network')
        model_data = miscfuncs.json_load('model', save_path)
        # assertions to check that the model.json file is for the right neural network architecture
        try:
            assert model_data['model_data']['unit_type'] == args.unit_type
            assert model_data['model_data']['input_size'] == args.input_size
            assert model_data['model_data']['hidden_size'] == args.hidden_size
            assert model_data['model_data']['output_size'] == args.output_size
        except AssertionError:
            print("model file found with network structure not matching config file structure")
        network = networks.load_model(model_data)
    # If no existing model is found, create a new one
    else:
        print('no saved model found, creating new network')
        network = networks.SimpleRNN(input_size=args.input_size, unit_type=args.unit_type, hidden_size=args.hidden_size,
                                     output_size=args.output_size, skip=args.skip_con)
        network.save_state = False
        network.save_model('model', save_path)
    return network

if __name__ == "__main__":
    """The main method creates the recurrent network, trains it and carries out validation/testing """
    start_time = time.time()

    # If a load_config argument was provided, construct the file path to the config file
    if args.load_config:
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        configs = miscfuncs.json_load(args.load_config, args.config_location)
        for parameters in configs:
            args.__setattr__(parameters, configs[parameters])

    if args.model == 'SimpleRNN':
        model_name = args.model + args.device + '_' + args.unit_type + '_hs' + str(args.hidden_size) + '_pre_' + args.pre_filt
    if args.pre_filt == 'A-Weighting':
        with open('Configs/' + 'b_Awght_mk2.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            args.pre_filt = list(reader)
            args.pre_filt = args.pre_filt[0]
            for item in range(len(args.pre_filt)):
                args.pre_filt[item] = float(args.pre_filt[item])
    elif args.pre_filt == 'high_pass':
        args.pre_filt = [-0.85, 1]
    elif args.pre_filt == 'None':
        args.pre_filt = None

    # Generate name of directory where results will be saved
    save_path = os.path.join(args.save_location, args.device + '-' + args.load_config)

    # Check if an existing saved model exists, and load it, otherwise creates a new model
    network = init_model(save_path, args)

    # Check if a cuda device is available
    if not torch.cuda.is_available() or args.cuda == 0:
        print('cuda device not available/not selected')
        cuda = 0
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print('cuda device available')
        network = network.cuda()
        cuda = 1

    # Set up training optimiser + scheduler + loss fcns and training info tracker
    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
    loss_functions = training.LossWrapper(args.loss_fcns, args.pre_filt)
    train_track = training.TrainTrack()
    writer = SummaryWriter(os.path.join('runs2', model_name))

    # Load dataset
    dataset = dataset.DataSet(data_dir=args.data_location)

    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(os.path.join('train', args.file_name), 'train')

    dataset.create_subset('val')
    dataset.load_file(os.path.join('val', args.file_name), 'val')

    dataset.create_subset('test')
    dataset.load_file(os.path.join('test', args.file_name), 'test')


    # If training is restarting, this will ensure the previously elapsed training time is added to the total
    init_time = time.time() - start_time + train_track['total_time']*3600
    # Set network save_state flag to true, so when the save_model method is called the network weights are saved
    network.save_state = True
    patience_counter = 0

    # This is where training happens
    # the network records the last epoch number, so if training is restarted it will start at the correct epoch number
    for epoch in range(train_track['current_epoch'] + 1, args.epochs + 1):
        ep_st_time = time.time()

        # Run 1 epoch of training,
        epoch_loss = network.train_epoch(dataset.subsets['train'].data['input'][0],
                                         dataset.subsets['train'].data['target'][0],
                                         loss_functions, optimiser, args.batch_size, args.init_len, args.up_fr)

        # Run validation
        if epoch % args.validation_f == 0:
            val_ep_st_time = time.time()
            val_output, val_loss = network.process_data(dataset.subsets['val'].data['input'][0],
                                             dataset.subsets['val'].data['target'][0], loss_functions, args.val_chunk)
            scheduler.step(val_loss)
            if val_loss < train_track['best_val_loss']:
                patience_counter = 0
                network.save_model('model_best', save_path)
                write(os.path.join(save_path, "best_val_out.wav"),
                      dataset.subsets['test'].fs, val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_counter += 1
            train_track.val_epoch_update(val_loss.item(), val_ep_st_time, time.time())
            writer.add_scalar('Loss/val', train_track['validation_losses'][-1], epoch)

        print('current learning rate: ' + str(optimiser.param_groups[0]['lr']))
        train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
        # write loss to the tensorboard (just for recording purposes)
        writer.add_scalar('Loss/train', train_track['training_losses'][-1], epoch)
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'])
        network.save_model('model', save_path)
        miscfuncs.json_save(train_track, 'training_stats', save_path)

        if args.validation_p and patience_counter > args.validation_p:
            print('validation patience limit reached at epoch ' + str(epoch))
            break

    lossESR = training.ESRLoss()
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                     dataset.subsets['test'].data['target'][0], loss_functions, args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_final.wav"), dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 1)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 1)
    train_track['test_loss_final'] = test_loss.item()
    train_track['test_lossESR_final'] = test_loss_ESR.item()

    best_val_net = miscfuncs.json_load('model_best', save_path)
    network = networks.load_model(best_val_net)
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                     dataset.subsets['test'].data['target'][0], loss_functions, args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_bestv.wav"),
          dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 2)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 2)
    train_track['test_loss_best'] = test_loss.item()
    train_track['test_lossESR_best'] = test_loss_ESR.item()
    miscfuncs.json_save(train_track, 'training_stats', save_path)
    if cuda:
        with open(os.path.join(save_path, 'maxmemusage.txt'), 'w') as f:
            f.write(str(torch.cuda.max_memory_allocated()))

