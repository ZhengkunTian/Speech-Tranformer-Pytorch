'''
This script handling the multi-gpu training process.
'''
import os
import argparse
import sys
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from Dataset import AudioDateset
from transformer.Evaluator import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Utils import AttrDict, init_logger, count_parameters, save_model
from tensorboardX import SummaryWriter


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/rnnt.yaml')
    parser.add_argument('-load_model', type=str, default=None)

    parser.add_argument('-num_wokers', type=int, default=0,
                    help='how many subprocesses to use for data loading. '
                    '0 means that the data will be loaded in the main process')
    parser.add_argument('-log', type=str, default='decode.log')
    opt = parser.parse_args()

    if opt.load_model is None:
        raise AssertionError('Please load pretrained model')

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))

    exp_name = config.data.name
    if not os.path.isdir(exp_name):
        os.mkdir(exp_name)
    logger = init_logger(exp_name + '/' + opt.log)

    #========= Build DataLoader =========#
    test_dataset = AudioDateset(config.data, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.test.batch_size, shuffle=False)

    #========= Build A Model Or Load Pre-trained Model=========#
    model = Transformer(config.model)

    if torch.cuda.is_available():
        model.cuda()

    # load pretrain model
    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])
        logger.info('Load pretrainded Model and previous Optimizer!')


    for step, (inputs, targets, input_lengths, target_lengths, groud_truth) in enumerate(test_loader):
        
    
    logger.info('Traing Process Finished')


if __name__ == '__main__':
    main()