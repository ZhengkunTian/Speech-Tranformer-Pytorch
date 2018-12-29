'''
This script handling the multi-gpu training process.
'''
import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from Dataset import AudioDateset
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Utils import AttrDict, init_logger, count_parameters, save_model, init_parameters
from tensorboardX import SummaryWriter


def train(epoch, model, crit, optimizer, train_loader, logger, visualizer, config):
    global global_step
    model.train()
    start_epoch = time.clock()
    train_loss = 0
    batch_step = len(train_loader)
    for step, (inputs, targets, input_lengths, target_lengths, groud_truth) in enumerate(train_loader):
        global_step += 1
        if config.training.use_gpu:
            inputs, targets, groud_truth = inputs.cuda(), targets.cuda(), groud_truth.cuda()
            input_lengths, target_lengths = input_lengths.cuda(), target_lengths.cuda()

        max_inputs_length = input_lengths.max().item()
        max_targets_length = target_lengths.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]
        groud_truth = groud_truth[:, :max_targets_length]

        optimizer.zero_grad()
        start = time.clock()
        logits, _ = model(inputs, input_lengths, targets, target_lengths)
        loss = crit(logits.contiguous().view(-1, config.model.vocab_size), groud_truth.contiguous().view(-1))

        train_loss += loss.item()

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step(global_step)

        if step % config.training.show_interval == 0:
            end  = time.clock()
            process = 100.0 * step / len(train_loader)
            logger.info('-Training-Epoch:%d, Step:%d / %d (%.4f), Learning Rate:%.6f, Grad Norm:%.5f, RNNTLoss:%.5f, Run Time:%.3f' %
                        (epoch, step, batch_step, process, optimizer.lr, grad_norm, loss.item(), end-start))

        if visualizer is not None:
            visualizer.add_scalar('train_loss', loss.item(), global_step)

    end_epoch = time.clock()
    logger.info('-Training-Epoch:%d, AverageRNNTLoss:%.5f, Run Time:%.5f' % (epoch, train_loss / (step+1), end_epoch-start_epoch))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/rnnt.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-num_workers', type=int, default=0,
                    help='how many subprocesses to use for data loading. '
                    '0 means that the data will be loaded in the main process')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))

    exp_name = config.data.name
    if not os.path.isdir(exp_name):
        os.mkdir(exp_name)
    logger = init_logger(exp_name + '/' + opt.log)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        raise NotImplementedError

    #========= Build DataLoader =========#
    train_dataset = AudioDateset(config.data, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.train.batch_size, shuffle=True, num_workers=opt.num_workers)

    assert train_dataset.vocab_size == config.model.vocab_size

    #========= Build A Model Or Load Pre-trained Model=========#
    model = Transformer(config.model)

    n_params, enc_params, dec_params = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in encoder: %d' % enc_params)
    logger.info('# the number of parameters in decoder: %d' % dec_params)

    if torch.cuda.is_available():
        model.cuda()

    global global_step
    global_step = 0
    # define an optimizer
    optimizer = ScheduledOptim(model, config.model.d_model, config.optimizer)

    # load pretrain model
    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('Load pretrainded Model and previous Optimizer!')
    else:
        init_parameters(model)
        logger.info('Initialized all parameters!')

    # define loss function
    crit = nn.CrossEntropyLoss(ignore_index=0)

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(exp_name + '/log')
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(config.training.epoches):
        train(epoch, model, crit, optimizer, train_loader, logger, visualizer, config)
        save_model(epoch, model, optimizer, config, logger)
    
    logger.info('Traing Process Finished')


if __name__ == '__main__':
    main()