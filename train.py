'''
This script handling the training process.
'''
import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from DataLoader import build_data_loader
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Utils import AttrDict, init_logger, count_parameters
from tensorboardX import SummaryWriter


def train(config, model, training_data, validation_data, crit, optimizer, logger, visualizer=None):
    for epoch in range(config.training.epoches):
        model.train()
        for _, data_dict in enumerate(training_data):
            inputs = data_dict['inputs']
            inputs_pos = data_dict['inputs_pos']
            targets = data_dict['targets']
            targets_pos = data_dict['targets_pos']
            optimizer.zero_grad()
            logits, _ = model(inputs, inputs_pos, targets[:, :-1], targets_pos[:, :-1])
            loss = crit(logits.view(-1, logits.size(2)), targets[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            if visualizer is not None:
                visualizer.add_scalar('model/train_loss', loss.item(), optimizer.current_step)
                visualizer.add_scalar('model/learning_rate', optimizer.lr, optimizer.current_step)

            if optimizer.current_step % config.training.show_interval == 0:
                logger.info('-Training-Epoch:%d, Global Step:%d, Learning Rate:%.6f, CrossEntropyLoss:%.5f' %
                            (epoch, optimizer.current_step, optimizer.lr, loss.item()))

        model.eval()
        for step, data_dict in enumerate(validation_data):
            inputs = data_dict['inputs']
            inputs_pos = data_dict['inputs']
            targets = data_dict['targets']
            targets_pos = data_dict['targets_pos']
            logits, _ = model(inputs, inputs_pos, targets[:, :-1], targets_pos[:, :-1])
            loss = crit(logits.view(-1, logits.size(2)), targets[:, 1:].contiguous().view(-1))

            if visualizer is not None:
                visualizer.add_scalar('model/validation_loss', loss.item(), optimizer.current_step)

            if step % cofig.training.show_interval == 0:
                logger.info('-Validation-Step:%4d, CrossEntropyLoss:%.5f' % (step, loss.item()))

        # save model
        model_state_dict = model.state_dict()
        checkpoint = {
            'settings': config.model,
            'model': model_state_dict,
            'epoch': epoch,
            'global_step': optimizer.current_step
        }
        model_name = config.training.save_model + '.epoch%s.chkpt' % str(epoch)
        torch.save(checkpoint, model_name)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default=None)
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-log', type=str, default='./exp/train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))

    if not os.path.isdir():
        os.mkdir('exp')
    logger = init_logger(opt.log)

    #========= Build DataLoader =========#
    training_data = build_data_loader(config, 'train')
    validation_data = build_data_loader(config, 'dev')

    #========= Build A Model Or Load Pre-trained Model=========#
    if opt.load_model:
        checkpoint = torch.load(opt.load_model)
        model_config = checkpoint['settings']
        model = Transformer(model_config)

        model.load_state_dict(checkpoint['model'])
        logger.info('Loaded model from %s' % opt.load_model)

    else:
        model = Transformer(config.model)

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in encoder: %d' % enc)
    logger.info('# the number of parameters in decoder: %d' % dec)
    logger.info('# the number of parameters in the whole model: %d' % n_params)

    optimizer = ScheduledOptim(
        optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        config.model.d_model, config.optimizer.n_warmup_steps)
    logger.info('Created a optimizer.')

    crit = nn.CrossEntropyLoss()
    logger.info('Created cross entropy loss function')

    if config.training.use_gpu:
        model.cuda()
        logger.info('Loaded the model to the GPU')

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter()
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    train(config, model, training_data, validation_data, crit, optimizer, logger, visualizer)


if __name__ == '__main__':
    main()
