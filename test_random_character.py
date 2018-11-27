'''
This script handling the training process.
'''
import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from tests.random_character_loader import random_character_loader
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Utils import AttrDict, init_logger, count_parameters
from tensorboardX import SummaryWriter


def train(config, model, training_data, validation_data, crit, optimizer, logger, visualizer=None):
    dev_step = 0
    for epoch in range(config.training.epoches):
        model.train()
        for _, data_dict in enumerate(training_data):
            inputs = data_dict['inputs']
            inputs_pos = data_dict['inputs_pos']
            targets = data_dict['sos_targets']
            targets_pos = data_dict['sos_targets_pos']
            eos_targets = data_dict['targets_eos']
            optimizer.zero_grad()
            logits, _ = model(inputs, inputs_pos, targets, targets_pos)
            loss = crit(logits.view(-1, logits.size(2)), eos_targets.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            if visualizer is not None:
                visualizer.add_scalar('model/train_loss', loss.item(), optimizer.current_step)
                visualizer.add_scalar('model/learning_rate', optimizer.lr, optimizer.current_step)

            if optimizer.current_step % config.training.show_interval == 0:
                logger.info('-Training-Epoch:%d, Global Step:%d, Learning Rate:%.6f, CrossEntropyLoss:%.5f' %
                            (epoch, optimizer.current_step, optimizer.lr, loss.item()))

        if config.training.dev_on_training:
            model.eval()
            total_loss = 0
            for step, data_dict in enumerate(validation_data):
                dev_step += 1
                inputs = data_dict['inputs']
                inputs_pos = data_dict['inputs_pos']
                targets = data_dict['sos_targets']
                targets_pos = data_dict['sos_targets_pos']
                eos_targets = data_dict['targets_eos']
                logits, _ = model(inputs, inputs_pos, targets, targets_pos)
                loss = crit(logits.view(-1, logits.size(2)), eos_targets.contiguous().view(-1))
                total_loss += loss.item()

                if visualizer is not None:
                    visualizer.add_scalar('model/validation_loss', loss.item(), dev_step)

                if step % config.training.show_interval == 0:
                    logger.info('-Validation-Step:%4d, CrossEntropyLoss:%.5f' % (step, loss.item()))

            logger.info('-Validation-Epoch:%4d, AverageCrossEntropyLoss:%.5f' %
                        (epoch, total_loss/step))
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
    parser.add_argument('-config', type=str, default='config/random.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-log', type=str, default='./exp/train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))

    if not os.path.isdir('exp'):
        os.mkdir('exp')
    logger = init_logger(opt.log)

    if config.training.use_gpu:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    #========= Build DataLoader =========#
    if config.use_gpu:
        device_ids = int(config.training.gpu_ids)
        device = torch.device('cuda:%d' % device_ids)
    else:
        device = torch.device('cpu')
    training_data = random_character_loader(
        batch_size=config.data.batch_size,
        vocab_size=config.model.vocab_size,
        length=1000,
        device=device)
    validation_data = random_character_loader(
        batch_size=config.data.batch_size,
        vocab_size=config.model.vocab_size,
        length=100,
        device=device)

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

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)

    optimizer = ScheduledOptim(
        torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        config.model.d_model, config.optimizer.n_warmup_steps)
    logger.info('Created a optimizer.')

    crit = nn.CrossEntropyLoss()
    logger.info('Created cross entropy loss function')

    if config.training.use_gpu:
        model.cuda(device_ids)
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
