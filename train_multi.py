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
import horovod.torch as hvd
import torch.utils.data
import torch.utils.data.distributed
from Dataset import AudioDateset
from transformer.Models import Transformer
from transformer.Utils import AttrDict, init_logger, count_parameters, save_model, init_parameters
from tensorboardX import SummaryWriter


hvd.init()

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        val = torch.tensor(val)
        self.sum += hvd.allreduce(val, name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def train(epoch, model, crit, optimizer, train_loader, train_sampler, logger, visualizer, config):
    global global_step
    model.train()
    train_sampler.set_epoch(epoch)

    start_epoch = time.clock()
    train_loss = Metric('train_loss')
    batch_step = len(train_loader)
    for step, (inputs, targets, input_lengths, target_lengths, groud_truth) in enumerate(train_loader):
        global_step += 1
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

        train_loss.update(loss.item())

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        lr = update_lr(config, optimizer, global_step)
        optimizer.step()

        if step % config.training.show_interval == 0 and hvd.rank() == 0:
            end  = time.clock()
            process = 100.0 * step / len(train_loader)
            logger.info('-Training-Epoch:%d, Step:%d / %d (%.4f), Learning Rate:%.6f, Grad Norm:%.5f, RNNTLoss:%.5f, Run Time:%.3f' %
                        (epoch, step, batch_step, process, lr, grad_norm, loss.item(), end-start))

        if visualizer is not None:
            visualizer.add_scalar('train_loss', loss.item(), global_step)

    if hvd.rank() == 0:
        end_epoch = time.clock()
        logger.info('-Training-Epoch:%d, AverageRNNTLoss:%.5f, Run Time:%.5f' % (epoch, train_loss.avg, end_epoch-start_epoch))


def update_lr(config, optimizer, step):
    d_model = config.model.d_model
    lr = np.power(d_model, -0.5) * np.min([
        np.power(step, -0.5),
        np.power(config.optimizer.n_warmup_steps, -1.5) * step])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/rnnt.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-fp16_allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
    parser.add_argument('-batches_per_allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
    parser.add_argument('-num_wokers', type=int, default=0,
                    help='how many subprocesses to use for data loading. '
                    '0 means that the data will be loaded in the main process')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))

    global global_step
    global_step = 0

    if hvd.rank() == 0: 
        exp_name = config.data.name
        if not os.path.isdir(exp_name):
            os.mkdir(exp_name)
        logger = init_logger(exp_name + '/' + opt.log)
    else:
        logger = None

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        raise NotImplementedError

    #========= Build DataLoader =========#
    train_dataset = AudioDateset(config.data, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.train.batch_size, sampler=train_sampler)

    assert train_dataset.vocab_size == config.model.vocab_size

    #========= Build A Model Or Load Pre-trained Model=========#
    model =  Transformer(config.model)

    if hvd.rank() == 0:
        n_params, enc_params, dec_params = count_parameters(model)
        logger.info('# the number of parameters in the whole model: %d' % n_params)
        logger.info('# the number of parameters in encoder: %d' % enc_params)
        logger.info('# the number of parameters in decoder: %d' % dec_params)


    model.cuda()

    # define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if opt.fp16_allreduce else hvd.Compression.none
    
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    # load pretrain model
    if opt.load_model is not None and hvd.rank() == 0:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('Load pretrainded Model and previous Optimizer!')
    elif hvd.rank() == 0:
        init_parameters(model)
        logger.info('Initialized all parameters!')

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # define loss function
    crit = nn.CrossEntropyLoss(ignore_index=0)

    # create a visualizer
    if config.training.visualization and hvd.rank() == 0:
        visualizer = SummaryWriter(exp_name + '/log')
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(config.training.epoches):
        train(epoch, model, crit, optimizer, train_loader, train_sampler, logger, visualizer, config)

        if hvd.rank() == 0:
            save_model(epoch, model, optimizer, config, logger)
    
    if hvd.rank() == 0:
        logger.info('Traing Process Finished')

if __name__ == '__main__':
    main()