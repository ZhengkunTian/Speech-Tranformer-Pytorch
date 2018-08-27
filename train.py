'''
This script handling the training process.
'''
import argparse
import math
import time
import torch
import configparser
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from DataLoader import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer, all_steps, interval=5):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for step, batch in enumerate(training_data):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:].to(DEVICE)

        # forward
        optimizer.zero_grad()
        pred = model(src, tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words.item()
        n_total_correct += n_correct.item()
        total_loss += loss.item()

        if step % interval == 0:
            print("[Training] step: %6d/%d, learning rate: %4.6f, CELoss: %8.5f " %
                  (optimizer.n_current_steps, all_steps, optimizer.current_lr, loss.item()))

        if step > 10:
            break

    return total_loss / n_total_words, n_total_correct / n_total_words


def eval_epoch(model, validation_data, crit, all_steps, interval=100):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for step, batch in enumerate(validation_data):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:].to(DEVICE)

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words.item()
        n_total_correct += n_correct.item()
        total_loss += loss.item()

        if step % interval == 0:
            print("[Validating] step: %6d/%d, CELoss: %8.5f " %
                  (step, all_steps, loss.item()))

        if step > 10:
            break

    return total_loss / n_total_words, n_total_correct / n_total_words


def train(model, training_data, validation_data, crit, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        log_test_file = opt.log + '.test.log'
        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,CELOSS,accuracy\n')
            log_vf.write('epoch,loss,CELOSS,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        all_train_steps = opt.epoch * \
            math.floor(training_data.n_insts / opt.batch_size)
        train_loss, train_accu = train_epoch(
            model, training_data, crit, optimizer, all_train_steps, interval=opt.print_interval)
        print('  - (Training)   CELOSS: {CELOSS: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  CELOSS=train_loss, accu=100 * train_accu,
                  elapse=(time.time() - start) / 60))

        start = time.time()
        all_dev_steps = opt.epoch * \
            math.floor(validation_data.n_insts / opt.batch_size)
        valid_loss, valid_accu = eval_epoch(
            model, validation_data, crit, all_dev_steps, interval=opt.print_interval)
        print('  - (Validation) CELOSS: {CELOSS: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  CELOSS=valid_loss, accu=100 * valid_accu,
                  elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + \
                    '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{CELOSS: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    CELOSS=train_loss, accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{CELOSS: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    CELOSS=valid_loss, accu=100 * valid_accu))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-context_width', type=int, default=1)
    parser.add_argument('-frame_rate', type=int, default=30)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-scheduled_sample_ratio', type=int, default=0.5)
    parser.add_argument('-log', default=None)
    parser.add_argument('-print_interval', type=int, default=5)
    parser.add_argument('-save_model', default='./exp')
    parser.add_argument('-save_mode', type=str,
                        choices=['all', 'best'], default='best')

    opt = parser.parse_args()

    cfg_path = './config/transformer.cfg'
    config = configparser.ConfigParser()
    config.read(cfg_path)

    #========= Preparing DataLoader =========#
    training_data = DataLoader(
        'train', config, batch_size=opt.batch_size, context_width=opt.context_width, frame_rate=opt.frame_rate)
    validation_data = DataLoader(
        'dev', config, batch_size=opt.batch_size, context_width=opt.context_width, frame_rate=opt.frame_rate)
    test_data = DataLoader(
        'test', config, batch_size=opt.batch_size, context_width=opt.context_width, frame_rate=opt.frame_rate)

    #========= Preparing Model =========#

    print(opt)

    input_dim = training_data.features_dim
    output_dim = training_data.vocab_size
    n_inputs_max_seq = max(
        training_data.inputs_max_seq_lengths,
        validation_data.inputs_max_seq_lengths,
        test_data.inputs_max_seq_lengths)
    n_outputs_max_seq = max(
        training_data.outputs_max_seq_lengths,
        validation_data.outputs_max_seq_lengths,
        test_data.outputs_max_seq_lengths)

    transformer = Transformer(
        input_dim,
        output_dim,
        n_inputs_max_seq,
        n_outputs_max_seq,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=DEVICE,
        scheduled_sample_ratio=opt.scheduled_sample_ratio)

    # print(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    def get_criterion(output_dim):
        ''' With PAD token zero weight '''
        weight = torch.ones(output_dim)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight)

    crit = get_criterion(training_data.vocab_size)

    transformer = transformer.to(DEVICE)
    crit = crit.to(DEVICE)

    train(transformer, training_data, validation_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
