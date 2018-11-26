import numpy as np
import editdistance
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def padding_info_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    if seq_q.is_cuda:
        return pad_attn_mask.cuda()
    return pad_attn_mask


def feature_info_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        return subsequent_mask.cuda()
    return subsequent_mask


def learn_rate(d_model, n_warmup_steps, current_step):
    lr = np.power(d_model, -0.5) * np.min([
        np.power(current_step, -0.5),
        np.power(n_warmup_steps, -1.5) * current_step])
    return lr


def show_learning_rate():
    n_warmup_steps = 40000
    d_model = 512
    steps = list(range(500000))
    lr = [learn_rate(d_model, n_warmup_steps, i) for i in steps]
    plt.plot(steps, lr)
    plt.show()


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model):
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            print('Initialize the parameters: %s' % name)
            torch.nn.init.xavier_normal_(param)
    print('Parameter initialization completed')


def load_single_gpu_model(model, state_dict):
    model.load_state_dict(state_dict)
    return model


def load_multi_gpu_model(model, state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]  # remove 'module.'
        new_state_dict[namekey] = v
        # load params
    model.load_state_dict(new_state_dict)
    return model
