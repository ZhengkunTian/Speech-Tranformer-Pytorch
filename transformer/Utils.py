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


def padding_info_mask(seq_q_length, seq_k_length):
    ''' Indicate the padding-related part to mask '''
    assert seq_q_length.dim() == 1 and seq_k_length.dim() == 1
    batch_size = seq_k_length.size(0)
    len_q = seq_q_length.max().item()
    len_k = seq_k_length.max().item()
    mask_mat = []
    for i in range(batch_size):
        max_len = len_k
        length = seq_k_length[i].item()
        mask_mat.append(np.column_stack([np.zeros([1, length]), np.ones([1, max_len-length])]))
    mask_mat = np.row_stack(mask_mat).astype('uint8')
    pad_attn_mask = torch.from_numpy(mask_mat).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # bxsqxsk
    if seq_q_length.is_cuda:
        return pad_attn_mask.cuda()
    return pad_attn_mask


def feature_info_mask(seq_length):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq_length.dim() == 1
    batch_size = seq_length.size(0)
    max_len = seq_length.max().item()
    attn_shape = (batch_size, max_len, max_len)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq_length.is_cuda:
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
    for _, param in model.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.xavier_normal_(param)


def get_saved_model_name(config):
    name_list = [config.data.name]
    name_list.append(config.model.type)
    name_list.append('enc_%dly' % (config.model.num_enc_layers))
    name_list.append('dec_%dly' % (config.model.num_dec_layers))
    name_list.append('%dhd_%ddm'% (config.model.n_head, config.model.d_model))
    return '_'.join(name_list)


def save_model(epoch, model, optimizer, config, logger):
    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    model_name = get_saved_model_name(config)
    model_path = config.data.name + '/' + model_name + '.epoch%s.chkpt' % str(epoch)
    torch.save(checkpoint, model_path)
    logger.info('Saved the model!')


if __name__ == '__main__':
    inputs = torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    length = torch.LongTensor([5, 3])
    mask = padding_info_mask(length, length)
    print(mask)
    # mask2 = feature_info_mask(inputs)
    # print(mask2)
    length = torch.LongTensor([5, 3])
    mask = feature_info_mask(length)
    print(mask)