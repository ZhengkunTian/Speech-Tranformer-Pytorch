import torch
import DataLoader
import configparser

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = './config/transformer.cfg'
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    loader = DataLoader.DataLoader(
        'dev', cfg, device, context_width=1, frame_rate=30)
    # for i, (x, y) in enumerate(loader):
    # print(i)
    # x, y = loader.next()
    # print(x[0].shape)
    print(loader.features_dim)
    print(loader.inputs_max_seq_lengths)
    print(loader.outputs_max_seq_lengths)
