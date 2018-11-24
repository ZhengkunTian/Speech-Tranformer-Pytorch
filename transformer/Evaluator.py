''' Translate input text with trained model. '''

import torch
import argparse
import configparser
from transformer.Decode import Decode
from DataLoader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")

    opt = parser.parse_args()

    # load config
    cfg_path = './config/transformer.cfg'
    config = configparser.ConfigParser()
    config.read(cfg_path)

    # Prepare DataLoader
    test_data = DataLoader(
        'test', config, batch_size=opt.batch_size, context_width=opt.context_width, frame_rate=opt.frame_rate, return_target=False)

    opt.input_dim = test_data.features_dim
    opt.output_dim = test_data.vocab_size
    opt.n_inputs_max_seq = test_data.inputs_max_seq_lengths
    opt.n_outputs_max_seq = test_data.outputs_max_seq_lengths

    decoder = Decode(opt, device)
    decoder.model.eval()

    with open(opt.output, 'w') as f:
        for step, batch in enumerate(test_data):
            all_hyp, all_scores = decoder.decode_batch(batch)
            idx_in_batch = 0
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    idx_seq = [idx.item() for idx in idx_seq]
                    pred_line = test_data.target_coder.decode(idx_seq)
                    f.write(pred_line + '\n')
                    print('Index: %d  Decode Sequence: %s' %
                          (step + idx_in_batch, pred_line))
                idx_in_batch += 1

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
