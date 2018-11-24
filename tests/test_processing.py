from __future__ import absolute_import, division, print_function
import os
from six.moves import configparser
from processing import ark, prepare_data, feature_reader, batchdispenser, \
    target_coder, target_normalizers, score
from shutil import copyfile

config = configparser.ConfigParser()

#config_path = 'config/config_TIMIT.cfg'
#config_path = 'config/config_TIMIT_listener.cfg'
config_path = 'conf/config_TIMIT_las.cfg'

# config.read()
config.read(config_path)
current_dir = os.getcwd()

# compute the features of the training set for DNN training
# if they are different then the GMM features.

feat_cfg = dict(config.items('dnn-features'))

print('------- computing DNN training features ----------')
prepare_data.prepare_data(
    config.get('directories', 'dev_data'),
    config.get('directories', 'dev_features') + '/' + feat_cfg['name'],
    feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

print('------- computing cmvn stats ----------')
prepare_data.compute_cmvn(config.get(
    'directories', 'dev_features') + '/' + feat_cfg['name'])
