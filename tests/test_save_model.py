import torch
import yaml


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

    def get_dict(self):
        return


configfile = open('./config/random.yaml')
config = AttrDict(yaml.load(configfile))
a = dict(config)

model = torch.nn.Linear(2, 3)

checkpoint = {
    'setting': a,
    'model': model.state_dict()
}
torch.save(checkpoint, 'model.chkpt')
