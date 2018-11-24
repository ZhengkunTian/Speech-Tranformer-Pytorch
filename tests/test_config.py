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


configfile = open('./config/character.yaml')
config = AttrDict(yaml.load(configfile))

print(type(config))
print(config.units_type)
print(config.training.gpu_ids)

data = config.__getattr__('data')
print(data)
