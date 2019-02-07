from Sequences.translation.Models.transformer import Transformer


_models_ = {
    'Transformer': Transformer,
}

def model_builder(config: dict):
    model_name = config['name']
    config.pop('name')
    architecture = _models_.get(model_name)
    model = architecture(**config)
    return model