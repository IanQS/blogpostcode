"""
A generic data source backend for non-RL formats. Can be easily configured and abstracted upon via use of a yaml
    - prevents us from overfitting to a data source when scaling

Author: Ian Q.

Notes:
    
"""

import numpy as np


class Pipeline(object):
    def __init__(self, src: dict):
        self.__configuration_check(src)
        print('Pipeline using: {}'.format(src['backend']))
        for k, v in src:
            setattr(self, k, v)

    def provide(self):
        if not self.processed:
            return self.processor(self.setup_connection)
        else:
            return self.setup_connection

    def __configuration_check(self, src):
        self.backend = None
        self.processed = None
        self.processor = None
        self.setup_connection = None
        assert src.get('backend', None) is not None
        assert src.get('processed', None) is not None
        assert src.get('setup_connection', None) is not None

        if src.get('processed') is False:
            assert src.get('processor', None) is not None


if __name__ == '__main__':
    from DataPipelines.common_backends import np_backend
    from DataPipelines.common_preprocessors import log_normalizations

    pipeline_config = {
        'backend': 'Folder',
        'processed': True,
        'processor': log_normalizations,
        'connect_to_backend': np_backend
    }
