"""
common_backends
    - simple example of a data source.
    - another example would be establishing a connection to S3 folders and starting multiprocessing that downloads

Author: Ian Q.

Notes:
"""

from os import walk
import numpy as np



def np_backend(src: str):
    f = []
    path = src
    for (dirpath, filenames) in walk(path):
        for f in filenames:
            data = np.load(dirpath + '/{}'.format(dirpath + '/' + f))
            yield data
            data.close()
