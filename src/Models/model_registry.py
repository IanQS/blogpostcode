"""
Import all models here to "register" them, so that they can be dynamically imported within
the main loader file which can be configured via yaml

Author: Ian Q.

Notes:
    Implemented for interview with AssemblyAI
"""


registered_models = {
    'None': None
}



def verify_status():
    from abc import ABCMeta
    err_msg = "{} does not fit the spec for a registered model. Must inherit from base_model"

    for k, v in registered_models.items():
        assert isinstance(v, ABCMeta), err_msg.format(k)

verify_status()
