"""This package contains modules related to objective functions, optimizations, and network architectures.
"""

import importlib
from i2iTranslation.models.base_model import BaseModel

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = '.models.' + model_name
    modellib = importlib.import_module(model_filename, package='i2iTranslation')
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(args, device):
    """Create a model given the option.
    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    """
    model = find_model_using_name(args['train.model.model_name'])
    instance = model(args, device)
    print("model [%s] was created" % type(instance).__name__)
    return instance