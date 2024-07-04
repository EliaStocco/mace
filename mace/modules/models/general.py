import torch
import importlib
from ase.outputs import _defineprop, all_outputs
from typing import TypeVar, Type, Union
T = TypeVar('T', bound='MACEBaseModel')

class MACEBaseModel(torch.nn.Module):
    implemented_properties:dict

    def __init__(self:T)->None:
        """Initialize a `MACEBaseModel` object and add its implemented properties to ASE `all_outputs`"""
        super().__init__()
        self.set_prop()

    def set_prop(self:T)->None:
        for name,par in self.implemented_properties.items():
            if par is not None:
                if name not in all_outputs:
                    _defineprop(name,*par)

    @classmethod
    def from_parent(cls:Type[T], obj:T)->T:
        """Cast a parent class object to a child class object without copying all the attributes."""
        return obj

class BaseDipoleClass(MACEBaseModel):
    implemented_properties = {
        "dipole" : (float, 3),
    }

class BaseEnergyClass(MACEBaseModel):
    implemented_properties = {
        "energy"      : None,
        "free_energy" : None,
        "node_energy" : None,
        "forces"      : None,
        "stress"      : None
    }

def get_model(model_path:str,model_type:str,device:Union[str,torch.device])->MACEBaseModel:
    # Load model
    # print("reading model from file '{:s}'".format(model_path))
    model:MACEBaseModel = torch.load(f=model_path, map_location=device)
    # Change model type
    # print("model type: '{:s}'".format(model_type))

    from mace.modules import models
    child_class:MACEBaseModel = getattr(models, model_type)
    model = child_class.from_parent(model)
    model.set_prop()
    return model
    
    # if str(model_type).endswith("_BEC"):
    #     from mace.modules import models
    #     from .derivatives import addBEC2class
    #     parent_class = str(model_type).split('_BEC')[0]
    #     parent_class = getattr(models, parent_class)
    #     child_class = addBEC2class(parent_class)
    #     model = child_class.from_parent(model)

    # return model

def import_class(module_name:str, class_name:str)->Type[T]:
    try:
        module = importlib.import_module(module_name)
        class_instance = getattr(module, class_name)
        return class_instance
    except ImportError:
        print("Module '{}' not found.".format(module_name))
    except AttributeError:
        print("Class '{}' not found in module '{}'.".format(class_name, module_name))





