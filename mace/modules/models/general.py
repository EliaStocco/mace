import torch
from ase.outputs import _defineprop

class MACEBaseModel(torch.nn.Module):
    implemented_properties:dict

    def __init__(self):
        """Initialize a `MACEBaseModel` object and add its implemented properties to ASE `all_outputs`"""
        super().__init__()
        self.set_prop()

    def set_prop(self):
        for name,par in self.implemented_properties.items():
            if par is not None:
                _defineprop(name,*par)

    @classmethod
    def from_parent(cls, obj):
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

def get_model(model_path,model_type,device)->MACEBaseModel:
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




