from .dipole import *
from .energy import *
from .energy_dipole import *
from .derivatives import *

from .general import get_model, import_class

# def get_classes_with_properties(module_name, file_path):
#     package_path = os.path.dirname(os.path.abspath(__file__))
#     module_path = os.path.join(package_path, file_path)
#     spec = importlib.util.spec_from_file_location(module_name, module_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
    
#     filtered_classes = []
    
#     for name, obj in inspect.getmembers(module, inspect.isclass):
#         if issubclass(obj, torch.nn.Module) and hasattr(obj, "implemented_properties"):
#             filtered_classes.append(obj)
    
#     return filtered_classes

# module_name = "mace.modules.models"
# classes = []

# for file in ["energy.py", "dipole.py", "energy_dipole.py"]:
#     new_classes = get_classes_with_properties(module_name, file)
#     classes.extend(new_classes)

# energy_models = [cls for cls in classes if "energy" in getattr(cls, "implemented_properties", [])]
# dipole_models = [cls for cls in classes if "dipole" in getattr(cls, "implemented_properties", [])]

energy_models = ["MACE","EnergyDipolesMACE"]
dipole_models = ["AtomicDipolesMACE","AtomicDipolesMACElia","AtomicDipolesMACE_MTP","DipolesPointCharges"]