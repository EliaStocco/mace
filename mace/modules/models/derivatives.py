from typing import Tuple, Dict, List, Optional
import torch
from e3nn.util.jit import compile_mode
from .general import BaseDipoleClass, MACEBaseModel
from .dipole import AtomicDipolesMACElia

def compute_dielectric_gradients(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    d_dielectric_dr = []
    for i in range(dielectric.shape[-1]):
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones((dielectric.shape[0], 1)).to(dielectric.device)
        ]
        gradient = torch.autograd.grad(
            outputs=[dielectric[:, i].unsqueeze(-1)],  # [n_graphs, 3], [n_graphs, 9]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=True,  # Make sure the graph is not destroyed during training
            create_graph=True,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )
        d_dielectric_dr.append(gradient[0])
    d_dielectric_dr = torch.stack(d_dielectric_dr, dim=1)
    if gradient[0] is None:
        return torch.zeros((positions.shape[0], dielectric.shape[-1], 3))
    return d_dielectric_dr

def add_natoms(info):
    shape = info[1]
    if type(shape) == int:
        return (info[0], (shape,"natoms",3,))
    else:
        return (info[0], ("natoms",3,) + shape)

def get_d_prop_dR(props,basecls:MACEBaseModel,rename):
    # der = [None]*len(props)
    der = {}
    ip = basecls.implemented_properties
    for prop in props:
        if prop not in ip:
            raise ValueError("'{:s}' is not an implemented property of the parent class {}.".format(prop,basecls))
        name ="{:s}_dR".format(prop)
        name = name if name not in rename else rename[name]
        der[name] = add_natoms(ip[prop])
    return der


def add_dR( basecls:MACEBaseModel,\
            diff_props:list=[],\
            name:str=None,\
            rename:dict={})->MACEBaseModel:

    @compile_mode("script")
    class class_with_dR(basecls):
        """Inherits from a class that predicts dipoles, and add the evaluation of their spatial derivaties to 'forward'."""

        implemented_properties = {
            **basecls.implemented_properties,
            **get_d_prop_dR(diff_props,basecls,rename)
        }
        to_diff_props = diff_props
        rename_dR = rename

        @classmethod
        def from_parent(cls, obj):
            """Cast a parent class object to a child class object without copying all the attributes."""
            obj.__class__ = class_with_dR
            obj.__name__ = name if name is not None else "{:s}_dR".format(basecls.__name__)
            return obj

        def forward(
            self,
            data: Dict[str, torch.Tensor],
            training: bool = False,  # pylint: disable=W0613
            compute_force: bool = False,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
        ) -> Dict[str, Optional[torch.Tensor]]:

            if training:
                raise ValueError("`{:s}` can be used only in `eval` mode.".format(basecls))

            # data.keys():
            # dict_keys(['batch', 'cell', 'charges', 'dipole', 'edge_index',
            # 'energy', 'energy_weight', 'forces', 'forces_weight', 'node_attrs',
            # 'positions', 'ptr', 'shifts', 'stress', 'stress_weight', 'unit_shifts',
            # 'virials', 'virials_weight', 'weight'])

            # compute the dipoles
            output: dict = super().forward(
                data,
                training,
                compute_force,
                compute_virials,
                compute_stress,
                compute_displacement,
            )

            for prop in self.to_diff_props:
                array = compute_dielectric_gradients(
                    dielectric=output[prop],
                    positions=data["positions"],
                )
                name = "{:s}_dR".format(prop)
                name = name if name not in self.rename_dR else self.rename_dR[name]

                # # Determine the number of axes: (atom, positions coord, output coord)
                # num_axes = tmp.dim()

                # # Permute the axis order
                # permuted_order = list(range(num_axes))
                # permuted_order = [permuted_order[-1]] + permuted_order[:-1]  # Move last axis to the front
                # permuted_tensor = tmp.permute(permuted_order)

                # (output coord, atom, positions coord)
                output[name] = array

            return output
        
    return class_with_dR



AtomicDipolesMACElia_BEC = add_dR(basecls=AtomicDipolesMACElia,
                                  diff_props=["dipole"],
                                  rename={"dipole_dR":"BEC"})
