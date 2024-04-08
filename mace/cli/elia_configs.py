###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
from ase import Atoms
import numpy as np
import torch
from typing import List, Dict
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
# from mace.modules.models import AtomicDipolesMACE, AtomicDipolesBECMACE
from mace.modules import models
from mace.modules.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--model_type", help="model type (default: 'AtomicDipolesMACE')", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default="charges",
    )
    return parser.parse_args()

def make_dataloader(atoms_list:List[Atoms],
                    model:torch.nn.Module,
                    batch_size:int,
                    charges_key:str)->torch_geometric.dataloader.DataLoader:
    configs = [data.config_from_atoms(atoms,charges_key=charges_key) for atoms in atoms_list]

    # test
    # atoms = atoms_list[0]
    # atoms.set_calculator(calculator)
    # dipole = atoms.get_dipole_moment()

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return data_loader

def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    print("reading model from file '{:s}'".format(args.model))
    # model:torch.nn.Module = torch.load(f=args.model, map_location=args.device)
    model = get_model(  model_path=args.model,
                        model_type=args.model_type,
                        device=args.device  )
    # Change model type
    # print("model type: '{:s}'".format(args.model_type))

    # if str(args.model_type).endswith("_BEC"):
    #     parent_class = str(args.model_type).split('_BEC')[0]
    #     parent_class = getattr(models, parent_class)
    #     child_class = addBEC2class(parent_class)
    #     model = child_class.from_parent(model)

    # if args.model_type == "AtomicDipolesMACE":
    #     pass
    # elif args.model_type == "AtomicDipolesBECMACE": 
    #     model = AtomicDipolesBECMACE.from_parent(model)
    # else:
    #     raise ValueError("`model_type` can be only [`AtomicDipolesMACE`,`AtomicDipolesBECMACE`]")

    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems
    for param in model.parameters():
        param.requires_grad = False

    # calculator = MACEliaBECCalculator(model_type="MACEliaBECCalculator",model_paths=args.model,device=args.device)
    
    # Load data and prepare input
    print("reading atomic structure from file '{:s}'".format(args.configs))
    atoms_list = ase.io.read(args.configs, index=":")

    # create dataloader
    data_loader:torch_geometric.dataloader.DataLoader = make_dataloader(atoms_list,model,args.batch_size,charges_key=args.charges_key)

    # Collect data
    all_lists = {}

    # whereto = {
    #     "dipole" : "info"  ,
    #     "becx"   : "arrays",
    #     "becy"   : "arrays",
    #     "becz"   : "arrays",
    # }
    whereto = {}


    for batch in data_loader:
        batch = batch.to(device)
        output:dict = model(batch.to_dict(), compute_stress=args.compute_stress)

        for k in output.keys():
            if k not in model.implemented_properties:
                print("warning: {:s} not in `model.implemented_properties`".format(k))
            # if 'natoms' in model.implemented_properties[k][1]:
            #     # arrays
            #     pass
            # elif isinstance(model.implemented_properties[k][1],int) or len(model.implemented_properties[k][1]) == 1 :
            #     # info
            # else:
            #     raise ValueError("coding error")

            # for k in whereto.keys():
            #     if k in output:
            else:
                if k not in all_lists:
                    all_lists[k] = [torch_tools.to_numpy(output[k])]
                else:
                    all_lists[k].append(torch_tools.to_numpy(output[k]))

    data:Dict[str,np.ndarray] = {}
    for k in all_lists.keys():
        data[k] = np.concatenate(all_lists[k], axis=0)


    Nconf  = len(atoms_list)
    Natoms = atoms_list[0].get_global_number_of_atoms()
    print("N conf.  : {:d}".format(Nconf))
    print("N atoms. : {:d}".format(Natoms))
    for k in all_lists.keys():
        # print("reshaping '{:s}' from {}".format(k,data[k].shape),end="")
        if isinstance(model.implemented_properties[k][1],int) or len(model.implemented_properties[k][1]) == 1 :
            # info
            whereto[k] = "info"
        elif 'natoms' in model.implemented_properties[k][1]:
            # arrays
            whereto[k] = "arrays"
        else:
            raise ValueError("coding error")
        
    for k in all_lists.keys():    
        print("reshaping '{:s}' from {}".format(k,data[k].shape),end="")
        if whereto[k] == "info":
            data[k] = data[k].reshape((Nconf,-1))
        elif whereto[k] == "arrays":
            data[k] = data[k].reshape((Nconf,Natoms,-1))
        else:
            raise ValueError("`whereto[{k}]` can be either `info` or `arrays`.")
        print(" to {}".format(data[k].shape))

    # Store data in atoms objects
    for n,atoms in enumerate(atoms_list):
        atoms.calc = None  # crucial
        for k in all_lists.keys():
            if whereto[k] == "info":
                atoms.info[args.info_prefix + k] = data[k][n]
            elif whereto[k] == "arrays":
                atoms.arrays[args.info_prefix + k] = data[k][n]
            else:
                raise ValueError("`whereto[{k}]` can be either `info` or `arrays`.")

    # Write atoms to output path
    print("saving output to file '{:s}'".format(args.output))
    ase.io.write(args.output, images=atoms_list, format="extxyz")

if __name__ == "__main__":
    main()
