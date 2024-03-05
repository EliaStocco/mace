###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
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
    return parser.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    contributions_list = []
    dipoles_list = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=False)

        if "dipole" in output:
            dipoles_list.append(torch_tools.to_numpy(output["dipole"]))

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

    dipoles = np.concatenate(dipoles_list, axis=0)

    # assert len(atoms_list) == len(energies) == len(forces_list)
    assert len(atoms_list) == len(dipoles)

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    # Store data in atoms objects
    for i, (atoms, dipole) in enumerate(zip(atoms_list, dipoles)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "dipole"] = dipole

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
