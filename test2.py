import pandas as pd
from rdkit import Chem
import torch

from process_data import smiles_to_graph


if __name__ == "__main__":
    atom_symbol = {}
    atom_counter = {}
    drug = pd.read_csv("./data/drugbank-random/drug.csv")
    drug["mol"] = [smiles_to_graph(s) for s in drug["smile"]]
    is_nan_arr = []
    is_inf_arr = []
    # mol = drug.loc[2,"mol"]
    # print(mol.x)

    for i, mol in enumerate(drug["mol"]):
        x, edge_attr, g = mol.x, mol.edge_attr, mol.global_features
        x_is_nan = torch.any(torch.isnan(x))
        if x_is_nan.item():
            print(i, "x_is_nan")
            break
        e_is_nan = torch.any(torch.isnan(edge_attr))
        if e_is_nan.item():
            print(i, "e_is_nan")
            break
        g_is_nan = torch.any(torch.isnan(g))
        if x_is_nan.item():
            print(i, "g_is_nan")
            break
        x_is_inf = torch.any(torch.isinf(x))
        if x_is_inf.item():
            print(i, "x_is_inf")
            break
        e_is_inf = torch.any(torch.isinf(edge_attr))
        if e_is_inf.item():
            print(i, "e_is_inf")
            break
        g_is_inf = torch.any(torch.isinf(g))
        if g_is_inf.item():
            print(i, "g_is_inf")
            break
    print("ok")

    #     is_nan_arr.append(is_nan)
    #     is_inf_arr.append(is_inf)
    # is_nan = torch.tensor(is_nan_arr)
    # is_inf = torch.tensor(is_inf_arr)
    # print(torch.any(is_nan))
    # print(torch.any(is_inf))
    #     mol = Chem.MolFromSmiles(s)
    #     if mol is None:
    #         raise ValueError(f"Failed to parse SMILES: '{s}'")

    #     try:
    #         mol.UpdatePropertyCache(strict=False)
    #     except Exception as e:
    #         raise ValueError(
    #             f"UpdatePropertyCache failed for SMILES: '{s}'. Original error: {e}"
    #         ) from e
    #     for atom in mol.GetAtoms():
    #         symbol = atom.GetSymbol()
    #         if symbol not in atom_counter:
    #             atom_counter[symbol] = 1
    #         else:
    #             atom_counter[symbol] += 1
    #         atomic_num = atom.GetAtomicNum()
    #         if atomic_num not in atom_symbol:
    #             atom_symbol[atomic_num] = symbol

    # symbol_list = [atom_symbol.get(i) for i in sorted(atom_symbol.keys())]
    # print(symbol_list)
    # print(len(symbol_list))
