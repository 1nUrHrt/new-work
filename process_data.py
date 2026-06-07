import os.path
import logging
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, Batch
# from rdkit import RDLogger

# RDLogger.DisableLog("rdApp.warning")

import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

class DrugDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mask=None, add_global_features=False):
        self.drug = df
        self.add_global_features = add_global_features
        # self.pt = Chem.GetPeriodicTable()
        if mask is not None:
            self.drug.loc[mask, "mol"] = self.drug.loc[mask, "smile"].map(
                lambda s: smiles_to_graph(s, self.add_global_features)
            )
        else:
            self.drug["mol"] = self.drug["smile"].map(
                lambda s: smiles_to_graph(s, self.add_global_features)
            )

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, idx):
        return self.drug.loc[idx, "mol"]


class SubDrugDataset(Dataset):
    def __init__(self, indices: pd.Series, drug_dataset: DrugDataset):
        self.indices = indices
        self.drug_dataset = drug_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.drug_dataset[self.indices[idx]]

    def drug_collate_fn(self, batch):
        return Batch.from_data_list(batch)


class InteractionDataset(Dataset):
    def __init__(self, itc: pd.DataFrame):
        super().__init__()
        self.itc = itc

    def __len__(self):
        return len(self.itc)

    @property
    def scenario(self):
        return self.itc["scenario"].drop_duplicates(keep="first").reset_index(drop=True)

    @property
    def scenario_label(self):
        return self.itc["scenario"]

    def __getitem__(self, idx):
        return self.itc.loc[idx]

    @property
    def label(self):
        return self.itc["label"]

    def itc_collate_fn(self, batch):
        drug1 = []
        drug2 = []
        label = []
        for data in batch:
            drug1.append(data[0])
            drug2.append(data[1])
            label.append(data[2])
        return torch.tensor(drug1), torch.tensor(drug2), torch.tensor(label)


def load_train_data(
    data_dir: str = "./data/split-data",
    train_size: float = 0.8,
    seed=42,
    scenario=False,
):
    all_drug = pd.read_csv(os.path.join(data_dir, "drug.csv"))
    itc = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_drug = (
        pd.concat([itc["drug1"], itc["drug2"]])
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    train_drug_map = {key: i for i, key in enumerate(train_drug)}
    itc["drug1"] = itc["drug1"].map(train_drug_map)
    itc["drug2"] = itc["drug2"].map(train_drug_map)

    train_itc, valid_itc = train_test_split(
        itc, train_size=train_size, random_state=seed, stratify=itc["Y"]
    )

    mask = [True if i in train_drug else False for i in range(len(all_drug))]

    all_drug_set = DrugDataset(all_drug, mask=mask)
    train_drug_set = SubDrugDataset(train_drug, all_drug_set)

    return train_drug_set, InteractionDataset(train_itc), InteractionDataset(valid_itc)


def split_data(
    inuput_file="./data/drugbank.tab",
    save_dir="./data/split-data",
    train_size=0.8,
    seed=42,
    random=True,
):
    logger = logging.getLogger("DataSplit")
    df = pd.read_csv(inuput_file, sep="\t")
    drug1 = df[["ID1", "X1"]].drop_duplicates(keep="first")
    drug2 = df[["ID2", "X2"]].drop_duplicates(keep="first")
    columns = ["id", "smile"]
    drug1.columns = columns
    drug2.columns = columns
    drug = (
        pd.concat([drug1, drug2])
        .drop_duplicates(subset=columns, keep="first")
        .reset_index(drop=True)
    )
    id_map = {key: i for i, key in enumerate(drug["id"])}

    itc = df[["ID1", "ID2", "Y"]]
    itc["ID1"] = itc["ID1"].map(lambda x: id_map.get(x, -1))
    itc["ID2"] = itc["ID2"].map(lambda x: id_map.get(x, -1))
    train, test = train_test_split(
        itc, train_size=train_size, random_state=seed, stratify=itc["Y"]
    )
    os.makedirs(save_dir, exist_ok=True)
    drug["smile"].to_csv(os.path.join(save_dir, "drug.csv"), index=False)
    train.columns = ["drug1", "drug2", "label"]
    test.columns = ["drug1", "drug2", "label"]
    train.to_csv(
        os.path.join(save_dir, "train.csv"),
        columns=["drug1", "drug2", "label"],
        index=False,
    )
    test.to_csv(
        os.path.join(save_dir, "test.csv"),
        columns=["drug1", "drug2", "label"],
        index=False,
    )
    logger.info("Split data successfully, saved to directory: '%s'", save_dir)
    logger.info(
        "Total drug count: %d,Total pair count: %d ,Train pair count: %d, Test pair count: %d",
        len(drug),
        len(df),
        len(train),
        len(test),
    )


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    features = []

    # 1. 原子类型 (10)
    features += one_hot_encoding(
        atom.GetSymbol(), ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Other"]
    )

    # 2. 度 (6)
    features += one_hot_encoding(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6],
    )

    # 3. 总氢原子数 (5)
    features += one_hot_encoding(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4, 5],
    )

    # 4. 形式电荷 (5)
    features += one_hot_encoding(
        atom.GetFormalCharge(),
        [-2, -1, 0, 1, 2],
    )

    # 5. 芳香性 (1)
    features.append(int(atom.GetIsAromatic()))

    # 6. 是否在环中 (1)
    features.append(int(atom.IsInRing()))

    # 7. 杂化类型 (5)
    features += one_hot_encoding(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.OTHER,
        ],
    )

    # 8. 手性中心 (3)
    features += one_hot_encoding(
        atom.GetChiralTag(),
        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ],
    )

    # 9.原子质量 (1)
    features.append(atom.GetMass() / 100.0)

    return features


def bond_features(bond):
    bond_type = bond.GetBondType()

    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    # 键立体化学
    features += one_hot_encoding(
        bond.GetStereo(),
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
        ],
    )

    return features


def smiles_to_graph(smiles, add_global_features):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: '{smiles}'")

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception as e:
        raise ValueError(
            f"UpdatePropertyCache failed for SMILES: '{smiles}'. Original error: {e}"
        ) from e

    # 节点特征
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # 边特征
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 构建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 添加全局分子特征
    if add_global_features:
        try:
            global_features = torch.tensor(
                [
                    Descriptors.MolWt(mol) / 500.0,
                    Descriptors.MolLogP(mol) / 10.0,
                    Descriptors.TPSA(mol) / 200.0,
                    Descriptors.NumHDonors(mol) / 10.0,
                    Descriptors.NumHAcceptors(mol) / 10.0,
                    Descriptors.NumRotatableBonds(mol) / 20.0,
                    Chem.rdMolDescriptors.CalcNumRings(mol) / 10.0,
                ],
                dtype=torch.float,
            ).unsqueeze(0)
            data.global_features = global_features
        except:
            data.global_features = torch.zeros((1, 7))

    return data


def drug_collate_fn(batch):
    return Batch.from_data_list(batch)
