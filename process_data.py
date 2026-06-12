import math
import os.path
from typing import Literal
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, ValenceType
from torch_geometric.data import Data, InMemoryDataset, Batch
import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


class DrugDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        type: Literal["train", "val", "test"] = "train",
        transform=None,
        pre_transform=None,
    ):
        self.file_name = f"{type}_drug.csv"
        self.proc_name = f"{type}_drug.pt"
        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [self.proc_name]

    @property
    def raw_file_names(self):
        return [self.file_name]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smile in df["smile"]:
            mol = smiles_to_graph(smile)
            data_list.append(mol)
        self._data, self.slices = self.collate(data_list)
        torch.save((self._data, self.slices), self.processed_paths[0])


class InteractionDataset(Dataset):
    def __init__(self, root, type: Literal["train", "val", "test"] = "train"):
        super().__init__()
        cache_key = f"{type}_itc.pt"
        cache_file_path = os.path.join(root, "processed", cache_key)
        if not os.path.exists(cache_file_path):
            os.makedirs(os.path.join(root, "processed"), exist_ok=True)
            df = pd.read_csv(os.path.join(root, "raw", f"{type}_itc.csv"))
            drug1 = torch.tensor(df["drug1"].values, dtype=torch.long)
            drug2 = torch.tensor(df["drug2"].values, dtype=torch.long)
            label = torch.tensor(df["label"].values, dtype=torch.long)
            torch.save((drug1, drug2, label), cache_file_path)

        drug1, drug2, label = torch.load(cache_file_path, weights_only=False)
        self.drug1 = drug1.share_memory_()
        self.drug2 = drug2.share_memory_()
        self.label = label.share_memory_()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.drug1[idx], self.drug2[idx], self.label[idx]


def _load_data(root: str, type: Literal["train", "val", "test"] = "train"):
    return DrugDataset(root, type), InteractionDataset(root, type)


def load_data(
    data_source: Literal["drugbank", "twosides"],
    split_type: Literal["random", "cluster"],
    type: Literal["train", "val", "test"] = "train",
    seed=42,
):
    base_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed)
    )
    return _load_data(base_dir, type)


def split_data(
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
    train_size=0.8,
    seed=42,
):
    save_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed), "raw"
    )
    os.makedirs(save_dir, exist_ok=True)
    if data_source == "drugbank":
        if split_type == "random":
            _split_drugbank_random(
                pd.read_csv("./data/drugbank.tab", sep="\t"), train_size, seed, save_dir
            )
        else:
            raise TypeError()
    else:
        raise TypeError()


def _split_drugbank_random(df: pd.DataFrame, train_size, seed, save_dir):
    drug1 = df[["ID1", "X1"]].drop_duplicates(keep="first")
    drug2 = df[["ID2", "X2"]].drop_duplicates(keep="first")
    drug1.columns = ["id", "smile"]
    drug2.columns = ["id", "smile"]
    drug = (
        pd.concat([drug1, drug2])
        .drop_duplicates(subset=["id", "smile"], keep="first")
        .reset_index(drop=True)
    )
    id_map = {row["id"]: row["smile"] for _, row in drug.iterrows()}
    itc = df[["ID1", "ID2", "Y"]].drop_duplicates(keep="first").reset_index(drop=True)
    itc["Y"] = itc["Y"] - 1
    train, test = train_test_split(
        itc, train_size=train_size, random_state=seed, stratify=itc["Y"]
    )

    train_drug = (
        pd.concat([train["ID1"], train["ID2"]], axis=0)
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    train_map = {key: i for i, key in enumerate(train_drug)}
    train["ID1"] = train["ID1"].map(train_map)
    train["ID2"] = train["ID2"].map(train_map)
    train_drug = train_drug.map(id_map)

    test_drug = (
        pd.concat([test["ID1"], test["ID2"]], axis=0)
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    test_map = {key: i for i, key in enumerate(test_drug)}
    test["ID1"] = test["ID1"].map(test_map)
    test["ID2"] = test["ID2"].map(test_map)
    test_drug = test_drug.map(id_map)
    os.makedirs(save_dir, exist_ok=True)
    train_drug.name = "smile"
    train_drug.to_csv(
        os.path.join(save_dir, "train_drug.csv"),
        index=False,
    )
    train.columns = ["drug1", "drug2", "label"]
    train.to_csv(
        os.path.join(save_dir, "train_itc.csv"),
        index=False,
    )

    test_drug.name = "smile"
    test_drug.to_csv(
        os.path.join(save_dir, "test_drug.csv"),
        index=False,
    )
    test.columns = ["drug1", "drug2", "label"]
    test.to_csv(
        os.path.join(save_dir, "test_itc.csv"),
        index=False,
    )


def _one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def _atom_features(atom):
    features = []

    # 1. Atom symbol (38)
    features += _one_hot_encoding(
        atom.GetSymbol(),
        [
            "H",
            "Li",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "K",
            "Ca",
            "Ti",
            "Cr",
            "Fe",
            "Co",
            "Cu",
            "Zn",
            "Ga",
            "As",
            "Se",
            "Br",
            "Sr",
            "Tc",
            "Ag",
            "Sb",
            "I",
            "La",
            "Gd",
            "Pt",
            "Au",
            "Hg",
            "Bi",
            "Ra",
        ],
    )

    # 2. Degree (6)
    features += _one_hot_encoding(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6],
    )

    # 3. Total hydrogens (5)
    features += _one_hot_encoding(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4, 5],
    )

    # 4. Formal charge (5)
    features += _one_hot_encoding(
        atom.GetFormalCharge(),
        [-2, -1, 0, 1, 2],
    )

    # 5. Aromaticity (1)
    features.append(int(atom.GetIsAromatic()))

    # 6. In ring (1)
    features.append(int(atom.IsInRing()))

    # 7. Hybridization (5)
    features += _one_hot_encoding(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.OTHER,
        ],
    )

    # 8. Chiral tag (3)
    features += _one_hot_encoding(
        atom.GetChiralTag(),
        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ],
    )

    # 9. Atomic mass (1)
    features.append(atom.GetMass() / 100.0)
    # 10. 显式价态 7维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.EXPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 11. 隐式价态 7维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 12. 是否杂原子 1维 (C/H=0，其余=1)
    symbol = atom.GetSymbol()
    features.append(0 if symbol in ("C", "H") else 1)
    # 13. 原子电负性 1维 (归一化)
    elect = atom.GetNumRadicalElectrons()  # 替换为电负性可自行查表映射
    features.append(elect / 4.0)
    # 14. Gasteiger 部分电荷 1维
    try:
        charge = float(atom.GetProp("_GasteigerCharge"))
    except:
        charge = 0.0
    if math.isnan(charge) or math.isinf(charge):
        charge = 0.0
    features.append(charge)

    return features


def _bond_features(bond):
    bond_type = bond.GetBondType()

    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    # Bond stereochemistry
    features += _one_hot_encoding(
        bond.GetStereo(),
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
        ],
    )

    # 1. 浮点键级 1维
    features.append(bond.GetBondTypeAsDouble())
    # 2. 是否芳香键（二次强化）1维
    features.append(int(bond.GetIsAromatic()))
    # 3. 共轭环键 1维
    features.append(int(bond.GetIsConjugated() and bond.IsInRing()))

    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol.UpdatePropertyCache(strict=False)
    AllChem.ComputeGasteigerCharges(mol)
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # ========== 全局特征 ==========
    gw = Descriptors.MolWt(mol) / 500.0
    logp = Descriptors.MolLogP(mol) / 10.0
    tpsa = Descriptors.TPSA(mol) / 200.0
    hdonor = Descriptors.NumHDonors(mol) / 10.0
    haccept = Descriptors.NumHAcceptors(mol) / 10.0
    rot_bond = Descriptors.NumRotatableBonds(mol) / 20.0
    ring_num = rdMolDescriptors.CalcNumRings(mol) / 10.0

    # 重原子数
    heavy_atom = Descriptors.HeavyAtomCount(mol) / 50.0
    # 芳香环数量
    aromatic_ring = rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0
    # 脂肪环数量
    aliphatic_ring = rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0
    # 摩尔折射率
    mr = Descriptors.MolMR(mol) / 100.0
    # 分子柔性指数
    frac_rot = Descriptors.NumRotatableBonds(mol) / max(1, mol.GetNumBonds())
    # 卤素原子总数
    halogens = (
        sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")) / 10.0
    )
    # 氧原子数、氮原子数
    o_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O") / 10.0
    n_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N") / 10.0

    graph_attr = [
        gw,
        logp,
        tpsa,
        hdonor,
        haccept,
        rot_bond,
        ring_num,
        heavy_atom,
        aromatic_ring,
        aliphatic_ring,
        mr,
        frac_rot,
        halogens,
        o_count,
        n_count,
    ]
    graph_attr = torch.tensor(graph_attr, dtype=torch.float).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr)


def drug_collate_fn(batch):
    return Batch.from_data_list(batch)


def itc_collate_fn(batch):
    drug1, drug2, label = zip(*batch)
    return torch.stack(drug1), torch.stack(drug2), torch.stack(label)


__all__ = ["Timer", "load_data", "split_data", "itc_collate_fn", "drug_collate_fn"]

if __name__ == "__main__":
    load_data("drugbank", "random", "train", 42)
