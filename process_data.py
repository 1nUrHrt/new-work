import os.path
import logging
from typing import Literal
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, Batch
import time


logger = logging.getLogger("data")


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
        n_drugs = len(df)
        logger.info(
            "Building molecular graphs for %d drugs  |  add_global=%s",
            n_drugs,
            add_global_features,
        )
        if mask is not None:
            n_masked = mask.sum() if hasattr(mask, "sum") else len(mask)
            self.drug.loc[mask, "mol"] = self.drug.loc[mask, "smile"].map(
                lambda s: smiles_to_graph(s, self.add_global_features)
            )
            logger.info("Graphs computed for %d/%d drugs (masked)", n_masked, n_drugs)
        else:
            self.drug["mol"] = self.drug["smile"].map(
                lambda s: smiles_to_graph(s, self.add_global_features)
            )
            logger.info("Graphs computed for all %d drugs", n_drugs)

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


def load_data(
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
    train_size: float = 0.8,
    seed=42,
    data_split: Literal["train", "test"] = "train",
):
    base_dir = os.path.join("./data", data_source + "-" + split_type)
    all_drug = pd.read_csv(os.path.join(base_dir, "drug.csv"))
    csv_name = "test.csv" if data_split == "test" else "train.csv"
    itc = pd.read_csv(os.path.join(base_dir, csv_name))
    logger.info(
        "Loading data  |  source=%s  split=%s  file=%s  drugs=%d  pairs=%d",
        data_source,
        split_type,
        csv_name,
        len(all_drug),
        len(itc),
    )
    sub_drug = (
        pd.concat([itc["drug1"], itc["drug2"]])
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    sub_drug_map = {key: i for i, key in enumerate(sub_drug)}
    itc["drug1"] = itc["drug1"].map(sub_drug_map)
    itc["drug2"] = itc["drug2"].map(sub_drug_map)

    sub_drug_ids = set(sub_drug.values)
    mask = [i in sub_drug_ids for i in range(len(all_drug))]

    all_drug_set = DrugDataset(all_drug, mask=mask)
    sub_drug_set = SubDrugDataset(sub_drug, all_drug_set)

    if data_split == "test":
        logger.info("Test data ready  |  drugs=%d  pairs=%d", len(sub_drug), len(itc))
        return (sub_drug_set, InteractionDataset(itc))

    train_itc, valid_itc = train_test_split(
        itc, train_size=train_size, random_state=seed, stratify=itc["label"]
    )
    train_itc = train_itc.reset_index(drop=True)
    valid_itc = valid_itc.reset_index(drop=True)
    logger.info(
        "Train/val split  |  train_pairs=%d  val_pairs=%d  seed=%d",
        len(train_itc),
        len(valid_itc),
        seed,
    )

    return (sub_drug_set, InteractionDataset(train_itc), InteractionDataset(valid_itc))


def split_data(
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
    train_size=0.8,
    seed=42,
):
    save_dir = os.path.join("./data", data_source + "-" + split_type)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(
        "Splitting data  |  source=%s  split=%s  train_size=%.2f  seed=%d",
        data_source,
        split_type,
        train_size,
        seed,
    )
    if data_source == "drugbank":
        if split_type == "random":
            split_drugbank_random(
                pd.read_csv("./data/drugbank.tab", sep="\t"), train_size, seed, save_dir
            )
        else:
            logger.warning(
                "Split type '%s' not yet implemented for drugbank", split_type
            )
    else:
        logger.warning("Data source '%s' not yet implemented", data_source)


def split_drugbank_random(df, train_size, seed, save_dir):
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
    itc["Y"] = itc["Y"] - 1
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
    logger.info(
        "Split complete  |  drugs=%d  total_pairs=%d  train=%d  test=%d  saved to %s",
        len(drug),
        len(df),
        len(train),
        len(test),
        save_dir,
    )


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    features = []

    # 1. Atom symbol (10)
    features += one_hot_encoding(
        atom.GetSymbol(), ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Other"]
    )

    # 2. Degree (6)
    features += one_hot_encoding(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6],
    )

    # 3. Total hydrogens (5)
    features += one_hot_encoding(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4, 5],
    )

    # 4. Formal charge (5)
    features += one_hot_encoding(
        atom.GetFormalCharge(),
        [-2, -1, 0, 1, 2],
    )

    # 5. Aromaticity (1)
    features.append(int(atom.GetIsAromatic()))

    # 6. In ring (1)
    features.append(int(atom.IsInRing()))

    # 7. Hybridization (5)
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

    # 8. Chiral tag (3)
    features += one_hot_encoding(
        atom.GetChiralTag(),
        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ],
    )

    # 9. Atomic mass (1)
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

    # Bond stereochemistry
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

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge features
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

    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Attach global molecular descriptors
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


def itc_collate_fn(batch):
    drug1 = []
    drug2 = []
    label = []
    for data in batch:
        drug1.append(data["drug1"])
        drug2.append(data["drug2"])
        label.append(data["label"])
    return torch.tensor(drug1), torch.tensor(drug2), torch.tensor(label)


__all__ = ["Timer", "load_data", "split_data", "drug_collate_fn", "itc_collate_fn"]
