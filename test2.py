import time

import torch
from process_data import (
    DrugDataset,
    InteractionDataset,
    drug_collate_fn,
    itc_collate_fn,
)
from torch.utils.data import DataLoader


def test_drug_num_workers(num_workers):

    drug_set = DrugDataset("./split_data/drugbank-random-42")
    drug_loader = DataLoader(
        drug_set,
        batch_size=2048,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=drug_collate_fn,
    )
    start = time.time()
    for drug in drug_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")


def test_itc_num_workers(num_workers):

    start = time.time()
    itc_set = InteractionDataset("./split_data/drugbank-random-42", "train")
    itc_loader = DataLoader(
        itc_set,
        batch_size=20480,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=itc_collate_fn,
    )

    for itc in itc_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")


if __name__ == "__main__":
    pass
