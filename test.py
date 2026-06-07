import os
import pandas as pd
import torch
import config
from process_data import load_data, drug_collate_fn, itc_collate_fn
from model import AttnEncoder, AttnResEncoder, Classifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import logging

logger = logging.getLogger("test")


def test(
    name,
    encoder,
    metric_average,
    data_source,
    split_type,
    node_dim,
    edge_dim,
    h_dim,
    heads,
    dp_r,
    seed,
    block_num,
    block_size,
    class_num,
    drug_batch_size,
    itc_batch_size,
    num_workers,
    label_smoothing,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(data_source, split_type, seed=seed)

    if len(datasets) != 2:
        raise ValueError("train_size must be None,got {train_size}")
    drug_set, test_itc = datasets

    pin_memory = True if torch.cuda.is_available() else False

    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False,
    )

    if encoder == "AttnEncoder":
        encoder = AttnEncoder(node_dim, edge_dim, h_dim, block_num, dp_r, heads)
    else:
        encoder = AttnResEncoder(
            node_dim, edge_dim, h_dim, block_num, dp_r, heads, block_size=block_size
        )
    classifier = Classifier(h_dim, class_num, dp_r).to(device)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)

    base_dir = os.path.join("./checkpoints", name)
    best_path = os.path.join(base_dir, "best.pt")
    evaluate_path = os.path.join(base_dir, "evaluate.csv")
    evaluate = {}
    if not os.path.exists(best_path):
        print(f"The best model of current experiment:{name} don't exist")
        return

    best_model = torch.load(best_path, weights_only=False)
    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])

    encoder.eval()
    classifier.eval()

    test_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in test_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            test_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    evaluate["best_epoch"] = best_model["epoch"]
    evaluate["test_loss"] = test_loss / len(test_loader)
    evaluate["test_acc"] = accuracy_score(all_labels, all_preds)
    evaluate["test_f1_score"] = f1_score(
        all_labels, all_preds, average=metric_average, zero_division=0
    )
    evaluate["test_auc"] = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=metric_average
    )
    pd.DataFrame(evaluate).to_csv(evaluate_path, index=False)


def run_test(name: str):
    cfg = None
    try:
        cfg = getattr(config, name).get()
    except AttributeError:
        logger.warning("%s 配置不存在", name)
        return
    cfg["name"] = name
    del cfg["epochs"]
    del cfg["lr"]
    del cfg["min_delta"]
    del cfg["train_size"]
    test(**cfg)


__all__ = ['run_test']
