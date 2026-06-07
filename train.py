import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from torch.utils.data import DataLoader
from process_data import Timer, load_data, drug_collate_fn, itc_collate_fn
from model import EarlyStop, AttnEncoder, AttnResEncoder, Classifier
from typing import Literal
import config
import logging

logger = logging.getLogger("train")


def train_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):

    encoder.train()
    classifier.train()

    train_loss = 0.0
    train_acc = 0.0
    total_batch = len(itc_loader)
    batch_counter = 0
    for d1, d2, labels in itc_loader:
        batch_counter += 1
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                all_drugs = torch.cat(
                    [encoder(drugs.to(device)) for drugs in drug_loader]
                )
                logits = classifier(all_drugs[d1], all_drugs[d2])
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()

        train_loss += loss.item()
        train_acc += acc.item()
        print(
            f"\r[Train] [Batch:{batch_counter}/{total_batch}] loss:{loss.item():.5f},acc:{acc.item():.5f}",
            end="",
            flush=True,
        )
    print()
    avg_train_loss = train_loss / len(itc_loader)
    avg_train_acc = train_acc / len(itc_loader)
    return avg_train_loss, avg_train_acc


def val_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    criterion,
    metric_average,
    device,
):
    encoder.eval()
    classifier.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    return (
        val_loss / len(itc_loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average=metric_average, zero_division=0),
        roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average=metric_average,
        ),
    )


def train(
    name,
    encoder,
    metric_average,
    data_source,
    split_type,
    epochs,
    node_dim,
    edge_dim,
    h_dim,
    lr,
    heads,
    dp_r,
    train_size,
    seed,
    block_num,
    block_size,
    class_num,
    drug_batch_size,
    itc_batch_size,
    num_workers,
    label_smoothing,
    min_delta,
    history=None,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_itc_generator = torch.Generator()
    train_itc_generator.manual_seed(seed)

    pin_memory = True if torch.cuda.is_available() else False

    print(
        f"[Train Config] Total Epochs:{epochs} Device:{device} Manual Seed:{seed} Metric Average:{metric_average}"
    )

    base_dir = os.path.join("./checkpoints", name)
    os.makedirs(base_dir, exist_ok=True)
    best_path = os.path.join(base_dir, "best.pt")
    history_path = os.path.join(base_dir, "history.pt")
    result_path = os.path.join(base_dir, "result.csv")

    datasets = load_data(data_source, split_type, train_size, seed)
    if len(datasets) != 3:
        raise ValueError("train_size must be not None,got {train_size}")
    drug_set, train_itc, val_itc = datasets
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
        generator=train_itc_generator,
    )

    val_loader = DataLoader(
        val_itc,
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
    optimizer = Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stop = EarlyStop(patience=10, mode="max", min_delta=min_delta)
    scaler = torch.GradScaler() if torch.cuda.is_available() else None

    result = {
        "train_loss": [],
        "train_acc": [],
        "train_timer": [],
        "val_timer": [],
    }

    total_timer = 0

    if history is not None:
        current_checkpoint = history
        start_epoch = current_checkpoint["epoch"]
        encoder.load_state_dict(history["encoder"])
        classifier.load_state_dict(history["classifier"])
        optimizer.load_state_dict(history["optimizer"])
        scheduler.load_state_dict(history["scheduler"])
        early_stop.load_state_dict(history["early_stop"])
        if scaler is not None and history["scaler"] is not None:
            scaler.load_state_dict(history["scaler"])

        if (
            "train_itc_generator" in history
            and history["train_itc_generator"] is not None
        ):
            train_itc_generator.set_state(history["train_itc_generator"])

        if torch.cuda.is_available() and history["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(history["cuda_random"])
        torch.random.set_rng_state(history["torch_random"])
        np.random.set_state(history["numpy_random"])
        random.setstate(history["python_random"])
        print("[History Model Loaded]")

    if early_stop.early_stop:
        print(f"[Early Stop] [Epoch:{start_epoch}/{epochs}] trigger early stop")
        return

    if start_epoch >= epochs:
        print(
            f"[Stop Experiment] [Epoch:{start_epoch}/{epochs}] experiment has finished"
        )
        return

    if history is not None and history["result"] is not None:
        df = pd.read_csv(result_path)
        result = df.to_dict(orient="list")
        total_timer = sum(result["train_timer"]) + sum(result["val_timer"])

    print("[Train Srart]")
    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"[Current Epoch:{current_epoch}/{epochs}]")
        with Timer() as timer:
            train_loss, train_acc = train_one_epoch(
                encoder,
                classifier,
                drug_loader,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        print(
            f"[Train] loss:{train_loss:.5f},acc:{train_acc:.5f},elapsed:{timer.elapsed:.5f} s"
        )
        total_timer += timer.elapsed
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["train_timer"].append(timer.elapsed)

        with Timer() as timer:
            val_loss, val_acc, val_f1_score, val_auc = val_one_epoch(
                encoder,
                classifier,
                drug_loader,
                val_loader,
                criterion,
                metric_average,
                device,
            )

        print(
            f"[Val Total] loss:{val_loss:.5f},acc:{val_acc:.5f},f1_score:{val_f1_score:.5f},auc:{val_auc:.5f},elapsed:{timer.elapsed:.5f} s"
        )
        total_timer += timer.elapsed
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["val_f1_score"].append(val_f1_score)
        result["val_auc"].append(val_auc)
        result["val_timer"].append(timer.elapsed)

        scheduler.step(val_f1_score)

        is_improved = early_stop(val_f1_score)

        if not is_improved:
            print(
                f"[Early Stop] [Epoch:{current_epoch}/{epochs}] trigger counter: {early_stop.counter}/{early_stop.patience}"
            )
        else:
            torch.save(
                {
                    "epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                best_path,
            )
            print(
                f"[Save Model] [Epoch:{current_epoch}/{epochs}] save best Model successfully"
            )

        checkpoint = {
            "epoch": current_epoch,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stop": early_stop.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cuda_random": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "torch_random": torch.random.get_rng_state(),
            "numpy_random": np.random.get_state(),
            "python_random": random.getstate(),
            "train_itc_generator": train_itc_generator.get_state(),
        }

        torch.save(checkpoint, history_path)
        print(
            f"[Save Checkpoint] [Epoch:{current_epoch}/{epochs}] save current checkpoint successfully"
        )

        print(f"[Epoch {current_epoch}/{epochs}] total elapsed:{total_timer}")

        pd.DataFrame(result).to_csv(result_path, index=False)

        if early_stop.early_stop:
            print(f"[Early Stop] [Epoch:{current_epoch}/{epochs}] trigger early stop")
            break


def resume_training(name: str):
    cfg = None
    try:
        cfg = getattr(config, name).get()
    except AttributeError:
        logger.warning("%s 配置不存在", name)
        return

    history = torch.load(os.path.join("./checkpoin", name, "history.pt"))
    result = pd.read_csv(os.path.join("./checkpoin", name, "result.csv"))
    history["result"] = result
    train(history=history, **cfg)


def run_training(
    name: str,
    encoder: Literal["AttnEncoder", "AttnResEncoder"] = "AttnEncoder",
    metric_average: Literal["macro", "weighted", "micro"] = "macro",
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
):
    cfg = None
    try:
        cfg = getattr(config, name).get()
    except AttributeError:
        logger.warning("%s 配置不存在", name)
        return
    cfg["name"] = name
    cfg["encoder"] = encoder
    cfg["metric_average"] = metric_average
    cfg["data_source"] = data_source
    cfg["split_type"] = split_type
    train(**cfg)


__all__ = ["resume_training", "run_training"]
