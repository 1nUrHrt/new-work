import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from process_data import Timer, load_data, itc_collate_fn, drug_collate_fn
from model import EarlyStop, Classifier, AttnGINTFEncoder
import config
from config import Config
from custom_printer import train_ptr as ptr, ptr_color


def _train_one_epoch(
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
        ptr.w_flush(
            "train",
            f"[Batch:{batch_counter}/{total_batch}] loss:{loss.item():.5f},acc:{acc.item():.5f}",
        )
    avg_train_loss = train_loss / len(itc_loader)
    avg_train_acc = train_acc / len(itc_loader)
    return avg_train_loss, avg_train_acc


def _val_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    criterion,
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
    avg_loss = val_loss / len(itc_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    return (avg_loss, acc, f1, auc, cm)


def _train(
    config: Config,
    history=None,
):
    name = type(config).__name__
    clssifier_type = config.classifier
    data_source = config.data_source
    split_type = config.split_type
    epochs = config.epochs
    node_dim = config.node_dim
    edge_dim = config.edge_dim
    graph_dim = config.graph_dim
    d_model = config.d_model
    lr = config.lr
    heads = config.heads
    dp_r = config.dp_r
    train_size = config.train_size
    weight_decay = config.weight_decay
    seed = config.seed
    block_num = config.block_num
    class_num = config.class_num
    drug_batch_size = config.drug_batch_size
    itc_batch_size = config.itc_batch_size
    label_smoothing = config.label_smoothing
    num_workers = config.num_workers
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_itc_generator = torch.Generator()
    train_itc_generator.manual_seed(seed)

    base_dir = os.path.join("./checkpoints", name)
    os.makedirs(base_dir, exist_ok=True)
    best_path = os.path.join(base_dir, "best.pt")
    history_path = os.path.join(base_dir, "history.pt")
    result_path = os.path.join(base_dir, "result.csv")
    cm_path = os.path.join(base_dir, "confusion_matrix.csv")

    drug_set, itc_set = load_data(data_source, split_type, "train", seed)
    train_idx, val_idx = train_test_split(
        range(len(itc_set)),
        train_size=train_size,
        stratify=itc_set.label,
        random_state=42,
    )

    train_itc = Subset(itc_set, train_idx)
    val_itc = Subset(itc_set, val_idx)
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=train_itc_generator,
    )

    val_loader = DataLoader(
        val_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    encoder = AttnGINTFEncoder(
        node_dim, edge_dim, graph_dim, d_model, block_num, dp_r, heads
    ).to(device)

    classifier = Classifier(d_model, class_num, dp_r).to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stop = EarlyStop(patience=10, mode="max", min_delta=0.001)
    scaler = torch.GradScaler() if torch.cuda.is_available() else None

    ptr.set_value_batch(
        {
            "name": name,
            "epochs": epochs,
            "encoder": type(encoder).__name__,
            "classifier": "classifier",
            "lr": lr,
            "data_source": data_source,
            "split_type": split_type,
            "seed": seed,
            "device": device,
            "resume": "False",
            "epoch": f"0/{epochs}",
            "current_lr": lr,
            "elapsed": 0,
            "early_stop": f"0/{early_stop.patience}",
            "state": "pending",
        }
    )

    result = {
        "train_loss": [],
        "train_acc": [],
        "train_timer": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_score": [],
        "val_auc": [],
        "val_timer": [],
    }

    total_timer = 0

    if history is not None:
        start_epoch = history["epoch"]
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
        result = history["result"].to_dict(orient="list")
        total_timer = sum(result["train_timer"]) + sum(result["val_timer"])

        ptr.write("epoch", f"{start_epoch}/{epochs}")
        ptr.write("resume", "True", ptr_color.flag)
        ptr.write("current_lr", f"{optimizer.param_groups[0]['lr']:.7f}")
        ptr.scl_flush(
            "info",
            f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}.",
            ptr_color.notice,
        )

    if early_stop.early_stop:
        ptr.scroll(
            "info",
            f"Early stop already triggered at epoch {start_epoch}/{epochs} — nothing to resume",
            ptr_color.warning,
        )
        ptr.write(
            "early_stop",
            f"{early_stop.counter}/{early_stop.patience}",
            ptr_color.error,
        )
        ptr.w_flush("state", "finished", ptr_color.done)
        return

    if start_epoch >= epochs:
        ptr.scroll(
            "info",
            f"Experiment already finished at epoch {start_epoch}/{epochs} — nothing to resume",
            ptr_color.warning,
        )
        ptr.w_flush("state", "finished", ptr_color.done)
        return

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        ptr.w_flush("epoch", f"{current_epoch}/{epochs}")
        with Timer() as timer:
            ptr.w_flush("state", "training", ptr_color.training)
            train_loss, train_acc = _train_one_epoch(
                encoder,
                classifier,
                drug_loader,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        total_timer += timer.elapsed
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["train_timer"].append(timer.elapsed)
        ptr.write(
            "train",
            f"loss={train_loss:.5f}  acc={train_acc:.5f}  ({timer.elapsed:.5f} s)",
        )
        ptr.w_flush(
            "elapsed",
            f"{total_timer:.5f}",
        )

        with Timer() as timer:
            ptr.w_flush("state", "valdating", ptr_color.validating)
            val_loss, val_acc, val_f1_score, val_auc, cm = _val_one_epoch(
                encoder,
                classifier,
                drug_loader,
                val_loader,
                criterion,
                device,
            )
        total_timer += timer.elapsed
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["val_f1_score"].append(val_f1_score)
        result["val_auc"].append(val_auc)
        result["val_timer"].append(timer.elapsed)

        ptr.write(
            "val",
            f"loss={val_loss:.5f}  acc={val_acc:.5f}  f1_score={val_f1_score:.5f}  auc={val_auc:.5f}  ({timer.elapsed:.5f} s)",
        )
        ptr.write(
            "elapsed",
            f"{total_timer:.5f}",
        )

        scheduler.step()
        is_improved = early_stop(val_f1_score)
        ptr.write("state", "waiting", ptr_color.pending)
        ptr.w_flush("current_lr", f"{optimizer.param_groups[0]['lr']:.7f}")
        if is_improved:
            torch.save(
                {
                    "epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                best_path,
            )
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{i}" for i in range(class_num)],
                columns=[f"Pred_{i}" for i in range(class_num)],
            )
            cm_df.to_csv(cm_path)
            ptr.write(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.info,
            )
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Model performance improved",
                ptr_color.notice,
            )
            ptr.scl_flush(
                "info",
                f"[{current_epoch}/{epochs}] best model improved → saved best.pt and confusion_matrix.csv",
                ptr_color.notice,
            )
        else:
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Model performance not improved",
                ptr_color.warning,
            )
            ptr.w_flush(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.warning,
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

        if current_epoch % 5 == 0:
            torch.save(checkpoint, history_path)
            pd.DataFrame(result).to_csv(result_path, index=False)
            ptr.scl_flush(
                "info",
                "[{current_epoch}/{epochs}]  checkpoint saved (history.pt + result.csv)",
                ptr_color.notice,
            )

        if early_stop.early_stop:
            ptr.write(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.error,
            )
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Early stopping triggered",
                ptr_color.warning,
            )
            ptr.w_flush("state", "finished", ptr_color.done)
            break


def resume_training(config_class_name: str):
    cfg = getattr(config, config_class_name)
    history_path = os.path.join("./checkpoints", type(cfg).__name__, "history.pt")
    result_path = os.path.join("./checkpoints", type(cfg).__name__, "result.csv")
    history = torch.load(history_path, weights_only=False)
    result = pd.read_csv(result_path)
    history["result"] = result
    _train(cfg, history=history)


def run_training(config_class_name: str):
    cfg = getattr(config, config_class_name)
    _train(cfg)


__all__ = ["resume_training", "run_training"]
