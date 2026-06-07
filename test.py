import torch

from torch.utils.data import DataLoader
from process_data import Timer, load_data, drug_collate_fn, itc_collate_fn


def test(
    name, data_source, split_type, seed, drug_batch_size, num_workers, itc_batch_size
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
    train_loader = DataLoader(
        test_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )

    encoder = get_encoder(config["encoder"]).to(device)
    classifier = get_classifier(config["classifier"]).to(device)
    criterion = get_criterion(config["criterion"])

    base_dir = os.path.join(config["save_dir"], experiment_name)
    best_save_path = os.path.join(base_dir, config["best_save_name"])
    test_dict_path = os.path.join(base_dir, config["test_dict_name"])
    record = {"loss": [], "acc": [], "f1_score": [], "auc": []}
    if not os.path.exists(best_save_path):
        print(f"The best model of current experiment:{experiment_name} don't exist")
        return
    print("loading best models")
    best_model = torch.load(best_save_path, weights_only=False)
    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])
    record["loss"].append(best_model["loss"])
    record["acc"].append(best_model["acc"])
    record["f1_score"].append(best_model["f1_score"])
    record["auc"].append(best_model["auc"])

    print(
        f"[Test Config] Device:{device} Metric:{best_model['metric']} Metric Average:{best_model['metric_average']}"
    )

    print("[Test Start]")

    encoder.eval()
    classifier.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in test_set_loader])

        for d1, d2, labels in test_itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    avg_val_loss = val_loss / len(test_itc_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(
        all_labels, all_preds, average=best_model["metric_average"], zero_division=0
    )
    val_auc = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=best_model["metric_average"]
    )

    record["loss"].append(avg_val_loss)
    record["acc"].append(val_acc)
    record["f1_score"].append(val_f1)
    record["auc"].append(val_auc)
    print(
        f"[Best] loss:{record['loss'][0]:.5},acc:{record['acc'][0]:.5},f1:{record['f1_score'][0]:.5},auc:{record['auc'][0]:.5}"
    )
    print(
        f"[Test] loss:{record['loss'][1]:.5},acc:{record['acc'][1]:.5},f1:{record['f1_score'][1]:.5},auc:{record['auc'][1]:.5}"
    )
    pd.DataFrame(record).to_csv(test_dict_path, index=False)
