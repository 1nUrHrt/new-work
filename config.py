from typing import Literal


class Config:
    @classmethod
    def get(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class default(Config):
    encoder: Literal["AttnEncoder", "AttnResEncoder"] = "AttnEncoder"
    metric_average: Literal["macro", "weighted", "micro"] = "macro"
    data_source: Literal["drugbank", "twosides"] = "drugbank"
    split_type: Literal["random", "cluster"] = "random"

    epochs: int = 200
    node_dim = 39
    edge_dim = 10
    h_dim = 128
    lr = 0.001
    heads = 8
    dp_r = 0.1
    train_size = 0.8
    seed = 42
    block_num = 6
    block_size = 2
    class_num = 86
    drug_batch_size = 2048
    itc_batch_size = 20480
    num_workers = 2
    label_smoothing = 0.1
    min_delta = 0.001
