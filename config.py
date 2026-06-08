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
    node_dim: int = 39
    edge_dim: int = 10
    h_dim: int = 128
    lr: float = 0.001
    heads: int = 8
    dp_r: float = 0.1
    train_size: float = 0.8
    seed: int = 42
    block_num: int = 6
    block_size: int = 2
    class_num: int = 86
    drug_batch_size: int = 2048
    itc_batch_size: int = 20480
    num_workers: int = 2
    label_smoothing: float = 0.1
    min_delta: float = 0.001



class attn_CosLR(Config):
    encoder: Literal["AttnEncoder", "AttnResEncoder"] = "AttnEncoder"
    metric_average: Literal["macro", "weighted", "micro"] = "macro"
    data_source: Literal["drugbank", "twosides"] = "drugbank"
    split_type: Literal["random", "cluster"] = "random"

    epochs: int = 200
    node_dim: int = 86
    edge_dim: int = 13
    h_dim: int = 128
    lr: float = 0.0001
    heads: int = 8
    dp_r: float = 0.1
    train_size: float = 0.8
    seed: int = 42
    block_num: int = 6
    block_size: int = 1
    class_num: int = 86
    drug_batch_size: int = 2048
    itc_batch_size: int = 20480
    num_workers: int = 2
    label_smoothing: float = 0.1
    min_delta: float = 0.001
