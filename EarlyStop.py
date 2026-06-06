from typing import Literal


class EarlyStop:
    def __init__(
        self,
        patience: int = 5,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 1e-4,
    ):
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be either 'max' or 'min', got {mode}")
        self.patience = patience
        self.mode = mode

        self.min_delta = min_delta
        self.counter = 0
        self.best_metric_val = None
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_metric_val is None:
            is_improved = True
        else:
            if self.mode == "min":
                is_improved = metric_value < self.best_metric_val - self.min_delta
            else:
                is_improved = metric_value > self.best_metric_val + self.min_delta

        if is_improved:
            self.counter = 0
            self.best_metric_val = metric_value
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return is_improved

    def state_dict(self):
        state = {
            "counter": self.counter,
            "best_metric_val": self.best_metric_val,
            "early_stop": self.early_stop,
        }
        return state

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_metric_val = state_dict["best_metric_val"]
        self.early_stop = state_dict["early_stop"]
