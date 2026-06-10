import os

import pandas as pd
import matplotlib.pyplot as plt



def heatmap(file_name: str):
   
    out_dir = "./heatmap"
    os.makedirs(out_dir, exist_ok=True)
    cm_df = pd.read_csv(f"./checkpoints/{file_name}/confusion_matrix.csv", index_col=0)

    cm = cm_df.values

    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # 绘图
    plt.figure(figsize=(20, 18))
    plt.imshow(cm, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(label="Count")
    plt.title("Confusion Matrix (86 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{file_name}-cm.png"), dpi=150)


if __name__ == "__main__":
    file_names = ["attn_gin_tf"]
    for i in file_names:
        heatmap(i)