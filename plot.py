import os

from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def heatmap(file_name: str):

    out_dir = os.path.join("./graph", file_name)
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
    plt.savefig(os.path.join(out_dir, "cm.png"), dpi=150)
    plt.clf()


def loss_chart(df: pd.DataFrame, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for col in df.columns:
        ax.plot(df.index + 1, df[col], label=col.replace("_", " "))
    # ax.set_xlim(0, 150)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.autoscale(axis="x")
    ax.set_ylim(0, 4.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.set_title("训练 & 验证Loss变化曲线", fontsize=14, pad=15)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    fig.savefig(
        os.path.join(save_dir, "loss.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def acc_chart(df: pd.DataFrame, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for col in df.columns:
        ax.plot(df.index + 1, df[col], label=col.replace("_", " "))
    # ax.set_xlim(0, 150)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.autoscale(axis="x")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.set_title("训练 & 验证ACC变化曲线", fontsize=14, pad=15)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("ACC", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    fig.savefig(
        os.path.join(save_dir, "acc.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def line_chart(file_name: str):
    save_dir = os.path.join("./graph", file_name)
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(f"./checkpoints/{file_name}/result.csv")
    loss_chart(df[["train_loss", "val_loss"]], save_dir)
    acc_chart(df[["train_acc", "val_acc"]], save_dir)
    # fig_type = {
    #     "Loss": ["train_loss", "val_loss"],
    #     "Acc": ["train_acc", "val_acc"],
    #     "F1_Score": ["val_f1_score"],
    #     "Auc": ["val_auc"],
    # }
    # for type, cols in fig_type.items():
    #     type_df = df[cols]
    #     fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    #     for col in type_df.columns:
    #         ax.plot(type_df.index + 1, type_df[col], label=col.replace("_", " "))
    #     ax.set_xlim(0, 150)
    #     ax.xaxis.set_major_locator(MultipleLocator(30))
    #     ax.set_xticks(np.arange(0, 151, 30))
    #     ax.set_ylim(0, 4.5)
    #     ax.yaxis.set_major_locator(MultipleLocator(0.5))

    #     ax.set_title("训练 & 验证Loss变化曲线", fontsize=14, pad=15)
    #     ax.set_xlabel("Epoch", fontsize=12)
    #     ax.set_ylabel(type, fontsize=12)
    #     ax.legend(loc="upper right", fontsize=11)
    #     fig.savefig(
    #         os.path.join(out_dir, f"{type}.png"),
    #         dpi=300,
    #         bbox_inches="tight",
    #     )
    #     plt.clf()


if __name__ == "__main__":
    dirs = os.listdir("./checkpoints/")
    for i in dirs:
        line_chart(i)
        heatmap(i)
