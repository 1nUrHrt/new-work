import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
   # 读取混淆矩阵
    cm_df = pd.read_csv('./checkpoints/default/confusion_matrix.csv', index_col=0)

    # 转换为 numpy 数组
    cm = cm_df.values

    # 绘图
    plt.figure(figsize=(20, 18))
    plt.imshow(cm, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.title('Confusion Matrix (86 classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 可选：显示数字（但 86x86 会非常拥挤，建议关闭）
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=4)

    plt.tight_layout()
    plt.savefig('cm_matplotlib.png', dpi=150)
    plt.show()