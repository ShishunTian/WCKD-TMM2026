import matplotlib.pyplot as plt

def plot_scatter(predictions_1, ground_truth, dataset):
    # 绘制模型1的散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(ground_truth, predictions_1, color='blue', alpha=0.5)
    # plt.title(f'{model_name_1} Predictions vs Ground Truth (CSIQ)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'r--')
    plt.tight_layout()

    # # 保存模型1的散点图
    # scatter_img_path_1 = f"/data/hlz/Base_VIT/Scatter/Cross/{dataset}_scatter.png"
    # plt.savefig(scatter_img_path_1)
    # plt.show()

