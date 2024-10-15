import scipy.io
import numpy as np
from sklearn.metrics import cohen_kappa_score

def calculate_metrics(true_labels, pred_labels):
    total_samples = np.sum(true_labels != 0)
    correct_samples = np.sum((true_labels == pred_labels) & (true_labels != 0))
    overall_accuracy = correct_samples / total_samples

    class_accuracy = np.zeros(np.max(true_labels) + 1)
    for i in range(1, np.max(true_labels) + 1):
        class_samples = np.sum(true_labels == i)
        class_correct_samples = np.sum((true_labels == i) & (pred_labels == i))
        if class_samples != 0:
            class_accuracy[i] = class_correct_samples / class_samples
        else:
            class_accuracy[i] = 0

    true_labels_flatten = true_labels.flatten()
    pred_labels_flatten = pred_labels.flatten()

    nonzero_indices = np.where((true_labels_flatten != 0) & (pred_labels_flatten != 0))

    true_labels_nonzero = true_labels_flatten[nonzero_indices]
    pred_labels_nonzero = pred_labels_flatten[nonzero_indices]


    kappa = cohen_kappa_score(true_labels_nonzero, pred_labels_nonzero)

    return overall_accuracy, class_accuracy, kappa


true_labels = scipy.io.loadmat('data/PaviaU/PaviaU_gt.mat')['paviaU_gt']


num_files = 3
predicted_labels = []
for i in range(num_files):
    file_path = f'Maxvoting/Paviau/5/{i}_final_prediction.mat'
    pred_labels = scipy.io.loadmat(file_path)['pred']
    predicted_labels.append(pred_labels)

votes = np.zeros((predicted_labels[0].shape[0], predicted_labels[0].shape[1], np.max(predicted_labels[0])))  # 创建用于记录投票结果的三维数组
for i in range(len(predicted_labels)):
    for j in range(predicted_labels[i].shape[0]):
        for k in range(predicted_labels[i].shape[1]):
            label = predicted_labels[i][j, k]
            if label != 0:
                votes[j, k, label - 1] += 1


final_pred_labels = np.argmax(votes, axis=2) + 1

oa, aa, kappa = calculate_metrics(true_labels, final_pred_labels)

overall_aa = np.mean(aa[1:])


print("总体准确率（OA）： {:.2f}%".format(oa * 100))
print("分类准确率（AA）：")
for i in range(1, np.max(true_labels) + 1):
    print("类别", i, "的准确率： {:.2f}%".format(aa[i] * 100))
print("AA： {:.2f}%".format(overall_aa * 100))
print("Kappa系数： {:.2f}".format(kappa))
