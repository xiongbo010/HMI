"""Implements mean average precision using sklearn"""
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import torch

def mean_average_precision(predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
    if not predictions.is_cuda:
        predictions = predictions.clone()
    labels, scores = gold_labels.cpu().detach().numpy(),predictions.cpu().detach().numpy()
    mAP = 0
    for idx,(single_example_labels, single_example_scores) in enumerate(zip(labels, scores)):
        avg_precision = average_precision_score(
            single_example_labels, single_example_scores
        )
        mAP += avg_precision
    mAP /= (idx+1)
    return mAP

def f1(predictions: torch.Tensor, gold_labels: torch.Tensor, threshold=0.5) -> None:  # type: ignore
    if not predictions.is_cuda:
        predictions = predictions.clone()
    labels, scores = gold_labels.cpu().detach().numpy(),predictions.cpu().detach().numpy()
    scores[scores < threshold] = 0
    scores[scores >= threshold] = 1

    f1 = 0
    for idx,(single_example_labels, single_example_scores) in enumerate(zip(labels, scores)):
        sample_f1 = f1_score(single_example_labels, single_example_scores)
        f1 += sample_f1
    f1 /= (idx+1)
    return f1


def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean().item()
        precision.append(p*100/batch_size)
    return precision

