from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm


def compute_train_dy_metrics(
    training_dynamics: dict[int, dict],
    variability_include_ci: bool = True,
    loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(),
    gold_label: str = 'gold',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coordinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and
     Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    records = []

    for idx, guid in tqdm(
        enumerate(training_dynamics),
        desc="Computing metrics for training dynamics",
    ):
        correctness_trend = []
        true_probs_trend = []

        sample = training_dynamics[guid]
        class_id = int(sample[gold_label])
        for i, epoch_logits in enumerate(sample["logits"]):
            probs = torch.nn.functional.softmax(Tensor(epoch_logits), dim=-1)
            true_probs_trend.append(float(probs[class_id]))

            is_correct = (np.argmax(epoch_logits) == class_id).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(class_id)

        confidence = compute_confidence(true_probs_trend)
        records.append(
            {
                'index': idx,
                'guid': guid,
                'confidence': confidence,
                'variability': compute_variability(
                    true_probs_trend,
                    include_ci=variability_include_ci,
                ),
                'correctness': compute_correctness(correctness_trend),
                'threshold_closeness': compute_threshold_closeness(confidence),
                'forgetfulness': compute_forgetfulness(correctness_trend),
                'epochs': num_tot_epochs,
            }
        )

    df = pd.DataFrame.from_records(records)

    df_train = pd.DataFrame(
        [
            [
                i,
                loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item()
                / len(training_dynamics),
                training_accuracy[i] / len(training_dynamics),
            ]
            for i in range(num_tot_epochs)
        ],
        columns=['epoch', 'loss', 'train_acc'],
    )
    return df, df_train


def compute_threshold_closeness(confidence: np.ndarray) -> np.ndarray:
    return confidence * (1 - confidence)


def compute_confidence(true_probs_trend: list[float]) -> np.ndarray:
    return np.mean(true_probs_trend)


def compute_variability(
    probs_trend: list[float],
    include_ci: bool = False,
) -> np.ndarray:
    if include_ci:
        return np.sqrt(
            np.var(probs_trend) + np.var(probs_trend) * np.var(probs_trend) / (len(probs_trend) - 1)
        )
    else:
        return np.std(probs_trend)


def compute_correctness(trend: list[float]) -> float:
    """
    Aggregate #times an example is predicted correctly during all training epochs.
    """
    return sum(trend)


def compute_forgetfulness(correctness_trend: list[float]) -> int:
    """
    Given an epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """
    if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
        return 1000

    learnt = False  # Predicted correctly in the current epoch.
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # nothing changed
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten
