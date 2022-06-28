"""
This code is adapted version of: https://github.com/allenai/cartography
"""
from typing import List
from collections import defaultdict


import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_forgetfulness(correctness_trend: List[float]) -> int:
    """
    Given a epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """
    if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
        return 1000
    learnt = False  # Predicted correctly in the current epoch.
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # nothing changed.
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten


def compute_correctness(trend: List[float]) -> float:
    """
    Aggregate #times an example is predicted correctly during all training epochs.
    """
    return sum(trend)


def compute_metrics(training_dynamics, num_epochs: int):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coorodinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and
    Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}

    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    # if args.include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
    #     variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()

    # print(list(training_dynamics.values())[0]["logits"])
    # raise None
    # print(num_tot_epochs)

    logits = {i: [] for i in range(num_epochs)}
    targets = {i: [] for i in range(num_epochs)}
    training_accuracy = defaultdict(float)

    for guid in tqdm.tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[int(record["gold"])])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])
            
        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])


    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness',]
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)
    
    df_train = pd.DataFrame([[i,
                                loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                                training_accuracy[i] / len(training_dynamics)
                                ] for i in range(num_epochs)],
                            columns=['epoch', 'loss', 'train_acc'])

    return df, df_train


def compute_avg_metrics(metrics: List[pd.DataFrame]) -> pd.DataFrame:
    metrics_cat = pd.concat(metrics, ignore_index=True)
    metrics_avg = metrics_cat.groupby('guid').mean().reset_index()
    return metrics_avg


def write_filtered_data(metrics: pd.DataFrame, save_path: str, sorted_by: str = 'variability', take_size: float = 0.3):
    """Create filter dataframe for data

    Args:
        metrics (pd.DataFrame): _description_
        first_group (str, optional): _description_. Defaults to 'ambigous'.
        take_size (float, optional): _description_. Defaults to 0.3.
    """
    sorted_scores = metrics.sort_values(by=[sorted_by], ascending=False)
    
    selected = sorted_scores.head(n=int(len(sorted_scores) * take_size))
    
    selected[['text_id', 'annotator_id']] = selected['guid'].str.split('_', expand=True)
    
    selected[['text_id', 'annotator_id']].to_csv(save_path, index=False)
    
    

def plot_data_map(dataframe: pd.DataFrame,
                  save_path: str, 
                  hue_metric: str = 'correct.',
                  save_name: str = '',
                  show_hist: bool = False,
                  max_instances_to_plot = 55000) -> None:
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    
    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        plot.set_title(f"{save_name} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{save_path}/{save_name}.pdf' # if show_hist else f'figures/compact_{title}_{model}.pdf'
    fig.savefig(filename, dpi=300)
