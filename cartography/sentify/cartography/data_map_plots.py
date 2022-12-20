import logging
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from seaborn import FacetGrid

from sentify.cartography.colormap import shifted_colormap
from sentify.cartography.data_utils import _normalize_by_epochs

logger = logging.getLogger(__name__)
sns.set(style='whitegrid', font_scale=1.6, context='paper')

X_METRIC = 'variability'
Y_METRIC = 'confidence'


def save_plot(
        fig: Union[Figure, FacetGrid],
        filename: Path,
        dpi: int = 300,
) -> None:
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")


def plot_change_data_map(
        df: pd.DataFrame,
        quarter_counts: dict[str, int],
        x_dim: str = 'var_diff',
        y_dim: str = 'conf_diff',
        hue_dim: str = 'corr_diff',
        x_label: str = 'variability change',
        y_label: str = 'confidence change',
        hue_label: str = 'correctness change',
        title: Optional[str] = None,
        xlim: Optional[list[float]] = None,
        ylim: Optional[list[float]] = None,
):
    y = df[y_dim].values
    x = df[x_dim].values
    c = df[hue_dim].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    div_palette = sns.diverging_palette(
        6,
        180,
        s=75,
        l=40,
        sep=1,
        as_cmap=True,
        center='light',
    )
    v_min = np.min(c)
    v_max = np.max(c)
    if hue_dim == 'corr_diff':
        div_palette = shifted_colormap(
            div_palette,
            start=0.0,
            midpoint=min(1.0, 1 - v_max / (v_max + abs(v_min))),
            stop=1.0,
            name='shrunk',
        )

    p1 = ax.scatter(
        x=x,
        y=y,
        c=c,
        s=2,
        cmap=div_palette if hue_dim == 'corr_diff' else 'coolwarm',
        vmin=v_min,
        vmax=v_max,
    )

    ax.axvline(0, c='grey', ls='--')
    ax.axhline(0, c='grey', ls='--')

    if quarter_counts['quarter_I'] > 0:
        plt.text(
            x=0.9,
            y=0.9,
            s='{:,}'.format(quarter_counts['quarter_I']),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontdict={'fontsize': 10},
            bbox=dict(boxstyle='round,pad=0.3', ec='grey', lw=1, fc='white'),
        )
    if quarter_counts['quarter_II'] > 0:
        plt.text(
            x=0.9,
            y=0.1,
            s='{:,}'.format(quarter_counts['quarter_II']),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontdict={'fontsize': 10},
            bbox=dict(boxstyle='round,pad=0.3', ec='grey', lw=1, fc='white'),
        )
    if quarter_counts['quarter_III'] > 0:
        plt.text(
            x=0.1,
            y=0.1,
            s='{:,}'.format(quarter_counts['quarter_III']),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontdict={'fontsize': 10},
            bbox=dict(boxstyle='round,pad=0.3', ec='grey', lw=1, fc='white'),
        )
    if quarter_counts['quarter_IV'] > 0:
        plt.text(
            x=0.1,
            y=0.9,
            s='{:,}'.format(quarter_counts['quarter_IV']),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontdict={'fontsize': 10},
            bbox=dict(boxstyle='round,pad=0.3', ec='grey', lw=1, fc='white'),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    assign_sign = lambda y, pos: ('%+g' if y > 0 else '%g') % y
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(assign_sign))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(assign_sign))

    if title:
        ax.set_title(title)

    if hue_dim == 'corr_diff':
        colorbar_fmt = lambda y, pos: ('%+.1f' if y > 0 else '%.1f') % y
        cbar = fig.colorbar(p1, label=hue_label, format=ticker.FuncFormatter(colorbar_fmt))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        cbar = fig.colorbar(p1, label=hue_label)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    return fig


def plot_data_map_pair(
        df_metrics1: pd.DataFrame,
        df_metrics2: pd.DataFrame,
        hue_metric: str = 'correctness',
        max_instances_to_plot: int = 55000,
        sample_seed: int = 2022,
) -> FacetGrid:
    df_metrics1 = _sample_dataframe(
        df=df_metrics1,
        max_length=max_instances_to_plot,
        seed=sample_seed,
    )
    df_metrics2 = _sample_dataframe(
        df=df_metrics2,
        max_length=max_instances_to_plot,
        seed=sample_seed,
    )
    df_metrics2 = _normalize_by_epochs(df=df_metrics2, num_epochs=df_metrics2['epochs'][0])
    df_metrics1 = _normalize_by_epochs(df=df_metrics1, num_epochs=df_metrics1['epochs'][0])

    df_metrics = pd.concat([df_metrics1, df_metrics2])

    hue_sorted = sorted(df_metrics[hue_metric].unique().tolist(), reverse=True)
    num_hues = len(hue_sorted)
    style = hue_metric if num_hues < 8 else None

    palette_div = sns.diverging_palette(
        h_neg=260,
        h_pos=15,
        n=num_hues,
        sep=10,
        center='dark',
    )

    g = sns.FacetGrid(
        data=df_metrics,
        col='model',
        sharey=True,
        sharex=True,
        height=5,
    )
    g.map_dataframe(
        sns.scatterplot,
        x=X_METRIC,
        y=Y_METRIC,
        hue=hue_metric,
        hue_order=hue_sorted,
        palette=palette_div,
        style=style,
        style_order=hue_sorted,
        s=10,
    )

    # annotate regions
    bb = lambda c: dict(boxstyle='round,pad=0.2', ec=c, lw=1, fc='white')
    func_annotate = lambda text, xyc, bbc, ax: ax.annotate(
        text,
        xy=xyc,
        xycoords='axes fraction',
        fontsize=11,
        color='black',
        va='center',
        ha='center',
        rotation=350,
        bbox=bb(bbc),
    )
    for ax in g.axes.ravel():
        func_annotate('ambiguous', xyc=(0.9, 0.5), bbc='black', ax=ax)
        func_annotate('easy-to-learn', xyc=(0.27, 0.85), bbc='blue', ax=ax)
        func_annotate('hard-to-learn', xyc=(0.25, 0.2), bbc='red', ax=ax)

    g.add_legend(title=hue_metric, loc='center right', borderaxespad=1.5)
    plt.setp(g._legend.get_title(), fontsize='small')
    plt.setp(g._legend.get_texts(), fontsize='small')

    return g


def plot_data_map(
        df_metrics: pd.DataFrame,
        hue_metric: str = 'correctness',
        max_instances_to_plot: int = 50000,
        sample_seed: int = 2022,
        figsize: tuple = (7, 6),
        add_legend: bool = True,
        x_label: str = 'variability',
        y_label: str = 'confidence',
        xlim: Optional[list[float]] = None,
        ylim: Optional[list[float]] = None,
) -> Figure:
    df_metrics = _sample_dataframe(
        df=df_metrics,
        max_length=max_instances_to_plot,
        seed=sample_seed,
    )
    df_metrics = _normalize_by_epochs(df=df_metrics, num_epochs=df_metrics['epochs'][0])

    hue_sorted = sorted(df_metrics[hue_metric].unique().tolist(), reverse=True)
    num_hues = len(hue_sorted)
    style = hue_metric if num_hues < 8 else None

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    palette_div = sns.diverging_palette(
        h_neg=260,
        h_pos=15,
        n=num_hues,
        sep=10,
        center='dark',
    )
    plot = sns.scatterplot(
        x=X_METRIC,
        y=Y_METRIC,
        ax=ax0,
        data=df_metrics,
        hue=hue_metric,
        hue_order=hue_sorted,
        palette=palette_div,
        style=style,
        s=30,
        legend=add_legend,
    )

    # annotate regions
    bb = lambda c: dict(boxstyle='round,pad=0.3', ec=c, lw=1, fc='white')
    func_annotate = lambda text, xyc, bbc: ax0.annotate(
        text,
        xy=xyc,
        xycoords='axes fraction',
        fontsize=15,
        color='black',
        va='center',
        ha='center',
        rotation=350,
        bbox=bb(bbc),
    )
    func_annotate('ambiguous', xyc=(0.9, 0.5), bbc='black')
    func_annotate('easy-to-learn', xyc=(0.27, 0.85), bbc='blue')
    func_annotate('hard-to-learn', xyc=(0.27, 0.2), bbc='red')

    if add_legend:
        legend = plot.legend(
            loc='upper left',
            title=hue_metric,
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False,
        )
        plt.setp(legend.get_title(), fontsize='x-small')
        plt.setp(legend.get_texts(), fontsize='x-small')

    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    return fig


def plot_data_map_with_hist(
        df_metrics: pd.DataFrame,
        hue_metric: str = 'correctness',
        title: str = '',
        model: str = 'RoBERTa',
        max_instances_to_plot: int = 55000,
        sample_seed: int = 2022,
        figsize: tuple = (14, 10),
) -> Figure:
    df_metrics = _sample_dataframe(
        df=df_metrics,
        max_length=max_instances_to_plot,
        seed=sample_seed,
    )
    df_metrics = _normalize_by_epochs(df=df_metrics, num_epochs=df_metrics['epochs'][0])

    hue_sorted = sorted(df_metrics[hue_metric].unique().tolist())
    num_hues = len(hue_sorted)
    style = hue_metric if num_hues < 8 else None

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(3, 2, width_ratios=[5, 1])
    ax0 = fig.add_subplot(grid[:, 0])

    # Choose a palette
    palette_div = sns.diverging_palette(
        h_neg=260,
        h_pos=15,
        n=num_hues,
        sep=10,
        center='dark',
    )

    # Make the scatterplot
    plot = sns.scatterplot(
        x=X_METRIC,
        y=Y_METRIC,
        ax=ax0,
        data=df_metrics,
        hue=hue_metric,
        palette=palette_div,
        style=style,
        s=30,
        hue_order=reversed(hue_sorted),
    )

    # annotate regions
    bb = lambda c: dict(boxstyle='round,pad=0.3', ec=c, lw=2, fc='white')
    func_annotate = lambda text, xyc, bbc: ax0.annotate(
        text,
        xy=xyc,
        xycoords='axes fraction',
        fontsize=15,
        color='black',
        va='center',
        ha='center',
        rotation=350,
        bbox=bb(bbc),
    )
    an1 = func_annotate('ambiguous', xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate('easy-to-learn', xyc=(0.27, 0.85), bbc='blue')
    an3 = func_annotate('hard-to-learn', xyc=(0.35, 0.25), bbc='red')

    legend = plot.legend(fancybox=True, shadow=False, ncol=1, title=hue_metric)
    plt.setp(legend.get_title(), fontsize='small')
    plt.setp(legend.get_texts(), fontsize='small')

    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    plot.set_title(f'{title}-{model} Data Map', fontsize=17)

    # Make the histograms.
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 1])
    ax3 = fig.add_subplot(grid[2, 1])

    # confidence subplot
    plott0 = df_metrics.hist(column=['confidence'], ax=ax1, color='#622a87')
    plott0[0].set_title('')
    plott0[0].set_xlabel('confidence')
    plott0[0].set_ylabel('density')

    # variability subplot
    plott1 = df_metrics.hist(column=['variability'], ax=ax2, color='teal')
    plott1[0].set_title('')
    plott1[0].set_xlabel('variability')
    plott1[0].set_ylabel('density')

    # corectness subplot
    plot2 = sns.countplot(
        x="correctness",
        data=df_metrics,
        ax=ax3,
        color='#86bf91',
        order=hue_sorted,
    )
    ax3.xaxis.grid(True)  # Show the vertical gridlines

    plot2.set_title('')
    plot2.set_xlabel('correctness')
    plot2.set_ylabel('density')

    return fig


def _sample_dataframe(
        df: pd.DataFrame,
        max_length: int = 50000,
        seed: int = 2022,
) -> pd.DataFrame:
    if len(df) > max_length:
        df = df.sample(
            n=max_length,
            random_state=seed,
        )
    return df
