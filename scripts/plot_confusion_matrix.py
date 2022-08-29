import os
import sys
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.entities_list import Entities_list_values as entities  # noqa:E402

if len(sys.argv) < 3:
    raise RuntimeError(
        'Usage: <matrix_data_path> <output_path>'
    )


def plot_confusion_matrix(
    error_matrix: np.array,
    class_labels: list
):
    FONT_SIZES = {
        'title': 18,
        'axes': 14,
        'ticks': 5
    }

    perc_data = np.copy(error_matrix).astype(float)
    for i in range(perc_data.shape[0]):
        tot = np.sum(perc_data[i])
        if tot > 0.0:
            perc_data[i, :] *= 100.0 / tot

    ax = sns.heatmap(
        perc_data, vmin=0, vmax=100,
        annot=error_matrix.astype(int), fmt='d',
        cmap='Blues', cbar_kws={'ticks': [0, 50, 100], 'format': '%d%%'})

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth ')

    ax.title.set_fontsize(FONT_SIZES['title'])
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(FONT_SIZES['axes'])

    ax.xaxis.set_ticklabels(entities + ['MISS'], fontsize=FONT_SIZES['ticks'])
    ax.yaxis.set_ticklabels(entities, fontsize=FONT_SIZES['ticks'])

    return ax.get_figure()


if __name__ == '__main__':
    entities = set(entities)
    entities = sorted(entities)
    entities = list(map(
        lambda s: '\n'.join(s.split('_')), entities
    ))

    matrix_data_path = sys.argv[1]
    matrix_data = eval(Path(matrix_data_path).read_text().replace('array', 'np.array'))

    fig = plot_confusion_matrix(matrix_data, entities)
    fig.savefig(sys.argv[2], dpi=800)
