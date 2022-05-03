import os
import sys

from bounding_box import bounding_box as bb_draw
import cv2
import numpy as np

sys.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.entities_list import Entities_list as entities  # noqa:E402

if len(sys.argv) < 3:
    raise RuntimeError(
        'Usage: <predictions_dpath> <data_dpath>'
    )

predictions_dpath = sys.argv[1]
data_dpath = sys.argv[2]
images_dpath = os.path.join(data_dpath, 'images')
anno_dpath = os.path.join(data_dpath, 'boxes_and_transcripts-train')


def read_pred_file(fname):
    pred = []
    with open(fname) as fp:
        for line in fp:
            entity, transcript = map(
                lambda s: s.strip(), line.split('\t'))
            if entity != 'unknown':
                pred.append({
                    'entity': entity,
                    'transcript': transcript
                })
    return pred


def read_anno_file(fname):
    anno = []
    with open(fname) as fp:
        for line in fp:
            _, x1, y1, x2, y2, x3, y3, x4, y4, transcript, entity = map(
                lambda s: s.strip(), line.split(','))
            if entity != 'unknown':
                X = [x1, x2, x3, x4]
                Y = [y1, y2, y3, y4]
                anno.append({
                    'entity': entity,
                    'transcript': transcript,
                    'xtl': min(X), 'ytl': min(Y),
                    'xbr': max(X), 'ybr': max(Y)
                })
    return anno


def anno_box2bbcoord(anno):
    return anno['xtl'], anno['ytl'], anno['xbr'], anno['ybr']


entities = set(entities)
entities.remove('unknown')
entities = sorted(entities)
print('Entities: ', entities)

confusion_matrix = np.zeros(
    (len(entities), 1 + len(entities)), dtype=float)

output_dpath = os.path.join(data_dpath, 'inspection_output')
output_images_dpath = os.path.join(output_dpath, 'images')
os.makedirs(output_images_dpath, exist_ok=True)

num_erratic_samples = 0
for pred_fbname in os.listdir(predictions_dpath):
    pred_fname = os.path.join(predictions_dpath, pred_fbname)  # .txt
    image_fname = os.path.join(  # .png
        images_dpath,
        pred_fbname.replace('.txt', '.png')
    )
    anno_fname = os.path.join(  # .tsv
        anno_dpath,
        pred_fbname.replace('.txt', '.tsv')
    )
    out_fname = os.path.join(  # .jpg
        output_images_dpath,
        pred_fbname.replace('.txt', '.jpg')
    )

    image = cv2.imread(image_fname, cv2.IMREAD_COLOR)
    pred = read_pred_file(pred_fname)
    anno = read_anno_file(anno_fname)

    is_erratic = False
    for box_anno in anno:
        # print(box_anno['transcript'], 'is of entity', box_anno['entity'])
        found_anno = False
        for box_pred in pred:
            if box_anno['transcript'] == box_pred['transcript']:
                found_anno = True
                if box_anno['entity'] == box_pred['entity']:
                    # MATCH: TRUE POSITIVE
                    confusion_matrix[entities.index(box_anno['entity'])][entities.index(box_pred['entity'])] += 1
                else:
                    # MIS-MATCH: FALSE POSITIVE
                    is_erratic = True
                    confusion_matrix[entities.index(box_anno['entity'])][entities.index(box_pred['entity'])] += 1
                    bb_draw.add(image, *anno_box2bbcoord(box_anno),
                                '{0} ({1})'.format(box_pred['entity'], box_anno['entity']), 'red')
                # break

        if found_anno is False:
            # NO-MATCH: FALSE NEGATIVE
            is_erratic = True
            confusion_matrix[entities.index(box_anno['entity'])][len(entities)] += 1
            bb_draw.add(image, *anno_box2bbcoord(box_anno), box_anno['entity'], 'orange')

    if is_erratic:
        cv2.imwrite(out_fname, image)
        num_erratic_samples += 1
        # break
print('Erratic samples ({0} total) succesfully inspected at {1}'.format(num_erratic_samples, output_images_dpath))

for i in range(len(entities)):
    confusion_matrix[i] /= np.sum(confusion_matrix[i])
print(confusion_matrix)

cm_fname = os.path.join(output_dpath, 'confusion_matrix.npy')
np.save(cm_fname, confusion_matrix)

np.testing.assert_array_equal(confusion_matrix, np.load(cm_fname))
print('Confusion matrix succesfully stored at {}'.format(cm_fname))
