# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
from pathlib import Path
import typing

import torch
from torch.utils.data.dataloader import DataLoader

from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from bounding_box import bounding_box as bb_draw
import cv2
import numpy as np
from tqdm import tqdm

from parse_config import strtobool
from model import pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset, BatchCollateFn
from data_utils.documents import read_gt_file_with_box_entity_type
from utils.util import iob_index_to_str, text_index_to_str, iob_tags_to_union_iob_tags, boxes_coordinates_from_mask
from utils.entities_list import Entities_list
from utils.class_utils import iob_labels_vocab_cls
from utils.metrics import SpanBasedF1MetricTracker


def format_ann_tuple(ann: tuple) -> dict:
    _, bb, transcript, entity = ann
    x1, y1, x2, y2, x3, y3, x4, y4 = bb
    X = [x1, x2, x3, x4]
    Y = [y1, y2, y3, y4]
    return {
        'entity': entity,
        'transcript': transcript,
        'xtl': min(X), 'ytl': min(Y),
        'xbr': max(X), 'ybr': max(Y)
    }


def main(args):
    if args.gpu and not torch.is_cuda_available():
        print('CUDA is not available. Falling back to CPU.')
        args.gpu = -1

    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved validation mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(data_root=args.data_root,
                               files_name=args.samples_list,
                               boxes_and_transcripts_folder=args.boxes_transcripts,
                               iob_tagging_type='box_level',
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn())

    # setup output path
    output_folder = Path(args.output_folder)
    output_dir_paths = {k: output_folder.joinpath(k) for k in ('predictions', 'errors')}
    for p in output_dir_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    f1_metric = SpanBasedF1MetricTracker(iob_labels_vocab_cls)
    confusion_matrix = np.zeros(
        (len(Entities_list), 1 + len(Entities_list)), dtype=int)
    num_erratic_samples = 0 if any((args.store_predictions, args.inspect_errors)) else None

    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            print(input_data_item.keys())

            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)


            mask = input_data_item['mask']
            predicted_tags = [
                path for path, score in
                pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            ]

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            gt_tags_list = iob_index_to_str(iob_tags_to_union_iob_tags(input_data_item['iob_tags_label'], mask))
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)


            for decoded_tags, decoded_texts, gt_tags, boxes_coord, fname in zip(
                decoded_tags_list, decoded_texts_list, gt_tags_list, input_data_item['boxes_coordinate'], input_data_item['filenames']
            ):
                spans = bio_tags_to_spans(decoded_tags, [])

                entities = []  # exists one to many case
                for box_coord, (pred_entity, range_tuple) in zip(boxes_coord, spans):
                    gt_span_tags = bio_tags_to_spans(gt_tags[range_tuple[0]:range_tuple[1] + 1])
                    gt_entity = gt_span_tags[0][0]

                    x1, y1, x2, y2, x3, y3, x4, y4 = box_coord.cpu().numpy().astype(int)
                    X = [x1, x2, x3, x4]
                    Y = [y1, y2, y3, y4]

                    entities.append({
                        'pred_entity': pred_entity,
                        'gt_entity': gt_entity,
                        'text': ''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]),
                        'coordinates': {
                            'xtl': min(X), 'ytl': min(Y),
                            'xbr': max(X), 'ybr': max(Y)
                        }
                    })

                if args.store_predictions:
                    result_file = output_dir_paths['predictions'] / (Path(fname).stem + '.txt')
                    with result_file.open(mode='w') as f:
                        for item in entities:
                            f.write('{}\t{}\t{}\n'.format(item['pred_entity'], item['gt_entity'], item['text']))

                if args.inspect_errors:
                    image = cv2.resize(
                        cv2.imread(fname, cv2.IMREAD_UNCHANGED),
                        test_dataset.resized_image_size, interpolation=cv2.INTER_LINEAR)




                    is_pred_erratic = False
                    to_match = set(Entities_list)
                    for item in entities:
                        confusion_matrix[Entities_list.index(item['gt_entity'])][Entities_list.index(item['pred_entity'])] += 1

                        box_coord_tuple = item['coordinates']['xtl'], item['coordinates']['ytl'],\
                            item['coordinates']['xbr'], item['coordinates']['ybr']

                        if item['pred_entity'] in to_match:
                            to_match.remove(item['pred_entity'])

                        if item['pred_entity'] != item['gt_entity']:
                            # MIS-MATCH: FALSE POSITIVE
                            is_pred_erratic = True
                            bb_draw.add(
                                image, *box_coord_tuple,
                                '{0} ({1})'.format(item['pred_entity'], item['gt_entity']), 'red')

                    for label in to_match:
                        # NO-MATCH: FALSE NEGATIVE
                        is_pred_erratic = True
                        confusion_matrix[Entities_list.index(label)][len(Entities_list)] += 1
                        for item in entities:
                            if item['gt_entity'] == label:
                                box_coord_tuple = item['coordinates']['xtl'], item['coordinates']['ytl'],\
                                    item['coordinates']['xbr'], item['coordinates']['ybr']
                                bb_draw.add(image, *box_coord_tuple, label, 'orange')

                if is_pred_erratic:
                    cv2.imwrite((output_dir_paths['errors'] / Path(fname).name).as_posix(), image)
                    num_erratic_samples += 1

            # metric validation
            if args.evaluate_model:
                # calculate and update f1 metrics
                predicted_tags_hard_prob = logits * 0
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        predicted_tags_hard_prob[i, j, tag_id] = 1

                f1_metric.update(
                    predicted_tags_hard_prob.long(),
                    iob_tags_to_union_iob_tags(input_data_item['iob_tags_label'], mask),
                    new_mask)

    print('Erratic samples ({:.2f}% - {} total) succesfully inspected at {}'.format(
        100 * num_erratic_samples / len(test_dataset), num_erratic_samples, output_dir_paths['errors']))

    f1_result = f1_metric.result()
    print(f'SpanBasedF1Measure: {f1_result}')

    with open(output_folder.joinpath('error_matrix.txt'), 'w') as fp:
        fp.write(repr(confusion_matrix))
    print(f'Confusion matrix stored ')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('-sl', '--samples_list', default=None, type=str,
                      help='Path to file specifying which files are part of test partition.')
    args.add_argument('-bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('--data_root', default=None, type=str)
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=0, type=int,
                      help='GPU id to use. (default: 0, main/first GPU)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args.add_argument('--store_predictions', default='true', type=strtobool,
                      help='Stores prediction results in output folder.')
    args.add_argument('--evaluate_model', default='true', type=strtobool,
                      help='Computes validation metrics on predictions.')
    args.add_argument('--inspect_errors', default='true', type=strtobool,
                      help='Plots confusion matrix and stores visual debug of erronoeous predictions.')
    args = args.parse_args()
    main(args)
