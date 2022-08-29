# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/9/2020 9:16 PM
import glob
import os
from typing import Tuple, List
from pathlib import Path
import warnings
import random
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from . import documents
from .documents import Document
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls
from utils import is_none_fn


class PICKDataset(Dataset):
    def __init__(
        self,
        data_root: str = None,
        files_name: str = None,
        boxes_and_transcripts_folder: str = 'boxes_and_transcripts',
        images_folder: str = 'images',
        iob_tagging_type: str = 'box_and_within_box_level',
        resized_image_size: Tuple[int, int] = (720, 480),
        keep_ratio: bool = True,
        ignore_error: bool = False,
        training: bool = True
    ):
        '''

        :param files_name: containing training and validation samples list file.
        TODO: update signature
        :param boxes_and_transcripts_folder: gt or ocr result containing transcripts, boxes and box entity type (optional).
        :param images_folder: whole images file folder
        :param entities_folder: exactly entity type and entity value of documents, containing json format file
        :param iob_tagging_type: 'box_level', 'document_level', 'box_and_within_box_level'
        :param resized_image_size: resize whole image size, (w, h)
        :param keep_ratio: TODO implement this parames
        :param ignore_error:
        :param training: True for train and validation mode, False for test mode. True will also load labels,
        and files_name and entities_file must be set.
        '''
        super().__init__()
        self._image_ext = None
        self._ann_ext = None
        self.iob_tagging_type = iob_tagging_type
        self.keep_ratio = keep_ratio
        self.ignore_error = ignore_error
        self.training = training
        assert resized_image_size and len(resized_image_size) == 2, 'resized image size not be set.'
        self.resized_image_size = tuple(resized_image_size)  # (w, h)


        if all(map(is_none_fn, (files_name, data_root))):
            raise ValueError('`data_root` should be explicitly specified or via `files_name`.')
        if all(map(is_none_fn, (files_name, boxes_and_transcripts_folder))):
            raise ValueError('`files_name` should be explicitly specified or via `boxes_and_transcripts`.')
        if training and files_name is None:
            raise ValueError('`files_name` CSV should be explicitly specified when training.')

        self.data_root: Path = Path(data_root) if data_root else Path(files_name).parent.parent
        self.boxes_and_transcripts_folder: Path = self.data_root.joinpath('boxes_and_transcripts')
        self.images_folder: Path = self.data_root.joinpath('images')

        if self.training:
            self.partition_files_list: pd.DataFrame = pd.read_csv(Path(files_name).as_posix(), header=None,
                names=['index', 'document_class', 'file_name'],
                dtype={'index': int, 'document_class': str, 'file_name': str}
            ) if files_name else None
            self.files_list = self.partition_files_list['file_name'].tolist()
        else:
            self.files_list = list(self.boxes_and_transcripts_folder.glob('*.tsv'))

        if self.iob_tagging_type != 'box_level' and not self.entities_folder.exists():
            raise FileNotFoundError(f'Entity folder {self.entities_folder} is not exist!')

        if not (self.boxes_and_transcripts_folder.exists() and self.images_folder.exists()):
            raise FileNotFoundError('Not contain boxes_and_transcripts floader {} or images folder {}.'
                                    .format(self.boxes_and_transcripts_folder.as_posix(),
                                            self.images_folder.as_posix()))

    def __len__(self):
        return len(self.files_list)

    def get_image_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._image_ext is None:
            filename = list(self.images_folder.glob(f'**/{basename}.*'))[0]
            self._image_ext = os.path.splitext(filename)[1]

        return self.images_folder.joinpath(basename + self._image_ext)

    def get_ann_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._ann_ext is None:
            filename = list(self.boxes_and_transcripts_folder.glob(f'**/{basename}.*'))[0]
            self._ann_ext = os.path.splitext(filename)[1]

        return self.boxes_and_transcripts_folder.joinpath(basename + self._ann_ext)

    @overrides
    def __getitem__(self, index):

        boxes_and_transcripts_file = self.get_ann_file(Path(self.files_list[index]).stem)
        image_file = self.get_image_file(Path(self.files_list[index]).stem)
        if not boxes_and_transcripts_file.exists() or not image_file.exists():
            if self.ignore_error and self.training:
                warnings.warn('{} is not exist. get a new one.'.format(boxes_and_transcripts_file))
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Sample: {} not exist.'.format(boxes_and_transcripts_file.stem))

        try:
            # TODO add read and save cache function, to speed up data loaders
            document = documents.Document(
                boxes_and_transcripts_file, image_file, self.resized_image_size,
                iob_tagging_type=self.iob_tagging_type, image_index=index)
            return document
        except Exception as e:
            if self.ignore_error:
                warnings.warn('loading samples is occurring error, try to regenerate a new one.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Error occurs in image {}: {}'.format(boxes_and_transcripts_file.stem, e.args)) from e


class BatchCollateFn(object):
    '''
    padding input (List[Example]) with same shape, then convert it to batch input.
    '''

    def __init__(self):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, batch_list: List[Document]):
        # dynamic calculate max boxes number of batch,
        # this is suitable to one gpus or multi-nodes multi-gpus trianing mode, due to pytorch distributed training strategy.
        max_boxes_num_batch = max([x.boxes_num for x in batch_list])
        max_transcript_len = max([x.transcript_len for x in batch_list])

        # fix MAX_BOXES_NUM and MAX_TRANSCRIPT_LEN. this ensures batch has same shape, but lead to waste memory and slow speed..
        # this is suitable to one nodes multi gpus training mode, due to pytorch DataParallel training strategy
        # max_boxes_num_batch = documents.MAX_BOXES_NUM
        # max_transcript_len = documents.MAX_TRANSCRIPT_LEN

        ''' padding every sample with same shape, then construct batch_list samples  '''
        # whole image, B, C, H, W
        image_batch_tensor = torch.stack([self.trsfm(x.whole_image) for x in batch_list], dim=0).float()

        # relation features, (B, num_boxes, num_boxes, 6)
        relation_features_padded_list = [F.pad(torch.FloatTensor(x.relation_features),
                                               (0, 0, 0, max_boxes_num_batch - x.boxes_num,
                                                0, max_boxes_num_batch - x.boxes_num))
                                         for i, x in enumerate(batch_list)]
        relation_features_batch_tensor = torch.stack(relation_features_padded_list, dim=0)

        # boxes coordinates,  (B, num_boxes, 8)
        boxes_coordinate_padded_list = [F.pad(torch.FloatTensor(x.boxes_coordinate),
                                              (0, 0, 0, max_boxes_num_batch - x.boxes_num))
                                        for i, x in enumerate(batch_list)]
        boxes_coordinate_batch_tensor = torch.stack(boxes_coordinate_padded_list, dim=0)

        # text segments (B, num_boxes, T)
        text_segments_padded_list = [F.pad(torch.LongTensor(x.text_segments[0]),
                                           (0, max_transcript_len - x.transcript_len,
                                            0, max_boxes_num_batch - x.boxes_num),
                                           value=keys_vocab_cls.stoi['<pad>'])
                                     for i, x in enumerate(batch_list)]
        text_segments_batch_tensor = torch.stack(text_segments_padded_list, dim=0)

        # text length (B, num_boxes)
        text_length_padded_list = [F.pad(torch.LongTensor(x.text_segments[1]),
                                         (0, max_boxes_num_batch - x.boxes_num))
                                   for i, x in enumerate(batch_list)]
        text_length_batch_tensor = torch.stack(text_length_padded_list, dim=0)

        # text mask, (B, num_boxes, T)
        mask_padded_list = [F.pad(torch.ByteTensor(x.mask),
                                  (0, max_transcript_len - x.transcript_len,
                                   0, max_boxes_num_batch - x.boxes_num))
                            for i, x in enumerate(batch_list)]
        mask_batch_tensor = torch.stack(mask_padded_list, dim=0)

        # iob tag label for input text, (B, num_boxes, T)
        iob_tags_label_padded_list = [
            F.pad(torch.LongTensor(x.iob_tags_label), (
                0, max_transcript_len - x.transcript_len,
                0, max_boxes_num_batch - x.boxes_num
            ), value=iob_labels_vocab_cls.stoi['<pad>'])
            for i, x in enumerate(batch_list)]
        iob_tags_label_batch_tensor = torch.stack(iob_tags_label_padded_list, dim=0)

        # (B,)
        image_indexs_tensor = torch.tensor([x.image_index for x in batch_list])

        # For easier debug.
        filenames = [doc.image_filename for doc in batch_list]

        # Convert the data into dict.
        batch = dict(
            whole_image=image_batch_tensor,
            relation_features=relation_features_batch_tensor,
            text_segments=text_segments_batch_tensor,
            text_length=text_length_batch_tensor,
            boxes_coordinate=boxes_coordinate_batch_tensor,
            mask=mask_batch_tensor,
            iob_tags_label=iob_tags_label_batch_tensor,
            image_indexs=image_indexs_tensor,
            filenames=filenames)

        return batch
