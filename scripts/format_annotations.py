import os
import pandas as pd
import sys

sys.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.entities_list import Entities_list  # noqa:E402

input_anno_dirpath = 'data/rg-gfuhr/boxes_and_transcripts-raw'
output_anno_dirpath = 'data/rg-gfuhr/boxes_and_transcripts'

input_splits_dirpath = 'data/rg-gfuhr/data_splits-raw'
output_splits_dirpath = 'data/rg-gfuhr/data_splits'

os.makedirs(output_anno_dirpath, exist_ok=True)
os.makedirs(output_anno_dirpath + '-train', exist_ok=True)
os.makedirs(output_anno_dirpath + '-test', exist_ok=True)

unused_files = []
for fbname in os.listdir(input_anno_dirpath):
    ifpname = os.path.join(input_anno_dirpath, fbname)
    ofpname = os.path.join(output_anno_dirpath, fbname)

    annos = []
    texts = set()
    with open(ifpname) as ifp:
        for line in ifp:
            line = line.strip()
            if line and not os.path.isfile(ofpname):
                split = line.split(',')
                try:
                    assert len(split) == 11  # sample index (1) + bbox (8) + transcript (1) + entity (1)
                except AssertionError:
                    print('cant format annotation ', ifpname)
                    print(line)
                else:
                    entity_name = split[-1]
                    anno = split
                    anno[0] = Entities_list.index(entity_name)
                    annos.append(anno)

                    text = anno[-2]
                    if text != '-':
                        texts.add(text)

    # only formats non-empty files with non-blank texts
    ofpname_train, ofpname_test = os.path.join(output_anno_dirpath + '-train', fbname),\
        os.path.join(output_anno_dirpath + '-test', fbname)
    if annos and texts:
        with open(ofpname_train, 'w') as ofp_train,\
                open(ofpname_test, 'w') as ofp_test:
            for anno in annos:
                if anno[-1] != 'unknown':
                    ofp_train.write(','.join(map(str, anno)) + '\n')
                    ofp_test.write(','.join(map(str, anno[:-1])) + '\n')
    else:
        unused_files.append(fbname.split('.')[0])


os.makedirs(output_splits_dirpath, exist_ok=True)
for mode in ('train', 'val', 'test'):
    bname = mode + '_samples_list.csv'
    idf = pd.read_csv(
        os.path.join(input_splits_dirpath, bname),
        names=['sid', 'class', 'fname']
    )
    idf = idf[~idf['fname'].isin(unused_files)].to_csv(
        os.path.join(output_splits_dirpath, bname),
        index=False, header=False
    )
