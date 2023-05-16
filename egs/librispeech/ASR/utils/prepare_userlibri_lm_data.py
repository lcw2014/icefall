import os
import sys
import json
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument(
        "--out-dir",
        type=str,
    )

args = parser.parse_args()
with open('/DB/UserLibri/lm_data/metadata.tsv','r') as f, open(f'{args.out_dir}/userlibri_ids.txt','w') as f2:
    rdr = pd.read_csv(f,delimiter='\t')
    ids = rdr['Book ID'].tolist()
    for id in ids:
        f2.write(f'{id} ')

for id in ids:
    with open(f'/DB/UserLibri/lm_data/{id}_lm_train.txt','r') as f1, open(f'{args.out_dir}/userlibri-lm-norm-{id}.txt','w') as f2:
        rawtext = f1.readlines()
        for txt in sorted(set(rawtext)):
            if len(txt.split()) > 1:
                tx = re.sub('[=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`…》\”\“\’·]','',txt)
                tx = re.sub('[0-9]','',tx)
                tx = re.sub('-',' ',tx)
                tx = ' '.join(tx.strip().split(' ')).strip()
                if len(tx) == 0:
                    print('0 word, skip')
                    continue
                f2.write(f'{tx}\n')
        