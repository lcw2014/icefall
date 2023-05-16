#!/usr/bin/env python3

import csv

ids_bks = dict()
bks_ids = dict()
with open('data/userlibri_test_id.txt','r') as f:
    bks = [a.strip() for a in f.readlines()]


with open('/DB/UserLibri/audio_data/metadata.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    querys = [r[0] for r in reader]
    querys.pop(0)
    for x in querys: # x = 'speaker-260-book-3748'
        print(x)
        bks_ids.setdefault(x.split('-')[3],x.split('-')[1])

for bk in bks: # bk = 'test-clean_3748'
    bkid = bk.split('_')[1] # 3748
    ids_bks.setdefault(bks_ids[bkid], set()) # 260 : (3748)
    ids_bks[bks_ids[bkid]].add(bk)

with open('data/id_to_books.txt','w') as f2:
    for key in ids_bks.keys():
        bkids = ' '.join(list(ids_bks[key]))
        f2.write(f'{key}\t{bkids}\n')