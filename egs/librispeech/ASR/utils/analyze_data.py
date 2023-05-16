#!/usr/bin/env python3

import csv

audio_list = list()
audio_book_pair = dict()
audio_book_pair2 = dict()
audio_book_list = set()
audio_book_list2 = set()
lm_list = set()

with open('/DB/UserLibri/audio_data/metadata.tsv', 'r') as f1, open('/DB/UserLibri/lm_data/metadata.tsv', 'r') as f2:
    reader1 = csv.reader(f1, delimiter='\t')
    querys1, co = list(), list()
    for r in reader1:
        querys1.append(r[0])
        co.append(r[1])
    co.pop(0)
    querys1.pop(0)

    reader2 = csv.reader(f2, delimiter='\t')
    querys2 = [r[0] for r in reader2]
    querys2.pop(0)
    lm_list = set(querys2)
    for i, q in enumerate(querys1):
        spk, bk = q.split('-')[1], q.split('-')[3]

        audio_book_pair.setdefault(spk,set())
        audio_book_pair[spk].add(f'test-{co[i]}_{bk}')
        audio_book_pair2.setdefault(spk,set())
        audio_book_pair2[spk].add(bk)
    print(len(lm_list))
    print(len(audio_book_pair.keys()),len(audio_book_pair.values()))
    for v in audio_book_pair.values():
        # print(v)
        for v2 in v:
            audio_book_list.add(v2)
    for v in audio_book_pair2.values():
        for v2 in v:
            audio_book_list2.add(v2)
    print(len(audio_book_list))
    # print(sorted(lm_list))
    # print(sorted(audio_book_list,key= lambda x: x.split('_')[1]))

with open('data/id_to_books.txt','w') as f1, open('data/userlibri_test_id.txt','w') as f2:
    audio_book_pair = {key: audio_book_pair[key] for key in sorted(audio_book_pair)}
    # print(audio_book_pair)
    
    for k in audio_book_pair.keys():
        # print(k)
        bks = ' '.join(list(audio_book_pair[k]))
        # print(bks)
        f1.write(f'{k}\t{bks}\n')
    audio_book_list = sorted(audio_book_list)
    for k in audio_book_list:
        f2.write(f'{k}\n')
        

with open('data/id_to_books2.txt','w') as f1, open('data/userlibri_test_id2.txt','w') as f2:
    audio_book_pair2 = {key: audio_book_pair2[key] for key in sorted(audio_book_pair2)}
    # print(audio_book_pair2)
    
    for k in audio_book_pair2.keys():
        bks = ' '.join(list(audio_book_pair2[k]))
        f1.write(f'{k}\t{bks}\n')
    audio_book_list2 = sorted(audio_book_list2)
    for k in audio_book_list2:
        f2.write(f'{k}\n')

with open('data/spk_id.txt','w') as f:
    for k in audio_book_pair2.keys():
        f.write(f'{k}\n')

with open('data/book_to_data.txt', 'w') as f:
    print(len(audio_book_list))
    for a in audio_book_list2:
        temp = f'{a}\t'
        for b in audio_book_list:
            bkid = b.split('_')[-1]
            if bkid == a:
                temp += f'{b} '
        temp = temp.strip()
        temp += f'\n'
        f.write(temp)
