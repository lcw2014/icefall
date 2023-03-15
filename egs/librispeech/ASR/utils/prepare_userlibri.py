import os
import sys
import json

fname = ["data/manifests/librispeech_supervisions_test-clean_temp.jsonl", "data/manifests/librispeech_supervisions_test-other_temp.jsonl"]
fname2 = ["data/manifests/userlibri_supervisions_test-clean.jsonl", "data/manifests/userlibri_supervisions_test-other.jsonl"]

dir_path = ["/DB/UserLibri/audio_data/test-clean", "/DB/UserLibri/audio_data/test-other"]
co = ['test-clean','test-other']
if os.path.isfile('data/manifests/book_ids.txt'):
    os.remove('data/manifests/book_ids.txt')
dset = dict()
book_sets = set()

for i, fn1 in enumerate(fname):
    text_dir = list()
    with open(fn1) as f1:
        tset = [dict(eval(l)) for l in f1]

        # userlibri 텍스트 파일 불러오기
        for (root, dirs, files) in os.walk(dir_path[i]):
            for file in files:
                if '.txt' in file:
                    file_path = os.path.join(root, file)
                    text_dir.append(file_path)
        ids = list()
        book_ids = list()
        dict_per_book = dict()

        # 불러온 텍스트 파일로부터 id 불러오기
        for dir in text_dir:
            book_id = dir.split('/')[-2].split('-')[-1]
            # print(book_id)
            with open(dir, encoding='utf-8', mode='r') as ftemp:
                temp = [txt.split()[0] for txt in ftemp.readlines()]
                ids.extend(temp)
                book_ids.append(book_id)
                # print(book_id)
                dict_per_book.setdefault(book_id, [])
                dict_per_book[book_id].append(temp)
        dset[i] = dict_per_book
        book_sets = book_sets.union(set(book_ids))

        # 일치하는 id만 추출 전체
        with open(fname2[i], encoding='utf-8', mode='w') as f2:
            for id in ids:
                for t in tset:
                    if t['id'] == id:
                        f2.write(json.dumps(t) + "\n")
                        break
        
        # 일치하는 id만 추출 book별로
        for book_id in dset[i].keys():
            with open(f'data/manifests/userlibri_supervisions_{co[i]}_{book_id}.jsonl','w') as bookjson:
                jn = f'data/manifests/userlibri_supervisions_{co[i]}_{book_id}.jsonl'
                for ids in dset[i][book_id]:
                    for id in ids:
                        # print(id)
                        for t in tset:
                            if t['id'] == id:
                                bookjson.write(json.dumps(t) + "\n")
                                break
            os.system(f'gzip {jn}')

with open('data/manifests/book_ids.txt','w') as bf:
    for bids in book_sets:
        bf.write(f'{bids} ')



fname = ["data/manifests/librispeech_recordings_test-clean_temp.jsonl", "data/manifests/librispeech_recordings_test-other_temp.jsonl"]
fname2 = ["data/manifests/userlibri_recordings_test-clean.jsonl", "data/manifests/userlibri_recordings_test-other.jsonl"]

dir_path = ["/DB/UserLibri/audio_data/test-clean", "/DB/UserLibri/audio_data/test-other"]
for i, fn1 in enumerate(fname):
    text_dir = list()
    with open(fn1) as f1:
        tset = [dict(eval(l)) for l in f1]
        # userlibri 텍스트 파일 불러오기
        for (root, dirs, files) in os.walk(dir_path[i]):
            for file in files:
                if '.txt' in file:
                    file_path = os.path.join(root, file)
                    text_dir.append(file_path)
        ids = list()
        # 불러온 텍스트 파일로부터 id 불러오기
        for dir in text_dir:
            with open(dir, encoding='utf-8', mode='r') as ftemp:
                temp = [txt.split()[0] for txt in ftemp.readlines()]
                ids.extend(temp)
        
        # 일치하는 id만 추출 전체
        with open(fname2[i], encoding='utf-8', mode='w') as f2:
            for id in ids:
                for t in tset:
                    if t['id'] == id:
                        f2.write(json.dumps(t) + "\n")
                        break
        
        # 일치하는 id만 추출 book별로
        for book_id in dset[i].keys():
            with open(f'data/manifests/userlibri_recordings_{co[i]}_{book_id}.jsonl','w') as bookjson:
                jn = f'data/manifests/userlibri_recordings_{co[i]}_{book_id}.jsonl'
                for ids in dset[i][book_id]:
                    for id in ids:
                        for t in tset:
                            if t['id'] == id:
                                # print(id)
                                bookjson.write(json.dumps(t) + "\n")
                                break
            os.system(f'gzip {jn}')