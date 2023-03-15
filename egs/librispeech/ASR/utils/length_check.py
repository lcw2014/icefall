fname = '/home/Workspace/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz'

import numpy as np
import json
import gzip
import sys
{"id": "4051-11218-0000-0", "start": 0, "duration": 3.965, "channel": 0, "supervisions": [{"id": "4051-11218-0000", "recording_id": "4051-11218-0000", "start": 0.0, "duration": 3.965, "channel": 0, "text": "GREATLY ENCOURAGED AT FINDING HIMSELF NOT YET TURNED INTO A CINDER", "language": "English", "speaker": "4051"}], "features": {"type": "kaldi-fbank", "num_frames": 397, "num_features": 80, "frame_shift": 0.01, "sampling_rate": 16000, "start": 0, "duration": 3.965, "storage_type": "lilcom_chunky", "storage_path": "data/fbank/librispeech_feats_train-clean-100/feats-0.lca", "storage_key": "0,36133", "channels": 0}, "recording": {"id": "4051-11218-0000", "sources": [{"type": "file", "channels": [0], "source": "/home/Workspace/icefall/egs/librispeech/ASR/download/LibriSpeech/train-clean-100/4051/11218/4051-11218-0000.flac"}], "sampling_rate": 16000, "num_samples": 63440, "duration": 3.965, "channel_ids": [0]}, "type": "MonoCut"}

with gzip.open(fname, 'r') as f1:
    # for f in f1:
    #     print(dict(json.loads(f)))
    #     sys.exit()
    tset = [dict(json.loads(l)) for l in f1]
    tlen = []
    flen = []
    portion = []
    for t in tset:
        tlen.append(len(t['supervisions'][0]['text'].split()))
        flen.append(int(t['features']["num_frames"]))
    tlen = np.array(tlen)
    flen = np.array(flen)
    for t,f in zip(tlen,flen):
        portion.append(f/t)
    portion = np.array(portion)
print(tlen.min(),tlen.max()) # 1 87
print(flen.min(),flen.max()) # 75 3304
print(portion.min(),portion.max()) # 17.2 721.0
print(len(tlen),len(flen),len(portion)) # 843723

# i = 0
# a = 3305

# while 1:
#     if a % 2:
#         a = (a+1) // 2
#     else:
#         a = a//2
#     print(a)
#     i+=1
#     if a==1:
#         print(i)
#         break

# temp_list = [3304, 2512, 582, 75, ]