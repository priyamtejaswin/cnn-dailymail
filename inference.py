#!/usr/bin/env python
"""
created at: Mon 24 Aug 2020 04:35:36 AM EDT
created by: Priyam Tejaswin (ptejaswi)

Inference on test data.
"""


import pdb
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm

bart = BARTModel.from_pretrained(
    './checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/projects/metis1/users/ptejaswi/multistep-retrieve-summarize/models/bart.base/'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('cnn_dm/test.source') as source, open('cnn_dm/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in tqdm(source):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

