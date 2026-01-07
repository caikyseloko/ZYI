import json
import re
from collections import Counter
import torch

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\sáãâàéêíóôõúüç-]', '', text)
    toks = text.strip().split()
    return toks

class Vocab:
    def __init__(self, specials=['<pad>','<unk>','<bos>','<eos>']):
        self.stoi = {}
        self.itos = []
        for s in specials:
            self.add_token(s)
        self.pad = self.stoi['<pad>']; self.unk = self.stoi['<unk>']
        self.bos = self.stoi['<bos>']; self.eos = self.stoi['<eos>']

    def add_token(self, tok):
        if tok in self.stoi: return self.stoi[tok]
        idx = len(self.itos)
        self.itos.append(tok)
        self.stoi[tok] = idx
        return idx

    def build_from_corpus(self, corpus, min_freq=2, max_size=20000):
        cnt = Counter()
        for s in corpus:
            toks = simple_tokenize(s)
            cnt.update(toks)
        for tok, freq in cnt.most_common(max_size):
            if freq < min_freq: break
            self.add_token(tok)

    def encode(self, text, max_len=32):
        toks = simple_tokenize(text)
        ids = [self.bos] + [self.stoi.get(t, self.unk) for t in toks][:max_len-2] + [self.eos]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad]*pad_len
        return torch.LongTensor(ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            itos = json.load(f)
        v = cls()
        v.stoi = {tok:i for i,tok in enumerate(itos)}
        v.itos = itos
        v.pad = v.stoi['<pad>']; v.unk = v.stoi['<unk>']
        v.bos = v.stoi['<bos>']; v.eos = v.stoi['<eos>']
        return v