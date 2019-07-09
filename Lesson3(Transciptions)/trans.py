import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class Vocab:
    def __init__(self, counter, sos, eos, pad, unk, min_freq=None):
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        
        self._token2idx = {
            self.sos: self.sos_idx,
            self.eos: self.eos_idx,
            self.pad: self.pad_idx,
            self.unk: self.unk_idx,
        }
        self._idx2token = {idx:token for token, idx in self._token2idx.items()}
        
        idx = len(self._token2idx)
        min_freq = 0 if min_freq is None else min_freq
        
        for token, count in counter.items():
            if count > min_freq:
                self._token2idx[token] = idx
                self._idx2token[idx]   = token
                idx += 1
        
        self.vocab_size = len(self._token2idx)
        self.tokens     = list(self._token2idx.keys())
    
    def token2idx(self, token):
        return self._token2idx.get(token, self.pad_idx)
    
    def idx2token(self, idx):
        return self._idx2token.get(idx, self.pad)
    
    def __len__(self):
        return len(self._token2idx)
    
def padding(sequences, pad_idx):
    '''
    Inputs:
        sequences: list of list of tokens
    '''
    max_length = max(map(len, sequences))
    
    return [seq + [pad_idx]*(max_length - len(seq)) for seq in sequences]



import csv
from collections import Counter

def words_tokenize(line):
    return list(line)

def trans_tokenize(line):
    return line.split()

class Dataset(object):
    def __init__(self, path):
        val_size = 0.1
        shuffle  = True

        with open(path, 'r') as f:
            reader = csv.reader(f)
            lines   = list(reader)

        _, words, trans = zip(*lines[1:])

        c = list(zip(words, trans))
        random.shuffle(c)
        words, trans = zip(*c)

        val_size = int(len(words) * val_size)
        train_words, val_words = words[val_size:], words[:val_size]
        train_trans, val_trans = trans[val_size:], trans[:val_size]
        
        words_counter = Counter()
        trans_counter = Counter()

        for line in train_words:
            tokens = words_tokenize(line)
            for token in tokens:
                words_counter[token] += 1

        for line in train_trans:
            tokens = trans_tokenize(line)
            for token in tokens:
                trans_counter[token] += 1
                
        sos = "<sos>"
        eos = "<eos>"
        pad = "<pad>"
        unk = "<unk>"

        self.words_vocab = Vocab(words_counter, 
                            sos, eos, pad, unk)

        self.trans_vocab = Vocab(trans_counter, 
                            sos, eos, pad, unk)
        
        self.train_words = [[self.words_vocab.token2idx(item) for item in words_tokenize(word)] for word in train_words]
        self.val_words   = [[self.words_vocab.token2idx(item) for item in words_tokenize(word)] for word in val_words]

        self.train_trans = [[self.trans_vocab.token2idx(item) for item in trans_tokenize(trans)] for trans in train_trans]
        self.val_trans   = [[self.trans_vocab.token2idx(item) for item in trans_tokenize(trans)] for trans in val_trans]
        
    def __len__(self):
        return len(self.train_trans)
        
    def get_batch(self, batch_size, sort=False, val=False):
        if val:
            words, trans = self.val_words,   self.val_trans
        else:
            words, trans = self.train_words, self.train_trans

        random_ids = np.random.randint(0, len(words), batch_size)
        batch_words = [words[idx] for idx in random_ids]
        batch_trans = [trans[idx] for idx in random_ids]

        batch_trans_in  = [[self.trans_vocab.sos_idx] + tran for tran in batch_trans]
        batch_trans_out = [tran + [self.trans_vocab.eos_idx] for tran in batch_trans]

        words_lens = list(map(len, batch_words))
        trans_lens = list(map(len, batch_trans_in))

        batch_words     = padding(batch_words,     pad_idx=self.words_vocab.pad_idx)
        batch_trans_in  = padding(batch_trans_in,  pad_idx=self.trans_vocab.pad_idx)
        batch_trans_out = padding(batch_trans_out, pad_idx=self.trans_vocab.pad_idx)


        batch_words     = torch.LongTensor(batch_words).to(device)
        batch_trans_in  = torch.LongTensor(batch_trans_in).to(device)
        batch_trans_out = torch.LongTensor(batch_trans_out).to(device)
        words_lens = torch.LongTensor(words_lens).to(device)
        trans_lens = torch.LongTensor(trans_lens).to(device)

        if sort:
            lens, indices   = torch.sort(words_lens, 0, True)
            batch_words     = batch_words[indices]
            batch_trans_in  = batch_trans_in[indices]
            batch_trans_out = batch_trans_out[indices]
            trans_lens = trans_lens[indices]
            words_lens = lens

        return batch_words, batch_trans_in, batch_trans_out, words_lens, trans_lens
    

class Trainer:
    def __init__(self, dataset, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size):
        
        self.dataset = dataset
        self.train_losses = []
        self.val_losses = []
        self.batch_size = batch_size
        
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        
               
    def train(self, n_epochs):
        
        for epoch in range(n_epochs):

            for batch_idx in range(len(self.dataset)//self.batch_size):

                batch_words, batch_trans_in, batch_trans_out, words_lens, trans_lens = self.dataset.get_batch(32)
                hidden = self.encoder(batch_words)

                logits = []
                for  t in range(batch_trans_in.size(1)):
                    logit, hidden = self.decoder(batch_trans_in[:, t], hidden)
                    logits.append(logit)


                logits = torch.stack(logits, 1)
                logits = logits.view(-1, len(trans_vocab))
                batch_trans_out = batch_trans_out.view(-1)                

                mask = batch_trans_out != trans_vocab.pad_idx


                loss = self.criterion(logits[mask], batch_trans_out[mask])

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                self.train_losses.append(loss.item())
            
            val_loss = self.eval_()
            self.val_losses.append(val_loss)
            self.plot(epoch, batch_idx, self.train_losses, self.val_losses)
        
    def eval_(self):
        
        val_words, val_trans_in, val_trans_out, val_words_lens, val_trans_lens = self.dataset.get_batch(32, val = True)
        
        hidden = self.encoder(val_words)

        logits = []
        for  t in range(val_trans_in.size(1)):
            logit, hidden = self.decoder(val_trans_in[:, t], hidden)
            logits.append(logit)


        logits = torch.stack(logits, 1)
        logits = logits.view(-1, len(trans_vocab))
        val_trans_out = val_trans_out.view(-1)                

        mask = val_trans_out != trans_vocab.pad_idx


        loss = self.criterion(logits[mask], val_trans_out[mask])

    
    def plot(self, epoch, batch_idx, train_losses, val_losses):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('epoch %s. | batch: %s | loss: %s' % (epoch, batch_idx, train_losses[-1]))
        plt.plot(train_losses)
        plt.subplot(132)
        plt.title('epoch %s. | loss: %s' % (epoch, val_losses[-1]))
        plt.plot(val_losses)
        plt.show()

      