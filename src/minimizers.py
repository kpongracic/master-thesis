import math
import glob
import torch
from torch.nn import functional as F
from torch import nn,Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import torchmetrics
from tqdm.notebook import tqdm
import pyfastx
import random
from torch.nn.utils.rnn import pad_sequence
from minipy import minimize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

import time
start_time = time.time()

seed_everything(42, workers=True)

LENGTH = 9984
UPPER = 15000
LOWER = 1000
OVERLAP = 0.5
KMER_SIZE = 32

KMER_LEN = 31
WIN_LEN = 31

NUMBER_OF_SEQ_TRAIN = 5000
NUMBER_OF_SEQ_VAL = 5000
NUMBER_OF_SEQ_TEST = 500
NUM_CLASSES = 8

BATCH_SIZE = 16
EPOCHS = 20

print("base minimizers")
print(f'NUMBER_OF_SEQ_TRAIN: {NUMBER_OF_SEQ_TRAIN}')
print(f'NUMBER_OF_SEQ_VAL: {NUMBER_OF_SEQ_VAL}')
print(f'NUMBER_OF_SEQ_TEST: {NUMBER_OF_SEQ_TEST}')
print(f'EPOCHS: {EPOCHS}')
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'run at: {datetime.now()}')
print('#############################', flush=True)


def encode_letters(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = [mapping[i] for i in sequence]
    return seq


################################################
################################################
################################################

train_files = glob.glob("../../data/zymo/Genomes/*.fasta")
train_files.sort()

train_index_list = list()
train_sequences = list()

for file in tqdm(train_files):
    fa = pyfastx.Fasta(file)
    
    seq = fa.longest
    length = len(seq)
    
    print(file[21:])
    
    start = 0
    end = LENGTH
    counter = 0
    
    train_sequences.append(seq.seq)
    
    while length > end and counter < NUMBER_OF_SEQ_TRAIN:
        try:
            encode_letters(seq.seq[start:end])
            train_index_list.append((start, end))
            
            counter += 1
        except:
            pass
        
        start += int(OVERLAP * LENGTH)
        end += int(OVERLAP * LENGTH)
    
    print(f'covered by : {counter} sequences')
    
    while counter < NUMBER_OF_SEQ_TRAIN:
        start = random.randint(0, length-LENGTH)
        new_length = int(random.gauss(5000, 2500))
        if new_length < 1000 or new_length > LENGTH:
            new_length = LENGTH
        end = start + new_length
        
        try:
            encode_letters(seq.seq[start:end])
            train_index_list.append((start, end))
            
            counter += 1
        except:
            pass

test_files = glob.glob("../../data/zymo/*.fastq")
test_files.sort()

test_index_list = list()
test_sequences = list()
class_id = 0

for file in tqdm(test_files):
    fa = pyfastx.Fastq(file)

    print(f'{file[13:]} #seq: {len(fa)}')
    
    counter = 0
    test_sequences.append(fa)
    
    for i in range(len(fa)):
        seq = fa[i]
        seq_length = len(seq)

        if seq_length > UPPER or seq_length < LOWER:
            continue

        seq_id = seq.name
        
        start = 0
        end = seq_length if seq_length < LENGTH else LENGTH
        
        try:
            encode_letters(seq.seq[start:end])
            test_index_list.append((class_id, seq_id))

            counter += 1
        except:
#             print('fail')
            pass
        if counter >= NUMBER_OF_SEQ_VAL + NUMBER_OF_SEQ_TEST:
            break
    print(f'counter: {counter}')
    class_id += 1

################################################
################################################
################################################

class TrainDataset(Dataset):
    def __init__(self, index_list, sequences):
        self.index_list = index_list
        self.sequences = sequences
        
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        y = math.floor(idx / NUMBER_OF_SEQ_TRAIN)
        
        start, end = self.index_list[idx]
        seq = self.sequences[y][start:end]
        
        minimizers = minimize(seq, KMER_LEN, WIN_LEN)

        time = len(minimizers)
        
        x = torch.randn(time, 31 * 4)
        
        for i in range(time):
            m = minimizers[i]
            
            kmer_start = m.position()
            
            kmer = encode_letters(seq[kmer_start:kmer_start + KMER_LEN])
            
            kmer = torch.tensor(kmer)
            kmer = F.one_hot(kmer, 4)
            kmer = kmer.float()
            kmer = torch.flatten(kmer)

            x[i] = kmer
            
        return x, torch.tensor(y)     

class TestDataset(Dataset):
    def __init__(self, index_list, sequences):
        self.index_list = index_list
        self.sequences = sequences
        
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        class_id, seq_id = self.index_list[idx]

        seq = self.sequences[class_id][seq_id].seq
        end = len(seq) if len(seq) < LENGTH else LENGTH
        seq = seq[:end]
        
        minimizers = minimize(seq, KMER_LEN, WIN_LEN)
        
        time = len(minimizers)
        
        x = torch.randn(time, 31 * 4)
        
        for i in range(time):
            m = minimizers[i]
            
            kmer_start = m.position()
            
            kmer = encode_letters(seq[kmer_start:kmer_start + KMER_LEN])
            
            kmer = torch.tensor(kmer)
            kmer = F.one_hot(kmer, 4)
            kmer = kmer.float()
            kmer = torch.flatten(kmer)

            x[i] = kmer

        return x, torch.tensor(class_id)

def pad_collate_fn(batch):
    X, y = zip(*batch)
    
    src_padding_mask = [torch.tensor([False] * (seq.shape[0] + 1)) for seq in X] # +1 zbog cls tokena
    src_padding_mask = pad_sequence(src_padding_mask, padding_value=True, batch_first=True)
    
    X = pad_sequence(X, padding_value=0, batch_first=True)

    print(f'shape: {X.shape}')
    
    return X, torch.tensor(y), src_padding_mask
################################################
################################################
################################################


train_count = NUM_CLASSES * NUMBER_OF_SEQ_TRAIN
test_count = NUM_CLASSES * NUMBER_OF_SEQ_TEST
validation_count = NUM_CLASSES * NUMBER_OF_SEQ_VAL

assert test_count + validation_count == len(test_index_list)

print(f'train_count: {train_count}')
print(f'validation_count: {validation_count}')
print(f'test_count: {test_count}')

train = TrainDataset(train_index_list, train_sequences)
test_val_dataset = TestDataset(test_index_list, test_sequences)

validation, test = random_split(test_val_dataset, [validation_count, test_count])

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)
validation_loader = DataLoader(validation, batch_size=8, num_workers=1, collate_fn=pad_collate_fn)
test_loader = DataLoader(test, batch_size=8, num_workers=1, collate_fn=pad_collate_fn)


################################################
################################################
################################################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = LENGTH):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
################################################
################################################
###############################################

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.input_linear_layer = nn.Linear(124, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, norm_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)        
        self.output_linear_layer = nn.Linear(256, NUM_CLASSES)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=NUM_CLASSES)

        self.lr = 1e-3

    def forward(self, x, src_padding_mask=None):
        x = self.input_linear_layer(x)
        B, T, _ = x.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = torch.transpose(x, 0, 1)    
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        x = x[0]      
        x = self.output_linear_layer(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y, src_padding_mask = batch
        logits = self(x, src_padding_mask)
        loss = F.nll_loss(logits, y)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y, src_padding_mask = batch
        logits = self(x, src_padding_mask)
        val_loss = F.nll_loss(logits, y)
        self.valid_acc(logits, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y, src_padding_mask = batch
        logits = self(x, src_padding_mask)
        loss = F.nll_loss(logits, y)

        self.test_acc(logits, y)
        self.test_confmat(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


model = MyModel()
checkpoint_callback = ModelCheckpoint(
        dirpath=f"../model-checkpoints/minimizers/{NUMBER_OF_SEQ_TRAIN}",
        monitor='valid_acc',
        filename='basev2-{epoch:02d}-{valid_acc:.3f}',
        save_top_k=3,
        mode='max',
    )
early_stop_callback = EarlyStopping(monitor="valid_acc", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = Trainer(gpus=1, max_epochs=EPOCHS, progress_bar_refresh_rate=100, callbacks=[early_stop_callback, checkpoint_callback])


# trainer.test(model, test_loader)
# print(model.test_confmat.compute())
# model.test_confmat.reset()

trainer.fit(model, train_loader, validation_loader)

print(f'best model path {checkpoint_callback.best_model_path}')

# model = MyModel.load_from_checkpoint("/hdd/kpongracic/src/model-checkpoints/minimizers/5000/base-epoch=12-valid_acc=0.836.ckpt")


trainer.test(model, test_loader)
print(model.test_confmat.compute())
model.test_confmat.reset()

print("\nExecution time: %s seconds" % (time.time() - start_time))