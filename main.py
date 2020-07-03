
import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import GloVe

from torchtext import data
from torchtext.data import TabularDataset

from torchtext.data import Iterator

from lib.model import CNN_Text
import argparse

embedSize = 10000
embedDim = 300
maxLen = 200

# data filed definition
TEXT = data.Field(sequential = True, lower=True, batch_first = True, fix_length=maxLen)
LABEL = data.Field(sequential = False, use_vocab = False, is_target =True)

trainData, testData = TabularDataset.splits(path = 'data/corpus/', train='train.csv', test='test.csv', fields=[('text',TEXT),('label',LABEL)],format='csv',csv_reader_params={'delimiter':'\t', 'quotechar':None})

TEXT.build_vocab(testData,trainData, vectors = GloVe(name='6B',dim=embedDim), max_size=embedSize,min_freq =10)
LABEL.build_vocab(testData,trainData)
embeddingLayer = nn.Embedding.from_pretrained(TEXT.vocab.vectors,freeze=False)

batchSize = 5
trainLoader = Iterator(dataset=trainData,batch_size = batchSize)

batch = next(iter(trainLoader))

parser = argparse.ArgumentParser(description='CNN text classifier')

parser.add_argument('-kernel-num',type=int,default =100,help='number of each kind of kernel')
parser.add_argument('-kernel-sizes',type=str,default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-dropout',type=float,default=0.5, help='the probability for dropout [default : 0.5]')


args = parser.parse_args()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.class_num = len(LABEL.vocab)-1
args.embed_num = maxLen
args.embed_dim = embedDim

cnnModel = CNN_Text(args)
embedInput = embeddingLayer(batch.text)
print(cnnModel(embedInput).size())

