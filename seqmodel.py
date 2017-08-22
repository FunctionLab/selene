import math
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tabix
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


torch.set_num_threads(32)


BASES = np.array(['A', 'G', 'C', 'T'])
DIR="./splicing"  # TODO: REMOVE


def sequence_encoding(sequence):
    """Converts an input sequence to its binary encoding.

    Parameters
    ----------
    sequence : str
        The input sequence of length N.

    Returns
    -------
    numpy.ndarray, dtype=bool
        The N-by-4 encoding of the sequence.
    """
    encoding = np.zeros((len(sequence), 4), np.bool_)
    for base, index in zip(sequence, range(len(sequence))):
        encoding[index, :] = BASES == base
    return encoding


class Genome:

    def __init__(self, fa_file):
        """Wrapper class around the pyfaix.Fasta class

        Parameters
        ----------
        fa_file : str
            Path to an indexed FASTA file.
            File should contain the target organism's genome sequence.
        """
        self.genome=Fasta(fafile)

    def get_sequence(self, chrom, start, end, strand='+'):
        """Get the genomic sequence given the chromosome, sequence start,
        sequence end, and strand side information.

        Parameters
        ----------
        chrom : char|str|int
            Chromosome number or X/Y.
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.

        Returns
        -------
        str
            The genomic sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
        """
        if strand == '+':
            return self.genome[chrom][start:end].seq
        elif strand == '-':
            return self.genome[chrom][start:end].reverse.complement.seq
        else:
            raise ValueError(
                "Strand must be one of '+' or '-'. Input was {0}".format(
                    strand))


class GenomicData:

    def __init__(self, dataset, features):
        """Stores the dataset specifying sequence regions and features..
        Accepts a tabix-indexed .bed file with the following columns:
            [chrom, start (0-based), end, strand, feature]
        Additional columns are ignored.

        TODO: consider renaming this class?

        Parameters
        ----------
        dataset : str
            Path to the dataset.
        features : list[str]
            TODO: these could be retrieved from the dataset?
            The list of features (labels) we are interested in predicting.

        Attributes
        ----------
        data : TODO
        n_features : int
        features_map : dict
        """
        self.data = tabix.open(dataset)
        self.n_features = len(features)
        self.features_map = dict(
            [(feat, index) for index, feat in enumerate(features)])

    def get_feature_data(self, chrom, start, end, strand='+',
            verbose=False):
        """For a sequence of length L = `end` - `start`, return the features
        encoding corresponding to that region.
            e.g. for `n_features`, each position in that sequence will
            have a binary vector specifying whether each feature is
            present

        Parameters
        ----------
        chrom : char|str|int
            Chromosome number or X/Y.
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.
        verbose : bool, optional
            Default is False. TODO: keep this?

        Returns
        -------
        numpy.ndarray
            shape = [L, n_features]

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
        """
        encoding = np.zeros(end - start, self.n_features))
        rows = self.data.query(chrom, start, end)  # TODO: need strand?
        if strand == '+':
            for row in rows:
                if verbose:
                    print(row)
                # TODO: this could be a helper
                feat_start = int(row[1]) - start
                feat_end = int(row[2]) - start
                feature = row[4]
                feat_index = self.features_mapping[feature]
                code[feat_start:feat_end, feat_index] = 1
        elif strand == '-':
            for row in rows:
                if verbose:
                    print(row)
                feat_start = end - int(row[2])
                feat_end = end - int(row[1])
                feature = row[4]
                feat_index = self.features_mapping[feature]
                code[feat_start:feat_end, feat_index] = 1
        else:
            raise ValueError(
                "Strand must be one of '+' or '-'. Input was {0}".format(
                    strand))
        return encoding


class SplicingDataset:

    def __init__(self, genome, genome_data, features, holdout_chrs, radius=100, mode="all"):
        """TODO: documentation

        Parameters
        ----------
        genome : str
            Path to indexed FASTA file of target organism's complete
            genome sequence.
        genome_data : str
            Path to tabix-indexed .bed file that contains information
            about genomic features.
        features : list[str]
            List of genomic features.
        holdout_chrs : list[str]
            Chromosomes that act as our holdout dataset.
        radius : int, optional
            Default is 100.
        mode : {"all", "train", "test"}, optional
            Default is "all".

        Attributes
        ----------
        genome : Genome
        genome_features : GenomicData
        holdout_chrs : list[str]
        radius : int
        mode : {"all", "train", "test"}

        """
        self.genome = Genome(genome)
        self.genome_features = GenomicData(genome_data, features)
        self.holdout_chrs = holdout_chrs
        self.radius = radius
        self.mode = mode

        #self.genome = Genome(os.path.join(DIR, "hg38.fa"))
        #self.data = GenomicData(os.path.join(DIR, "splicejunc.database.bed.sorted.gz"),nfeatures=2,featureDict={'5p':0, '3p':1})

        #self.geneanno=pd.read_csv(os.path.join(DIR, "gencode.v25.annotation.gtf.gz.gene.pc"),sep='\t',header=None)
        #self.positives = pd.read_csv(os.path.join(DIR, "splicejunc.database.bed.sorted.gz"),sep='\t',header=None)
        #self.holdout = ['chr8','chr9']

        #initializations
        #self.geneanno['traininds']=pd.match(self.geneanno.iloc[:,0], ['chr8','chr9']) == -1
        #self.positives['traininds']=pd.match(self.positives.iloc[:,0], ['chr8','chr9']) == -1
        #self.geneanno['start_shifted']=self.geneanno.iloc[:,3]+self.radius
        #self.geneanno['end_shifted']=self.geneanno.iloc[:,4]-self.radius
        #self.geneanno['p'] = (self.geneanno['end_shifted'] - self.geneanno['start_shifted']+1)/np.sum(self.geneanno['end_shifted'] - self.geneanno['start_shifted']+self.geneanno.shape[0])
        #self.geneanno['p'] = np.asarray(self.geneanno['p'] * (self.geneanno['p']>=0))/np.sum(self.geneanno['p'] * (self.geneanno['p']>=0))

        if mode == 'all':
            self.all_mode()
        elif mode == 'train':
            self.train_mode(mode='train')
        elif mode == 'test':
            self.train_mode(mode='test')
        else:
            raise ValueError('Mode has to be one of "all", "train", and "test".')

    def _retrieve(self, chrom, position, strand, radius,
                  sequence_only=False, verbose=False, padding=(0, 0)):
        """
        Parameters
        ----------
        chrom : char|str|int
        position : int
        strand : {'+', '-'}
        radius : int
        sequence_only : bool, optional
            Default is False.
        verbose : bool, optional
            Default is False.
        padding : tuple(int, int), optional
            Default is (0, 0). Represents the amount of padding at the
            (start, end) of the region.

        Returns
        -------
        np.ndarray | tuple(np.ndarray, np.ndarray)
            If `sequence_only`, returns the sequence encoding and nothing else.
            Otherwise, returns both the sequence encoding and the feature labels
            for the specified range.
        """
        if verbose:
            print("{0}, {1}, {2}".format(chrom, position, strand))
        start = position - radius - padding[0]
        end = position + radius + padding[1] + 1
        retrieved_sequence = sequence_encoding(
            self.genome.get_sequence(chrom, start, end, strand))
        if sequence_only:
            return retrieved_sequence
        else:
            # TODO: need padding here?
            retrieved_data = self.genome_features.get_feature_data(
                chrom, pos - radius, pos + radius + 1, strand, verbose=verbose)
            return (retrieved_sequence, retrieved_data)

    def sample_background(self, sequence_only=False, verbose=False, padding=[0,0]):
        #select random gene weighted by gene length
        if len(self._randcache)==0:
            self._randcache = list(np.random.choice(self._inds, p=self._genep , size=2000))
        randind = self._randcache.pop()

        chr = self._genechr[randind]
        pos = int(self._genestart[randind] + np.random.uniform()*(self._geneend[randind]-self._genestart[randind]))
        strand = self._genestrand[randind]


        return self._retrieve(chr, pos, strand, self.radius, sequence_only, verbose, padding=padding)


    def sample_positive(self, sequence_only=False, verbose=False, padding=[0,0]):
        randind = np.random.randint(0,self._poschr.shape[0])
        chr = self._poschr[randind]
        pos = int(self._posstart[randind] + np.random.uniform()*(self._posend[randind]-self._posstart[randind]) + np.random.uniform()*self.radius)
        strand = self._posstrand[randind]

        if verbose:
            print chr, pos, strand
        return self._retrieve(chr, pos, strand, self.radius, sequence_only, verbose, padding=padding)

    def sample_mixture(self, positive_prop=0.5, sequence_only=False, verbose=False, padding=[0,0]):
        if np.random.uniform() < positive_prop:
            return self.sample_positive(sequence_only=sequence_only, verbose=verbose, padding=padding)
        else:
            return self.sample_background(sequence_only=sequence_only, verbose=verbose, padding=padding)

    def all_mode(self):
        self._genechr=np.asarray(self.geneanno.iloc[:,0])
        self._genestrand=np.asarray(self.geneanno.iloc[:,6])
        self._genestart=np.asarray(self.geneanno['start_shifted'])
        self._geneend=np.asarray(self.geneanno['end_shifted'])
        self._genep=np.asarray(self.geneanno['p']).copy()
        self._poschr=np.asarray(self.positives.iloc[:,0])
        self._posstrand=np.asarray(self.positives.iloc[:,3])
        self._posstart=np.asarray(self.positives.iloc[:,1])
        self._posend=np.asarray(self.positives.iloc[:,2])
        self._inds=np.arange(self.geneanno.shape[0])
        self._randcache=[]

    def train_mode(self, mode='train'):
        if mode == 'train':
            ind = np.asarray(self.geneanno['traininds'])
            pind = np.asarray(self.positives['traininds'])
        elif mode =='test':
            ind = ~np.asarray(self.geneanno['traininds'])
            pind = ~np.asarray(self.positives['traininds'])
        else:
            raise ValueError('mode has to be one of train and test.')

        self._genep=np.asarray(self.geneanno['p'])[ind].copy()
        self._genep=self._genep/np.sum(self._genep)
        self._genechr=np.asarray(self.geneanno.iloc[:,0])[ind]
        self._genestrand=np.asarray(self.geneanno.iloc[:,6])[ind]
        self._genestart=np.asarray(self.geneanno['start_shifted'])[ind]
        self._geneend=np.asarray(self.geneanno['end_shifted'])[ind]
        self._inds=np.arange(np.sum(ind))
        self._poschr=np.asarray(self.positives.iloc[:,0])[pind]
        self._posstrand=np.asarray(self.positives.iloc[:,3])[pind]
        self._posstart=np.asarray(self.positives.iloc[:,1])[pind]
        self._posend=np.asarray(self.positives.iloc[:,2])[pind]
        self._randcache=[]



# from matplotlib import pyplot as plt
# %matplotlib inline

#temporary fix
def _patched_view_4d(*tensors):
    output = []
    for t in tensors:
        assert t.dim() == 3
        size = list(t.size())
        size.insert(2, 1)
        output += [t.contiguous().view(*size)]
    return output

#import torch.nn._functions.conv
#torch.nn._functions.conv._view4d = _patched_view_4d

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)

    ratio = min(1, clip / (totalnorm + 1e-6))
    for p in model.parameters():
        p.grad.mul_(ratio)
    return min(1, clip / (totalnorm + 1e-6))


#model specification

hiddenSizes = [100,2]

n_lstm_layers = 2
rnn = nn.LSTM(input_size =  4, hidden_size = hiddenSizes[0], num_layers=n_lstm_layers, batch_first =True, bidirectional=True)

conv = nn.modules.container.Sequential(
    nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[0]*2, 1),
    nn.ReLU(),
    nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[1], 1))

model = [ rnn, conv]
useCuda = True
if useCuda:
    for module in model:
        module.cuda()

optimizers = [optim.SGD(module.parameters(), lr = 0.05, momentum=0.95) for module in model]
#optimizers = [optim.Adam(module.parameters(), lr = 0.1) for module in model]
criterion = nn.MSELoss()

sdata = SplicingDataset(mode='train',radius=100)

padding = [0,0]
def runBatch(batchSize= 16, update=True, plot=False):
    inputs = np.zeros((batchSize, sdata.radius*2+1+padding[0]+padding[1], 4))
    targets = np.zeros((batchSize, sdata.radius*2+1, 2))
    for i in range(batchSize):
        input, target = sdata.sample_mixture(0.5,padding=padding)
        inputs[i,:,:]=input
        targets[i,:,:]= np.log10(target+1e-6)+6
        #targets[targets<8]=0
    #temporary fix


    if useCuda:
        inputs = Variable(torch.Tensor(inputs).cuda(),requires_grad=True)
        targets = Variable(torch.Tensor(targets).cuda())
        h0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
        c0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
    else:
        inputs = Variable(torch.Tensor(inputs),requires_grad=True)
        targets = Variable(torch.Tensor(targets))
        h0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]))
        c0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]))

    outputs, hn = rnn(inputs, (h0, c0))
    outputs = conv(outputs.transpose(1,2)).transpose(1,2)

    loss = criterion(outputs,targets)

    if update:
        for module in model:
            module.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    if plot:
        # plt.figure()
        # plt.plot(outputs.data[0,:,:].numpy(),'r')
        # plt.plot(targets.data[0,:,:].numpy(),'g')
        # plt.show()
        plt.figure()
        plt.plot(outputs.data.numpy().flatten(),targets.data.numpy().flatten(),'.',alpha=0.2)
        plt.show()
    return loss.data[0]


for _ in range(10000):
    sdata.train_mode()
    cumlossTrain=0
    for _ in range(1000):
        cumlossTrain = cumlossTrain + runBatch()

    sdata.train_mode('test')
    cumlossTest=0
    for _ in range(100):
        cumlossTest = cumlossTest + runBatch(update=False)
    print("Train loss: %.5f, Test loss: %.5f." % (cumlossTrain, cumlossTest) )


torch.save(model,os.path.join(DIR, "models/101bp.h100.cpu.model"))
#model_trained = torch.load('/Genomics/ogtr04/jzthree/splicing/models/201bp.h100.cpu.model')
#sdata.radius=100

#model_trained = torch.load('/Genomics/ogtr04/jzthree/splicing/models/201bp.h500.posonly.cpu.model')
#sdata.radius=100
#sdata.train_mode('test')
#runBatch(batchSize=1024, update=False,  plot=True)
