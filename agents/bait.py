#######################################
# All Credits to Jordan Ash
# Code adapted from https://github.com/JordanAsh/badge/blob/master/query_strategies/bait_sampling.py
# Paper: Gone Fishing: Neural Active Learning with Fisher Embeddings
#######################################

import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from core.agent import BaseAgent


def select(X, K, fisher, iterates, lamb=1, nLabeled=0):
    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection, over-sample by 2x
    print('forward selection...', flush=True)
    over_sample = 2
    for i in range(int(over_sample * K)):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
            innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2,
                                  dim2=-1).sum(-1)

        # clear out gpu memory
        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        print(i, ind, traceEst[ind], flush=True)

        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    print('backward pruning...', flush=True)
    for i in range(len(indsAll) - K):
        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2,
                                  dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll


class Bait(BaseAgent):
    def __init__(self, agent_seed, config:dict, query_size, lamb=1.0):
        super().__init__(agent_seed, config, query_size)
        self.lamb = lamb


    # fisher embedding for bait (assumes cross-entropy loss)
    def get_regression_grad_embedding(self, X, Y, probs, model):
        if type(model) == list:
            model = self.clf

        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))

        embedding = np.zeros([len(Y), nLab, embDim * nLab])
        for ind in range(nLab):
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    cout, out = model(x)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        1 - batchProbs[j][c])
                            else:
                                embedding[idxs[j]][ind][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        -1 * batchProbs[j][c])
                        if len(probs) > 0:
                            embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[idxs[j]][ind])
                        else:
                            embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
        return torch.Tensor(embedding)

    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer):

        # get low-rank point-wise fishers
        xt = self.get_regression_grad_embedding(self.X, self.Y)

        # get fisher
        print('getting fisher matrix...', flush=True)
        batchSize = 1000  # should be as large as gpu memory allows
        nClass = torch.max(self.Y).item() + 1
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(len(self.X) / batchSize))
        for i in range(int(np.ceil(len(self.X) / batchSize))):
            xt_ = xt[i * batchSize: (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize: (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, nLabeled=np.sum(self.idxs_lb))
        return idxs_unlabeled[chosen]
