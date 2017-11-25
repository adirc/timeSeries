import numpy as np


def grad_norm(parameters,norm_type = 2):
    total_norm = 0
    for p in parameters:
        if  (p.grad is not  None):
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1/norm_type)
    return total_norm


def maybe_cuda(x,is_cuda):

    if (is_cuda):
        return x.cuda()
    return x


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


class predictions_analysis(object):

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


    def add(self,predicions, targets):
        self.tp += ((predicions == targets) & (1 == predicions)).sum()
        self.tn += ((predicions == targets) & (0 == predicions)).sum()
        self.fp += ((predicions != targets) & (1 == predicions)).sum()
        self.fn += ((predicions != targets) & (0 == predicions)).sum()


    def calc_recall(self):
        if self.tp  == 0 and self.fn == 0:
            return -1

        return np.true_divide(self.tp, self.tp + self.fn)

    def calc_precision(self):
        if self.tp  == 0 and self.fp == 0:
            return -1

        return  np.true_divide(self.tp,self.tp + self.fp)




    def get_f1(self):
        if (self.tp + self.fp == 0):
            return 0.0
        if (self.tp + self.fn == 0):
            return 0.0
        precision = self.calc_precision()
        recall = self.calc_recall()
        if (not ((precision + recall) == 0)):
            f1 = 2*(precision*recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def get_accuracy(self):

        total = self.tp + self.tn + self.fp + self.fn
        if (total == 0) :
            return 0.0
        else:
            return np.true_divide(self.tp + self.tn, total)


    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0


