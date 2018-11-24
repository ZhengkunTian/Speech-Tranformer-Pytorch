import torch
import torch.nn as nn
import torch.nn.functional as func


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, vocab_size, weight=None, size_average=True, ignore_index=-1):
        assert 0.0 <= label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (vocab_size - 1)
        one_hot = torch.full((vocab_size,), smoothing_value)
        if not ignore_index:
            one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing
        self.criterion = CrossEntropyLoss(
            weight=weight, size_average=size_average)

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.padding_idx >= 0:
            model_prob.masked_fill_(
                (target == self.padding_idx).unsqueeze(1), 0)

        return self.criterion(output, model_prob)


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, target):
        assert inputs.dim() == 2
        assert target.dim() == 2

        batch_size = inputs.size(0)
        output_dim = inputs.size(1)

        input_log_softmax = self.log_softmax(inputs)

        weight = self.weight.repeat(inputs.size(0), 1)

        tmp = torch.addcmul(torch.zeros(
            batch_size, output_dim), -1, input_log_softmax, target)

        tmp_weighted = torch.addcmul(torch.zeros(
            batch_size, output_dim), 1, weight, tmp)

        if self.size_average:
            # num = torch.sum(tmp_weighted.ne(0)).float()
            loss = torch.sum(tmp_weighted).div(batch_size)
        else:
            loss = torch.sum(tmp_weighted)

        return loss


if __name__ == '__main__':

    inputs = torch.randn(3, 5, requires_grad=False)
    target = torch.tensor([1, 0, 4])

    weight = torch.ones(inputs.size(1))
    #weight[0] = 0
    celoss = nn.CrossEntropyLoss(weight, size_average=True)
    loss1 = celoss(inputs, target.contiguous().view(-1))
    print('loss1')
    print(loss1)

    lsceloss = LabelSmoothingLoss(0.1, inputs.size(
        1), weight=weight, size_average=True, ignore_index=-1)
    loss2 = lsceloss(inputs, target.contiguous().view(-1))
    print('loss2')
    print(loss2)
