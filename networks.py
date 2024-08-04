
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.image_utils import apply_mask, compute_mask_ratio, calculate_iou
from collections import deque


def inverse_sigmoid(x):
    return torch.log(x/(1-x))
class LinearSVM(nn.Module):
    def __init__(self, set_bias=0.86, input_dim=256, lr=0.01):
        super(LinearSVM, self).__init__()
        # 只有一个线性层
        self.linear = nn.Linear(input_dim, 1)  # self.clip_feature shape为256
        
        nn.init.constant_(self.linear.bias, 2-inverse_sigmoid(torch.tensor(set_bias)))
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def weight_set(self, weights):
        with torch.no_grad():
            self.linear.weight.copy_(weights)

    @torch.no_grad()
    def eval_forward(self, x, y):
        self.optimizer.zero_grad()
        y = y.squeeze()
        output = self.forward(x).squeeze()
        iou = calculate_iou(y>0, output>0)
        return iou
    
    def step(self, x, y):
        self.optimizer.zero_grad()
        y = y.squeeze()
        output = self.forward(x).squeeze()
        
        # # 正样本权重为10
        weights = torch.tensor([50]).to(x.device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)(output, y)
        # loss = torch.nn.BCEWithLogitsLoss(neg_weight=weights)(output, y)
        
        # svm loss
        loss = hinge_loss(output, y)

        iou = calculate_iou(y>0, output>0)

        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            output = self.forward(x).squeeze()
            iou = calculate_iou(y>0, output>0)

        return loss, iou
    
    def forward(self, x):
        return self.linear(x/0.3438)

# 定义hinge损失函数
def hinge_loss(outputs, labels):
    # 将标签从{0,1}转换为{-1,1}
    labels = 2 * labels - 1
    # 计算hinge损失
    loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
    return loss

class ConvergenceTracker:
    def __init__(self, threshold=1e-5, patience=5):
        self.threshold = threshold
        self.patience = patience
        self.loss_history = deque(maxlen=patience)

    def add_loss(self, loss):
        self.loss_history.append(loss)

    def has_converged(self):
        if len(self.loss_history) < self.patience:
            return False

        max_loss = max(self.loss_history)
        min_loss = min(self.loss_history)

        return max_loss - min_loss < self.threshold