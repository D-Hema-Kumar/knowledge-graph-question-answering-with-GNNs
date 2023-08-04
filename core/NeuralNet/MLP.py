import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, num_node_features, dim_hidden_layer, num_classes):
        super().__init__()
        self.lin1 = Linear(num_node_features, dim_hidden_layer)
        self.lin2 = Linear(dim_hidden_layer, num_classes)

    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def evaluate(self, data):
        self.eval()
        pred = self.forward(data).argmax(dim=1)
        correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
        incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]
        TP = sum(pred[data.test_mask][correct_predictions_mask] == 1)
        FP = sum(pred[data.test_mask][incorrect_predictions_mask] == 1)
        FN = sum(pred[data.test_mask][incorrect_predictions_mask] == 0)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1
