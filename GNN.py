import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data


class MLP(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_layers, num_classes):
        super().__init__()
        self.lin1 = Linear(num_node_features, num_hidden_layers)
        self.lin2 = Linear(num_hidden_layers, num_classes)

    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_layers, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden_layers)
        self.conv2 = GCNConv(num_hidden_layers, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def evaluate_model(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
    incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]
    TP = sum(pred[data.test_mask][correct_predictions_mask] == 1)
    FP = sum(pred[data.test_mask][incorrect_predictions_mask] == 1)
    FN = sum(pred[data.test_mask][incorrect_predictions_mask] == 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1


def _predict_answer(model, data):
    return model(data).argmax(dim=1).argmax()


def evaluate_qa_model(model, qa_data_builder, test_mask):
    for q_index, question_embedding in enumerate(
        qa_data_builder.questions_embeddings_masked(test_mask)
    ):
        question, q_embedding = question_embedding
        x_q = qa_data_builder.get_x(
            to_concat=q_embedding
        )  # adding the question embedding to the node embeddings
        y_q = qa_data_builder.get_y(question=question)
        data = Data(x=x_q, edge_index=qa_data_builder.get_edge_index(), y=y_q)
        pred = _predict_answer(model, data)
        correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
        incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]

        # correct_predictions -> 1
