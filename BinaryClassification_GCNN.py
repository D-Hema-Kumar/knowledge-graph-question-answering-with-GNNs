from torch_geometric.data import Data, DataLoader
from DataBuilder import DataBuilder
from constants import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    EMBEDDINGS_PATH,
)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from loguru import logger

## CREATE DATA
logger.info("Creating data.")

data_builder = DataBuilder(
    triples_path=TRIPLES_PATH,
    entities_labels_path=ENTITIES_LABELS_PATH,
    properties_labels_path=PROPERTIES_LABELS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
    labeler=(lambda x: 1 if "APQC" in x else 0),
)

x = data_builder.get_x()
edge_index = data_builder.get_edge_index()
y = data_builder.get_y()
train_mask, val_mask, test_mask = data_builder.get_masks()

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)


## TRAIN GNN
logger.info("Training GNN.")


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

## EVALUATE
logger.info("Evaluating.")
model.eval()
pred = model(data).argmax(dim=1)

correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]
TP = sum(pred[data.test_mask][correct_predictions_mask] == 1)
FP = sum(pred[data.test_mask][incorrect_predictions_mask] == 1)
FN = sum(pred[data.test_mask][incorrect_predictions_mask] == 0)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {2 * (precision*recall)/(precision+recall):.4f}")
