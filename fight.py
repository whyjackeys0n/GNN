import torch
import time
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx


def visualize_graph(G, color):
    plt.figure(figsize=(12, 12))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=587), with_labels=False, node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(12, 12))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], c=color, s=150)
    if epoch is not None and loss is not None:
        plt.xlabel(f"Epoch: {epoch}, Loss: {loss:.4f}")
    plt.show()


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(587)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h


def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


dataset = KarateClub()
data = dataset[0]
edge_index = data.edge_index

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)

model = GCN()
print(model)
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize_embedding(h, color=data.y)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 1001):
    loss, h = train(data)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}')
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
