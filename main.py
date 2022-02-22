#!/usr/bin/env python3
"""
License: GNU General Public License, version 3
Author: Fabien Geyer

Example GNN applied to the prediction of delay bounds
This example was built in the scope of the AI in Networking Summer School 2022.
"""

import os
import enum
import pbzlib
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm, trange
from networkx.utils.misc import pairwise
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GNNModel(gnn.MessagePassing):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int, dropout: float, nunroll: int, final_sigmoid: bool):
        super(GNNModel, self).__init__()
        # First layers mapping the input dimension to the hidden dimension
        self.fci = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # GNN module
        self.nunroll = nunroll
        self.cell = gnn.GatedGraphConv(hidden_size, 1)

        # Final layers for node prediction
        self.fco = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
            # For the final layer we can also apply a sigmoid to map the output to the [0,1] interval
            nn.Sigmoid() if final_sigmoid else nn.Identity(),
        )

    def forward(self, x, edge_index):
        h = self.fci(x)
        for _ in range(self.nunroll):
            h = self.cell(h, edge_index)
        h = self.fco(h)
        return h


# These enums define the column indexes in the input matrix
class NodeType(enum.IntEnum):
    SERVER = 0
    FLOW = 1
    PATH_ORDER = 2


class Feature(enum.IntEnum):
    SERVER_RATE = 3
    SERVER_LATENCY = 4
    FLOW_RATE = 5
    FLOW_BURST = 6
    ORDER = 7


def network2graph(network):
    """
    Convert the network to its graph representation
    """

    G = nx.Graph()

    # Each server in the network is a node
    for server in network.server:
        G.add_node((NodeType.SERVER, server.id), rate=server.rate, latency=server.latency)

    # Each flow in the network is a node
    for flow in network.flow:
        prediction = flow.pmoo.delay_bound
        G.add_node((NodeType.FLOW, flow.id), rate=flow.rate, burst=flow.burst, prediction=prediction)

        # Connect the flow to the servers it traverses, each time with a path order node
        for i, server_id in enumerate(flow.path):
            G.add_node((NodeType.PATH_ORDER, (flow.id, i)), order=i)
            G.add_edge((NodeType.FLOW, flow.id), (NodeType.PATH_ORDER, (flow.id, i)))
            G.add_edge((NodeType.PATH_ORDER, (flow.id, i)), (NodeType.SERVER, server_id))

        for source_id, sink_id in pairwise(flow.path):
            G.add_edge((NodeType.SERVER, source_id), (NodeType.SERVER, sink_id))

    # Each node in the graph will be assigned an identifier. This will be used for defining the edges
    for nodeid, n in enumerate(G.nodes()):
        G.nodes[n]["nodeid"] = nodeid
    return G


def graph2torch(G: nx.Graph):
    """
    Convert the networkx graph to its torch matrices representation
    """

    # Input node features
    x = torch.zeros((G.number_of_nodes(), len(NodeType) + len(Feature)))
    # Label which we would like to predict. Here the end-to-end latency bound
    # The NaN is explicitly used for making sure that we only use labels on the relevant nodes
    y = torch.full((G.number_of_nodes(), 1), np.nan)
    # Mask used for selecting the query node in the loss function.
    # See torch.index_select(...) in the training function
    mask = torch.zeros(G.number_of_nodes(), dtype=torch.bool)

    for (nodetype, _), data in G.nodes(data=True):
        nid = data["nodeid"]  # Row in the x matrix

        x[nid, nodetype] = 1  # Node type encoded as categorical
        if nodetype == NodeType.SERVER:
            x[nid, Feature.SERVER_RATE] = data["rate"]
            x[nid, Feature.SERVER_LATENCY] = data["latency"]

        elif nodetype == NodeType.FLOW:
            x[nid, Feature.FLOW_RATE] = data["rate"]
            x[nid, Feature.FLOW_BURST] = data["burst"]
            if data["prediction"] > 0:
                y[nid] = data["prediction"]
                mask[nid] = True

        elif nodetype == NodeType.PATH_ORDER:
            x[nid, Feature.ORDER] = data["order"] + 1

    # Define the list of edges of the graph
    edge_index = set()
    for src, dst in G.edges():
        srcid, dstid = G.nodes[src]["nodeid"], G.nodes[dst]["nodeid"]
        # To make the graph non-directed, we make sure to add both edge directions
        edge_index.add((srcid, dstid))
        edge_index.add((dstid, srcid))
    edge_index = sorted(edge_index)
    edge_index = torch.tensor(edge_index).transpose(1, 0)

    return Data(x=x, y=y, edge_index=edge_index, mask=mask)


def parse_dataset(filename: str, N: int = 0):
    """
    Parse the networks from the pbz file and returns a list of graph data
    """
    dataset = []
    for network in tqdm(pbzlib.open_pbz(filename), ncols=0, desc=f"Process {filename}"):
        G = network2graph(network)
        dt = graph2torch(G)
        dataset.append(dt)
        if len(dataset) == N:
            break
    return dataset


def main(args):
    if os.path.exists("dataset-train.pt"):
        # Use the precomputed graphs
        graphs = torch.load("dataset-train.pt")
    else:
        # Prepare the graphs
        print("Parse the dataset and precompute the graphs")
        graphs = parse_dataset("dataset/dataset-train.pbz", 0)
        torch.save(graphs, "dataset-train.pt")
        print("Precomputed graphs are saved as: dataset-train.pt")
        return

    # Apply preprocessing to the target vector
    if args.preproc == "none":
        preproc = None
    elif args.preproc == "PowerTransformer":
        preproc = preprocessing.PowerTransformer()
    elif args.preproc == "QuantileTransformer":
        preproc = preprocessing.QuantileTransformer(random_state=args.seed)
    else:
        raise Exception("Invalid preproc argument")

    if preproc is not None:
        predictions = [torch.masked_select(dt.y.view(-1), dt.mask) for dt in graphs]
        predictions = torch.cat(predictions).numpy()
        preproc.fit(predictions.reshape(-1, 1))
        for dt in graphs:
            dt.yt = torch.from_numpy(preproc.transform(dt.y)).float()

    # Split the graphs in two subset: training and testing
    split = int(np.floor(len(graphs) * 0.8))
    np.random.shuffle(graphs)
    graphs_train, graphs_test = graphs[:split], graphs[split:]
    loader_train = DataLoader(graphs_train, shuffle=True, batch_size=args.batch_size)
    loader_test = DataLoader(graphs_test, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = GNNModel(
        num_features=len(NodeType) + len(Feature),
        num_classes=1,
        hidden_size=128,
        dropout=args.dropout,
        nunroll=args.unroll,
        final_sigmoid=args.final_sigmoid,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor)
    criterion = getattr(nn, args.lossfn)()

    def main_loop(loader: DataLoader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        losses = []
        metrics = []
        for dt in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)

            dt = dt.to(device)
            nodes_of_interest = torch.where(dt.mask)[0]
            output = model(dt.x, dt.edge_index)
            output = torch.index_select(output, 0, nodes_of_interest)
            if preproc is None:
                target = torch.index_select(dt.y, 0, nodes_of_interest)
            else:
                target = torch.index_select(dt.yt, 0, nodes_of_interest)
            loss = criterion(output, target)
            losses.append(loss.item())

            # Compute the metric as mean absolute percentage error (MAPE)
            if preproc is None:
                predictions = output.detach().cpu().numpy()
            else:
                # In case we did preprocessing on the output, we need to reverse it before computing the metric
                predictions = preproc.inverse_transform(output.detach().cpu().numpy())
            delays = torch.index_select(dt.y, 0, nodes_of_interest).cpu().numpy()
            metrics.extend(np.abs((delays - predictions) / delays).tolist())

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        return np.mean(losses), np.mean(metrics) * 100

    bestMetricV = np.inf
    lastImprovement = 0
    for epoch in trange(args.epochs, ncols=0):
        lossT, metricT = main_loop(loader_train, train=True)
        lossV, metricV = main_loop(loader_test, train=False)

        tqdm.write(f"{epoch} | LOSS train={lossT:.2e} test={lossV:.2e} | METRIC train={metricT:.1f}% {metricV:.1f}%")
        if args.nni:
            nni.report_intermediate_result(metricV)

        if metricV < bestMetricV:
            bestMetricV = metricV
            lastImprovement = epoch

        if epoch - lastImprovement > 15:
            tqdm.write("Early stop")
            break
        scheduler.step(lossT)

    if args.nni:
        nni.report_final_result(bestMetricV)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--unroll", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training")
    p.add_argument("--device", default="cuda", help="Device for the GNN (cpu or cuda)")
    p.add_argument("--max-grad-norm", type=float, default=10)
    p.add_argument("--lr-scheduler-factor", type=float, default=.5)
    p.add_argument("--nni", action="store_true")
    p.add_argument("--preproc", default="QuantileTransformer")
    p.add_argument("--lossfn", default="MSELoss")
    p.add_argument("--final-sigmoid", action="store_true")
    args = p.parse_args()

    if args.nni:
        # Import nni and gets the hyper-parameters
        import nni
        hparams = nni.get_next_parameter()
        for k, v in hparams.items():
            setattr(args, k, v)

    main(args)
