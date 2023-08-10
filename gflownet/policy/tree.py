from typing import Optional

import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool

from gflownet.envs.tree import Attribute, Stage, Tree
from gflownet.policy.base import Policy


class Backbone(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 64,
        input_dim: int = 5,
        layer: str = "GCNConv",
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        layers = []
        for i in range(n_layers):
            layers.append(
                (
                    layer(input_dim if i == 0 else hidden_dim, hidden_dim),
                    "x, edge_index -> x",
                )
            )
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

        self.model = torch_geometric.nn.Sequential("x, edge_index, batch", layers)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        return self.model(x, edge_index, batch)


class LeafSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_layers: int = 2,
        hidden_dim: int = 64,
        layer: str = "GCNConv",
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        body_layers = []
        for i in range(n_layers - 1):
            body_layers.append(
                (
                    layer(backbone.hidden_dim if i == 0 else hidden_dim, hidden_dim),
                    "x, edge_index -> x",
                )
            )
            body_layers.append(activation())
            if dropout > 0:
                body_layers.append(torch.nn.Dropout(p=dropout))

        self.backbone = backbone
        self.body = torch_geometric.nn.Sequential("x, edge_index, batch", body_layers)
        self.leaf_head_layer = layer(hidden_dim, 1)
        self.eos_head_layers = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data: torch_geometric.data.Data) -> (torch.Tensor, torch.Tensor):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.backbone(data)
        x = self.body(x, edge_index, batch)

        y_leaf = self.leaf_head_layer(x, edge_index, batch)
        y_leaf = y_leaf.squeeze(-1)

        x_pool = global_mean_pool(x, batch)
        y_eos = self.eos_head_layers(x_pool)[:, 0]

        return y_leaf, y_eos


def _construct_node_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    activation: str,
    dropout: float,
) -> torch.nn.Module:
    activation = getattr(torch.nn, activation)

    layers = []
    for i in range(n_layers):
        layers.append(
            torch.nn.Linear(
                input_dim if i == 0 else hidden_dim,
                output_dim if i == n_layers - 1 else hidden_dim,
            ),
        )
        if i < n_layers - 1:
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

    return torch.nn.Sequential(*layers)


class FeatureSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, output_dim, n_layers, activation, dropout
        )

    def forward(
        self, data: torch_geometric.data.Data, node_index: torch.Tensor
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node], dim=1)
        x = self.model(x)

        return x


class ThresholdSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, output_dim, n_layers, activation, dropout
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        node_index: torch.Tensor,
        feature_index: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node, feature_index], dim=1)
        x = self.model(x)

        return x


class OperatorSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, 2, n_layers, activation, dropout
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        node_index: torch.Tensor,
        feature_index: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node, feature_index, threshold], dim=1)
        x = self.model(x)

        return x


class TreeModel(torch.nn.Module):
    def __init__(
        self,
        continuous: bool,
        policy_output_dim: int,
        leaf_index: int,
        feature_index: int,
        threshold_index: int,
        operator_index: int,
        eos_index: int,
        base: Optional["TreePolicy"] = None,
        backbone_args: Optional[dict] = None,
        leaf_head_args: Optional[dict] = None,
        feature_head_args: Optional[dict] = None,
        threshold_head_args: Optional[dict] = None,
        operator_head_args: Optional[dict] = None,
    ):
        super().__init__()

        self.continuous = continuous
        self.policy_output_dim = policy_output_dim
        self.leaf_index = leaf_index
        self.feature_index = feature_index
        self.threshold_index = threshold_index
        self.operator_index = operator_index
        self.eos_index = eos_index

        if base is None:
            self.backbone = Backbone(**backbone_args)
        else:
            self.backbone = base.model.backbone

        self.leaf_head = LeafSelectionHead(backbone=self.backbone, **leaf_head_args)
        self.feature_head = FeatureSelectionHead(
            backbone=self.backbone,
            input_dim=(2 * self.backbone.hidden_dim),
            **feature_head_args,
        )
        self.threshold_head = ThresholdSelectionHead(
            backbone=self.backbone,
            input_dim=(2 * self.backbone.hidden_dim + 1),
            **threshold_head_args,
        )
        self.operator_head = OperatorSelectionHead(
            backbone=self.backbone,
            input_dim=(2 * self.backbone.hidden_dim + 2),
            **operator_head_args,
        )


class ForwardTreeModel(TreeModel):
    def forward(self, x):
        logits = torch.full((x.shape[0], self.policy_output_dim), torch.nan)

        for i, state in enumerate(x):
            stage = Tree.get_stage(state)
            graph = Tree.to_pyg(state)

            if stage == Stage.COMPLETE:
                y_leaf, y_eos = self.leaf_head(graph)
                logits[i, self.leaf_index : len(y_leaf)] = y_leaf
                logits[i, self.eos_index] = y_eos
            else:
                k = Tree._find_active(state)
                node_index = torch.Tensor([k]).long()
                feature_index = state[k, Attribute.FEATURE].unsqueeze(0).unsqueeze(0)
                threshold = state[k, Attribute.THRESHOLD].unsqueeze(0).unsqueeze(0)

                if stage == Stage.LEAF:
                    logits[
                        i, self.feature_index : self.threshold_index
                    ] = self.feature_head(graph, node_index)[0]
                elif stage == Stage.FEATURE:
                    head_output = self.threshold_head(
                        graph,
                        node_index,
                        feature_index,
                    )[0]
                    if self.continuous:
                        logits[
                            i, self.threshold_index : self.operator_index
                        ] = head_output
                    else:
                        logits[i, (self.eos_index + 1) :] = head_output
                elif stage == Stage.THRESHOLD:
                    logits[
                        i, self.operator_index : self.eos_index
                    ] = self.operator_head(
                        graph,
                        node_index,
                        feature_index,
                        threshold,
                    )[
                        0
                    ]
                else:
                    raise ValueError(f"Unrecognized stage = {stage}.")

        return logits


class BackwardTreeModel(TreeModel):
    def forward(self, x):
        logits = torch.full(
            (x.shape[0], self.policy_output_dim), 1.0
        )  # TODO: implement actual forward

        return logits


class TreePolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.backbone_args = {}
        self.leaf_head_args = {}
        self.feature_head_args = {"output_dim": env.X_train.shape[1]}
        if env.continuous:
            self.threshold_head_args = {"output_dim": env.components * 3}
        else:
            self.threshold_head_args = {"output_dim": len(env.thresholds)}
        self.operator_head_args = {}
        self.continuous = env.continuous
        self.policy_output_dim = env.policy_output_dim
        self.leaf_index = env._action_index_pick_leaf
        self.feature_index = env._action_index_pick_feature
        self.threshold_index = env._action_index_pick_threshold
        self.operator_index = env._action_index_pick_operator
        self.eos_index = env._action_index_eos

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

        self.is_model = True

    def parse_config(self, config):
        if config is not None:
            self.backbone_args.update(config.get("backbone_args", {}))
            self.leaf_head_args.update(config.get("leaf_head_args", {}))
            self.feature_head_args.update(config.get("feature_head_args", {}))
            self.threshold_head_args.update(config.get("threshold_head_args", {}))
            self.operator_head_args.update(config.get("operator_head_args", {}))

    def instantiate(self):
        if self.base is None:
            model_class = ForwardTreeModel
        else:
            model_class = BackwardTreeModel

        self.model = model_class(
            continuous=self.continuous,
            policy_output_dim=self.policy_output_dim,
            leaf_index=self.leaf_index,
            feature_index=self.feature_index,
            threshold_index=self.threshold_index,
            operator_index=self.operator_index,
            eos_index=self.eos_index,
            base=self.base,
            backbone_args=self.backbone_args,
            leaf_head_args=self.leaf_head_args,
            feature_head_args=self.feature_head_args,
            threshold_head_args=self.threshold_head_args,
            operator_head_args=self.operator_head_args,
        ).to(self.device)

    def __call__(self, states):
        return self.model(states)
