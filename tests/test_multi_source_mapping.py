"""Tests for multi-source mapping in GraphTransformer."""

import pandas as pd
import pytest

from src.transformers.graph_transformer import GraphTransformer


class TestMultiSourceMapping:
    """Each node and relationship reads from its own declared source DataFrame."""

    def setup_method(self):
        self.transformer = GraphTransformer(enable_parallel=False)

    def _build_data(self):
        users = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Carol"],
                "email": ["a@x", "b@x", "c@x"],
            }
        )
        orders = pd.DataFrame(
            {
                "order_id": [101, 102, 103, 104],
                "user_id": [1, 1, 2, 3],
                "total_amount": [10.0, 20.0, 30.0, 40.0],
            }
        )
        return {"users": users, "orders": orders}

    def _mapping_config(self):
        return {
            "name": "users_and_orders",
            "nodes": [
                {
                    "label": "User",
                    "source": "users",
                    "id_field": "user_id",
                    "properties": [
                        {"field": "name", "type": "string"},
                        {"field": "email", "type": "string"},
                    ],
                },
                {
                    "label": "Order",
                    "source": "orders",
                    "id_field": "order_id",
                    "properties": [
                        {"field": "total_amount", "type": "float"},
                    ],
                },
            ],
            "relationships": [
                {
                    "type": "PLACED_ORDER",
                    "source": "orders",
                    "from_id_column": "user_id",
                    "to_id_column": "order_id",
                    "from_node": "User",
                    "to_node": "Order",
                }
            ],
        }

    def test_nodes_read_from_their_own_source(self):
        """Each node label must come from its declared source df only."""
        data = self._build_data()
        nodes_data, _ = self.transformer.transform_mapping_multi_source(
            data, self._mapping_config()
        )

        users_node = next(n for n in nodes_data if n.iloc[0]["_label"] == "User")
        orders_node = next(n for n in nodes_data if n.iloc[0]["_label"] == "Order")

        # User nodes must reflect the users source — 3 rows, not 4 (which would
        # indicate the bug where the orders df was used for everything).
        assert len(users_node) == 3
        assert set(users_node["_id"].tolist()) == {1, 2, 3}
        assert "name" in users_node.columns

        # Order nodes must reflect the orders source.
        assert len(orders_node) == 4
        assert set(orders_node["_id"].tolist()) == {101, 102, 103, 104}

    def test_relationship_reads_from_its_own_source(self):
        data = self._build_data()
        _, rels_data = self.transformer.transform_mapping_multi_source(
            data, self._mapping_config()
        )

        assert len(rels_data) == 1
        rel_df = rels_data[0]
        assert len(rel_df) == 4
        assert rel_df["_type"].iloc[0] == "PLACED_ORDER"
        assert sorted(rel_df["_from_id"].tolist()) == [1, 1, 2, 3]
        assert sorted(rel_df["_to_id"].tolist()) == [101, 102, 103, 104]

    def test_shared_source_not_mutated_between_configs(self):
        """Two node configs sharing the same source must not corrupt each other via in-place ops."""
        # Both node configs read from the same 'orders' DataFrame.
        orders = pd.DataFrame(
            {
                "order_id": [101, 102],
                "user_id": [1, 2],
                "total_amount": [10.0, 20.0],
            }
        )
        data = {"orders": orders}

        config = {
            "name": "shared_source",
            "nodes": [
                {
                    "label": "Order",
                    "source": "orders",
                    "id_field": "order_id",
                    "properties": [{"field": "total_amount", "type": "float"}],
                },
                {
                    "label": "OrderByUser",
                    "source": "orders",
                    "id_field": "user_id",
                    "properties": [{"field": "total_amount", "type": "float"}],
                },
            ],
        }

        # Should not raise and each node should have correct row counts.
        nodes_data, _ = self.transformer.transform_mapping_multi_source(data, config)

        order_node = next(n for n in nodes_data if n.iloc[0]["_label"] == "Order")
        order_by_user_node = next(
            n for n in nodes_data if n.iloc[0]["_label"] == "OrderByUser"
        )

        assert len(order_node) == 2
        assert len(order_by_user_node) == 2

    def test_missing_source_raises(self):
        """A node referencing an unloaded source should fail loudly."""
        data = self._build_data()
        config = self._mapping_config()
        config["nodes"].append(
            {
                "label": "Product",
                "source": "products",
                "id_field": "product_id",
                "properties": [],
            }
        )

        with pytest.raises(KeyError, match="products"):
            self.transformer.transform_mapping_multi_source(data, config)
