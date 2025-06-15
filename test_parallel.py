"""Test parallel processing functionality."""

import pandas as pd
import time
from src.transformers.graph_transformer import GraphTransformer

# Test data
df = pd.DataFrame({
    'id': range(1000),
    'name': [f'User{i}' for i in range(1000)],
    'email': [f'user{i}@example.com' for i in range(1000)],
    'company_id': [i % 10 for i in range(1000)]
})

# Test mapping config with multiple similar nodes
mapping_config = {
    "name": "test_mapping",
    "nodes": [
        {
            "label": "User",
            "id_field": "id",
            "properties": [
                {"field": "name", "type": "string"},
                {"field": "email", "type": "string"}
            ]
        },
        {
            "label": "Person",
            "id_field": "id", 
            "properties": [
                {"field": "name", "type": "string"}
            ]
        },
        {
            "label": "Contact",
            "id_field": "id",
            "properties": [
                {"field": "email", "type": "string"}
            ]
        }
    ]
}

print("Testing parallel vs sequential processing...")

# Test sequential processing
transformer_seq = GraphTransformer(enable_parallel=False)
start_time = time.time()
nodes_seq, rels_seq = transformer_seq.transform_mapping(df, mapping_config)
seq_time = time.time() - start_time

# Test parallel processing  
transformer_par = GraphTransformer(enable_parallel=True, max_workers=2)
start_time = time.time()
nodes_par, rels_par = transformer_par.transform_mapping(df, mapping_config)
par_time = time.time() - start_time

print(f"Sequential processing: {seq_time:.4f} seconds")
print(f"Parallel processing: {par_time:.4f} seconds")
print(f"Performance improvement: {seq_time/par_time:.2f}x")

# Verify results are the same
print(f"Sequential nodes: {len(nodes_seq)}")
print(f"Parallel nodes: {len(nodes_par)}")
print(f"Results match: {len(nodes_seq) == len(nodes_par)}")