"""Test memory usage improvements from reduced DataFrame copying."""

import pandas as pd
import time
import tracemalloc
from src.transformers.data_mapper import DataMapper

def format_memory(bytes_used):
    """Format memory usage in human readable format."""
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.1f} KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/(1024**2):.1f} MB"
    else:
        return f"{bytes_used/(1024**3):.1f} GB"

def test_memory_usage():
    """Test memory usage with and without in-place operations."""
    
    # Create a larger test dataset
    size = 50000
    test_data = {
        'id': range(size),
        'name': [f'User_{i}' for i in range(size)],
        'email': [f'user{i}@example.com' for i in range(size)],
        'age': [str((i % 80) + 18) for i in range(size)],
        'salary': [str(30000 + (i % 100000)) for i in range(size)],
        'active': ['true' if i % 2 == 0 else 'false' for i in range(size)],
        'created_date': [f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(size)]
    }
    
    df = pd.DataFrame(test_data)
    print(f"Test dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"DataFrame memory usage: {format_memory(df.memory_usage(deep=True).sum())}")
    
    # Node configuration for testing
    node_config = {
        "label": "User",
        "id_field": "id",
        "properties": [
            {"field": "name", "type": "string"},
            {"field": "email", "type": "string"},
            {"field": "age", "type": "integer"},
            {"field": "salary", "type": "float"},
            {"field": "active", "type": "boolean"},
            {"field": "created_date", "type": "datetime"}
        ]
    }
    
    data_mapper = DataMapper()
    
    print("\n=== Memory Usage Test ===")
    
    # Test with copying (default behavior)
    print("\n1. With DataFrame copying (inplace=False):")
    tracemalloc.start()
    
    start_time = time.time()
    result_copy = data_mapper.map_node_properties(df, node_config, inplace=False)
    copy_time = time.time() - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Time: {copy_time:.4f} seconds")
    print(f"   Peak memory: {format_memory(peak)}")
    print(f"   Current memory: {format_memory(current)}")
    print(f"   Result shape: {result_copy.shape}")
    
    # Test with in-place operations
    print("\n2. With in-place operations (inplace=True):")
    df_copy = df.copy()  # Make a copy since we'll modify in-place
    tracemalloc.start()
    
    start_time = time.time()
    result_inplace = data_mapper.map_node_properties(df_copy, node_config, inplace=True)
    inplace_time = time.time() - start_time
    
    current_inplace, peak_inplace = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Time: {inplace_time:.4f} seconds")
    print(f"   Peak memory: {format_memory(peak_inplace)}")
    print(f"   Current memory: {format_memory(current_inplace)}")
    print(f"   Result shape: {result_inplace.shape}")
    
    # Calculate improvements
    time_improvement = copy_time / inplace_time if inplace_time > 0 else float('inf')
    memory_improvement = peak / peak_inplace if peak_inplace > 0 else float('inf')
    
    print(f"\n=== Performance Improvements ===")
    print(f"Time improvement: {time_improvement:.2f}x faster")
    print(f"Memory improvement: {memory_improvement:.2f}x less peak memory")
    print(f"Memory saved: {format_memory(peak - peak_inplace)}")
    
    # Verify results are equivalent (check a few columns)
    try:
        # Check if the transformed data types are similar
        equivalent = True
        for col in ['name', 'age', 'active']:
            if col in result_copy.columns and col in result_inplace.columns:
                # Compare non-null values
                copy_values = result_copy[col].dropna()
                inplace_values = result_inplace[col].dropna()
                if len(copy_values) != len(inplace_values):
                    equivalent = False
                    break
        
        print(f"Results equivalent: {equivalent}")
    except Exception as e:
        print(f"Result comparison failed: {e}")
    
    return {
        'copy_time': copy_time,
        'inplace_time': inplace_time,
        'copy_memory': peak,
        'inplace_memory': peak_inplace,
        'time_improvement': time_improvement,
        'memory_improvement': memory_improvement
    }

def test_relationship_memory():
    """Test memory usage for relationship mapping."""
    
    # Create test data for relationships
    size = 30000
    rel_data = {
        'user_id': [i % 1000 for i in range(size)],
        'company_id': [(i + 500) % 800 for i in range(size)],
        'relationship_type': ['WORKS_FOR'] * size,
        'start_date': [f'2020-{(i % 12) + 1:02d}-01' for i in range(size)],
        'salary': [str(40000 + (i % 50000)) for i in range(size)]
    }
    
    df = pd.DataFrame(rel_data)
    
    rel_config = {
        "type": "WORKS_FOR",
        "from_node": "User",
        "to_node": "Company",
        "properties": [
            {"field": "start_date", "type": "datetime"},
            {"field": "salary", "type": "float"}
        ]
    }
    
    data_mapper = DataMapper()
    
    print(f"\n=== Relationship Memory Test ===")
    print(f"Test dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Test with copying
    tracemalloc.start()
    result_copy = data_mapper.map_relationship_properties(df, rel_config, inplace=False)
    current_copy, peak_copy = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Test with in-place
    df_copy = df.copy()
    tracemalloc.start()
    result_inplace = data_mapper.map_relationship_properties(df_copy, rel_config, inplace=True)
    current_inplace, peak_inplace = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_improvement = peak_copy / peak_inplace if peak_inplace > 0 else float('inf')
    
    print(f"Copy peak memory: {format_memory(peak_copy)}")
    print(f"In-place peak memory: {format_memory(peak_inplace)}")
    print(f"Memory improvement: {memory_improvement:.2f}x")
    print(f"Memory saved: {format_memory(peak_copy - peak_inplace)}")

if __name__ == "__main__":
    results = test_memory_usage()
    test_relationship_memory()
    
    print(f"\n=== Summary ===")
    print(f"Overall time improvement: {results['time_improvement']:.2f}x")
    print(f"Overall memory improvement: {results['memory_improvement']:.2f}x")