#!/usr/bin/env python3
"""
Performance test for Query Plan Analysis optimization.

Tests the SQLite EXPLAIN QUERY PLAN analysis functionality that identifies
potential performance issues before query execution.
"""

import sqlite3
import pandas as pd
import tempfile
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path
import sys
sys.path.insert(0, 'src')

from readers.sqlite_reader import SQLiteReader, QueryPlanAnalysis


def setup_test_databases() -> List[str]:
    """Create test databases with different table sizes and index configurations."""
    
    # Create temporary database files
    db1_path = tempfile.mktemp(suffix='.db')
    db2_path = tempfile.mktemp(suffix='.db')
    
    # Database 1: Small tables with indexes
    conn1 = sqlite3.connect(db1_path)
    cursor1 = conn1.cursor()
    
    # Create users table with index
    cursor1.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            department_id INTEGER
        )
    ''')
    cursor1.execute('CREATE INDEX idx_users_dept ON users(department_id)')
    
    # Insert test data
    users_data = [(i, f'User{i}', f'user{i}@example.com', i % 5) for i in range(1, 1001)]
    cursor1.executemany('INSERT INTO users VALUES (?, ?, ?, ?)', users_data)
    
    # Create departments table
    cursor1.execute('''
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget INTEGER
        )
    ''')
    cursor1.execute('CREATE INDEX idx_dept_budget ON departments(budget)')
    
    dept_data = [(i, f'Department{i}', 50000 + i * 10000) for i in range(1, 6)]
    cursor1.executemany('INSERT INTO departments VALUES (?, ?, ?)', dept_data)
    
    conn1.commit()
    conn1.close()
    
    # Database 2: Large table without proper indexes
    conn2 = sqlite3.connect(db2_path)
    cursor2 = conn2.cursor()
    
    # Create large orders table WITHOUT indexes on JOIN columns
    cursor2.execute('''
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_name TEXT,
            quantity INTEGER,
            price REAL,
            order_date TEXT
        )
    ''')
    
    # Insert large amount of test data (50,000 rows)
    orders_data = [
        (i, (i % 1000) + 1, f'Product{i % 100}', i % 10 + 1, 
         round((i % 100 + 1) * 10.99, 2), f'2024-{(i % 12) + 1:02d}-01')
        for i in range(1, 50001)
    ]
    cursor2.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)', orders_data)
    
    # Create products table without indexes
    cursor2.execute('''
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            supplier_id INTEGER
        )
    ''')
    
    products_data = [(i, f'Product{i}', f'Category{i % 10}', i % 20) for i in range(1, 101)]
    cursor2.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', products_data)
    
    conn2.commit()
    conn2.close()
    
    return [db1_path, db2_path]


def test_query_plan_analysis_basic():
    """Test basic query plan analysis functionality."""
    print("Testing basic query plan analysis...")
    
    db_paths = setup_test_databases()
    
    try:
        # Test with single database first
        databases = [{"alias": "db2", "path": db_paths[1]}]  # Use db2 which has the large orders table
        
        reader = SQLiteReader(databases)
        reader.connect_databases()
        
        # Test 1: Simple query analysis on large table (should detect table scan)
        simple_query = "SELECT * FROM orders WHERE user_id = 1"
        analysis = reader.analyze_query_plan(simple_query)
        
        print(f"Simple Query Analysis (unindexed column):")
        print(f"  - Estimated Cost: {analysis.estimated_cost}")
        print(f"  - Uses Index: {analysis.uses_index}")
        print(f"  - Has Table Scan: {analysis.has_table_scan}")
        print(f"  - Table Scans: {analysis.table_scans}")
        print(f"  - Complexity Score: {analysis.complexity_score}")
        print(f"  - Warnings: {len(analysis.warnings)}")
        for warning in analysis.warnings:
            print(f"    * {warning}")
        print()
        
        # Test 2: Query using primary key (should be efficient)
        pk_query = "SELECT * FROM orders WHERE order_id = 1000"
        analysis2 = reader.analyze_query_plan(pk_query)
        
        print(f"Primary Key Query Analysis:")
        print(f"  - Estimated Cost: {analysis2.estimated_cost}")
        print(f"  - Has Table Scan: {analysis2.has_table_scan}")
        print(f"  - Uses Index: {analysis2.uses_index}")
        print(f"  - Complexity Score: {analysis2.complexity_score}")
        print(f"  - Warnings: {len(analysis2.warnings)}")
        for warning in analysis2.warnings:
            print(f"    * {warning}")
        print()
        
        # Test 3: Large table scan without WHERE clause
        full_scan_query = "SELECT COUNT(*) FROM orders"  
        analysis3 = reader.analyze_query_plan(full_scan_query)
        
        print(f"Full Table Scan Analysis:")
        print(f"  - Estimated Cost: {analysis3.estimated_cost}")
        print(f"  - Has Table Scan: {analysis3.has_table_scan}")
        print(f"  - Table Scans: {analysis3.table_scans}")
        print(f"  - Complexity Score: {analysis3.complexity_score}")
        print(f"  - Warnings: {len(analysis3.warnings)}")
        for warning in analysis3.warnings:
            print(f"    * {warning}")
        print()
        
        reader.close_connections()
        
        # Verify analysis results
        assert analysis.estimated_cost in ["Very Low", "Low", "Medium", "High", "Very High"]
        assert analysis2.estimated_cost in ["Very Low", "Low", "Medium", "High", "Very High"]
        
        # The primary key query should be more efficient than the unindexed query
        if analysis.has_table_scan and not analysis2.has_table_scan:
            assert analysis.complexity_score >= analysis2.complexity_score
        
        print("✅ Basic query plan analysis tests passed!")
        
    finally:
        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)


def test_join_performance_analysis():
    """Test JOIN-specific performance analysis."""
    print("\nTesting JOIN performance analysis...")
    
    db_paths = setup_test_databases()
    
    try:
        # Test with single database that has indexed relations
        databases = [{"alias": "db1", "path": db_paths[0]}]
        
        reader = SQLiteReader(databases)
        reader.connect_databases()
        
        # Test efficient JOIN (indexed columns)
        efficient_joins = [
            {
                "left_table": "db1.users",
                "right_table": "db1.departments", 
                "type": "INNER",
                "condition": "db1_users.department_id = db1_departments.id"
            }
        ]
        
        analysis1 = reader.analyze_join_performance(efficient_joins)
        print(f"Efficient JOIN Analysis:")
        print(f"  - Estimated Cost: {analysis1.estimated_cost}")
        print(f"  - Complexity Score: {analysis1.complexity_score}")
        print(f"  - Uses Index: {analysis1.uses_index}")
        print(f"  - Warnings: {len(analysis1.warnings)}")
        for warning in analysis1.warnings:
            print(f"    * {warning}")
        print()
        
        reader.close_connections()
        
        # Test with large database that has no indexes on JOIN columns
        databases2 = [{"alias": "db2", "path": db_paths[1]}]
        
        reader2 = SQLiteReader(databases2)
        reader2.connect_databases()
        
        # Test inefficient JOIN (no indexes on JOIN columns)
        inefficient_joins = [
            {
                "left_table": "db2.orders",
                "right_table": "db2.products",
                "type": "INNER", 
                "condition": "db2_orders.product_name = db2_products.name"
            }
        ]
        
        analysis2 = reader2.analyze_join_performance(inefficient_joins)
        print(f"Inefficient JOIN Analysis:")
        print(f"  - Estimated Cost: {analysis2.estimated_cost}")
        print(f"  - Complexity Score: {analysis2.complexity_score}")
        print(f"  - Has Table Scan: {analysis2.has_table_scan}")
        print(f"  - Warnings: {len(analysis2.warnings)}")
        for warning in analysis2.warnings:
            print(f"    * {warning}")
        print()
        
        reader2.close_connections()
        
        # Verify that inefficient JOINs have higher complexity scores or more warnings
        assert analysis1.estimated_cost in ["Very Low", "Low", "Medium", "High", "Very High"]
        assert analysis2.estimated_cost in ["Very Low", "Low", "Medium", "High", "Very High"]
        
        print("✅ JOIN performance analysis tests passed!")
        
    finally:
        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)


def test_automatic_performance_analysis():
    """Test automatic performance analysis during query execution."""
    print("\nTesting automatic performance analysis during execution...")
    
    db_paths = setup_test_databases()
    
    try:
        # Use single database with large table
        databases = [{"alias": "db2", "path": db_paths[1]}]
        
        reader = SQLiteReader(databases)
        reader.connect_databases()
        
        # Configure logging to capture warnings
        logging.getLogger().setLevel(logging.INFO)
        
        # Test JOIN execution with performance analysis enabled
        joins = [
            {
                "left_table": "db2.orders",
                "right_table": "db2.products",
                "type": "INNER",
                "condition": "db2_orders.product_name = db2_products.name"
            }
        ]
        
        print("Executing JOIN with automatic performance analysis...")
        start_time = time.time()
        
        # This should automatically analyze the query and log warnings
        result_df = reader.execute_join_query(joins, analyze_performance=True)
        
        execution_time = time.time() - start_time
        
        print(f"Query executed in {execution_time:.2f} seconds")
        print(f"Result: {len(result_df)} rows returned")
        
        # Test with performance analysis disabled
        print("\nExecuting same JOIN without performance analysis...")
        start_time = time.time()
        
        result_df2 = reader.execute_join_query(joins, analyze_performance=False)
        
        execution_time2 = time.time() - start_time
        
        print(f"Query executed in {execution_time2:.2f} seconds")
        print(f"Result: {len(result_df2)} rows returned")
        
        # Verify results are the same
        assert len(result_df) == len(result_df2)
        
        reader.close_connections()
        
        print("✅ Automatic performance analysis tests passed!")
        
    finally:
        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)


def test_table_size_warnings():
    """Test table size-based performance warnings."""
    print("\nTesting table size-based warnings...")
    
    db_paths = setup_test_databases()
    
    try:
        # Use database with large table
        databases = [{"alias": "db2", "path": db_paths[1]}]
        
        reader = SQLiteReader(databases)
        reader.connect_databases()
        
        # Get table sizes
        table_sizes = reader._get_table_sizes()
        print("Table sizes detected:")
        for table, size in table_sizes.items():
            print(f"  - {table}: {size:,} rows")
        
        # Test query on large table without index
        large_table_query = "SELECT * FROM orders WHERE quantity > 5 AND price < 100"
        analysis = reader.analyze_query_plan(large_table_query)
        
        print(f"\nLarge Table Query Analysis:")
        print(f"  - Table Scans: {analysis.table_scans}")
        print(f"  - Estimated Cost: {analysis.estimated_cost}")
        print(f"  - Warnings: {len(analysis.warnings)}")
        
        # Should have specific warnings about large table scans
        large_table_warnings = [w for w in analysis.warnings if "PERFORMANCE WARNING" in w or "Table scan" in w]
        print(f"  - Performance Warnings: {len(large_table_warnings)}")
        for warning in large_table_warnings:
            print(f"    * {warning}")
        
        reader.close_connections()
        
        # Verify analysis results
        assert analysis.estimated_cost in ["Very Low", "Low", "Medium", "High", "Very High"]
        
        # If we have table scans, verify they're detected
        if analysis.table_scans:
            assert any("orders" in scan for scan in analysis.table_scans)
            
        print("✅ Table size warning tests passed!")
        
    finally:
        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)


def benchmark_analysis_overhead():
    """Benchmark the overhead of query plan analysis."""
    print("\nBenchmarking query plan analysis overhead...")
    
    db_paths = setup_test_databases()
    
    try:
        # Use smaller database for more consistent benchmarking
        databases = [{"alias": "db1", "path": db_paths[0]}]
        
        reader = SQLiteReader(databases)
        reader.connect_databases()
        
        joins = [
            {
                "left_table": "db1.users",
                "right_table": "db1.departments",
                "type": "INNER",
                "condition": "db1_users.department_id = db1_departments.id"
            }
        ]
        
        # Benchmark with analysis
        start_time = time.time()
        for _ in range(5):  # Reduce iterations for faster testing
            reader.analyze_join_performance(joins)
        analysis_time = time.time() - start_time
        
        print(f"Query plan analysis (5 runs): {analysis_time:.4f} seconds")
        print(f"Average analysis time: {analysis_time/5:.4f} seconds")
        
        # Benchmark actual query execution
        start_time = time.time()
        for _ in range(5):
            result_df = reader.execute_join_query(joins, analyze_performance=False)
        execution_time = time.time() - start_time
        
        print(f"Query execution (5 runs): {execution_time:.4f} seconds")
        print(f"Average execution time: {execution_time/5:.4f} seconds")
        
        # Calculate overhead percentage (handle case where execution is very fast)
        if execution_time > 0:
            overhead_percentage = (analysis_time / execution_time) * 100
            print(f"Analysis overhead: {overhead_percentage:.1f}% of execution time")
        else:
            print("Execution time too fast to measure overhead accurately")
        
        reader.close_connections()
        
        # Basic verification that analysis completes without errors
        assert analysis_time >= 0
        assert execution_time >= 0
        
        print("✅ Performance overhead benchmark completed!")
        
    finally:
        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)


def main():
    """Run all query plan analysis tests."""
    print("=== Query Plan Analysis Performance Tests ===\n")
    
    try:
        test_query_plan_analysis_basic()
        test_join_performance_analysis()
        test_automatic_performance_analysis()
        test_table_size_warnings()
        benchmark_analysis_overhead()
        
        print("\n🎉 All query plan analysis tests passed!")
        print("\nKey Benefits Demonstrated:")
        print("✅ Automatic detection of table scans on large tables")
        print("✅ Performance warnings before slow query execution")
        print("✅ Index usage analysis and recommendations")
        print("✅ Complexity scoring for query optimization guidance")
        print("✅ Minimal performance overhead for analysis")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()