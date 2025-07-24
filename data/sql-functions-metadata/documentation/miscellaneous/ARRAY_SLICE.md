# ARRAY_SLICE

Extracts a slice of an array from a starting position to an ending position.

## Syntax

```sql
ARRAY_SLICE( <array>, <start_index>, <end_index> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to extract a slice from

### start_index
- **Type**: INTEGER
- **Required**: Yes
- **Description**: The starting position (1-based index). If negative, counts from the end of the array

### end_index
- **Type**: INTEGER
- **Required**: Yes
- **Description**: The ending position (1-based index, inclusive). If negative, counts from the end of the array

## Returns

- **Type**: ARRAY (same element type as input)
- **Description**: Returns a new array containing elements from start_index to end_index (inclusive)
- **NULL Handling**: Returns NULL if the input array is NULL. Invalid indices are handled gracefully

## Usage Notes

- Uses 1-based indexing (first element is at position 1)
- The slice includes both start and end positions
- Negative indices count from the end (-1 is the last element)
- If start_index > end_index, returns an empty array
- Has alias: SLICE

## Examples

### Example 1: Basic Array Slicing

Sample data in `sales_data` table:
```
+--------+----------+------------+-----------------------------+
| rep_id | rep_name | region     | monthly_sales               |
+--------+----------+------------+-----------------------------+
| 1      | Alice    | North      | [10000,12000,15000,11000,13000] |
| 2      | Bob      | South      | [8000,9000,7500,8500,9500]     |
| 3      | Carol    | East       | [11000,10500,12500,14000,13500] |
| 4      | David    | West       | [9000,9500,10000,11500,12000]  |
| 5      | Eve      | Central    | [7000,8000,8500,9000,10000]    |
+--------+----------+------------+-----------------------------+
```

Query:
```sql
SELECT 
    rep_name,
    monthly_sales,
    ARRAY_SLICE(monthly_sales, 2, 4) AS q2_sales
FROM sales_data
ORDER BY rep_id;
```

Result:
```
+----------+-----------------------------+-------------------+
| rep_name | monthly_sales               | q2_sales          |
+----------+-----------------------------+-------------------+
| Alice    | [10000,12000,15000,11000,13000] | [12000,15000,11000] |
| Bob      | [8000,9000,7500,8500,9500]     | [9000,7500,8500]   |
| Carol    | [11000,10500,12500,14000,13500] | [10500,12500,14000] |
| David    | [9000,9500,10000,11500,12000]  | [9500,10000,11500]  |
| Eve      | [7000,8000,8500,9000,10000]    | [8000,8500,9000]    |
+----------+-----------------------------+-------------------+
```

This example extracts the second quarter (months 2-4) sales from the monthly sales array.

### Example 2: Using Negative Indices

Sample data in `product_versions` table:
```
+------------+--------------+----------------------------------------+
| product_id | product_name | version_history                        |
+------------+--------------+----------------------------------------+
| 1          | App Alpha    | ['1.0','1.1','1.2','2.0','2.1','2.2'] |
| 2          | App Beta     | ['0.1','0.2','1.0','1.1']             |
| 3          | App Gamma    | ['1.0','2.0','3.0','3.1','3.2']       |
| 4          | App Delta    | ['1.0','1.5','2.0']                   |
| 5          | App Epsilon  | ['0.5','1.0','1.1','1.2','2.0']       |
+------------+--------------+----------------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    version_history,
    ARRAY_SLICE(version_history, -3, -1) AS last_three_versions
FROM product_versions
ORDER BY product_id;
```

Result:
```
+--------------+----------------------------------------+---------------------+
| product_name | version_history                        | last_three_versions |
+--------------+----------------------------------------+---------------------+
| App Alpha    | ['1.0','1.1','1.2','2.0','2.1','2.2'] | ['2.0','2.1','2.2'] |
| App Beta     | ['0.1','0.2','1.0','1.1']             | ['0.2','1.0','1.1'] |
| App Gamma    | ['1.0','2.0','3.0','3.1','3.2']       | ['3.0','3.1','3.2'] |
| App Delta    | ['1.0','1.5','2.0']                   | ['1.0','1.5','2.0'] |
| App Epsilon  | ['0.5','1.0','1.1','1.2','2.0']       | ['1.1','1.2','2.0'] |
+--------------+----------------------------------------+---------------------+
```

This example uses negative indices to get the last three versions of each product.

### Example 3: Extracting First N Elements

Sample data in `server_logs` table:
```
+----------+-------------+------------------------------------------+
| server_id| server_name | error_codes                              |
+----------+-------------+------------------------------------------+
| 1        | web-01      | [404,500,503,404,200,301,404]          |
| 2        | web-02      | [200,200,404,500]                       |
| 3        | db-01       | [1045,1062,1064,1045,1146]             |
| 4        | api-01      | [401,403,429,500,503]                   |
| 5        | cache-01    | [2006,2013,2006]                        |
+----------+-------------+------------------------------------------+
```

Query:
```sql
SELECT 
    server_name,
    error_codes,
    ARRAY_SLICE(error_codes, 1, 3) AS first_three_errors,
    ARRAY_SIZE(error_codes) AS total_errors
FROM server_logs
WHERE ARRAY_SIZE(error_codes) > 0
ORDER BY server_id;
```

Result:
```
+-------------+------------------------------------------+--------------------+--------------+
| server_name | error_codes                              | first_three_errors | total_errors |
+-------------+------------------------------------------+--------------------+--------------+
| web-01      | [404,500,503,404,200,301,404]          | [404,500,503]      | 7            |
| web-02      | [200,200,404,500]                       | [200,200,404]      | 4            |
| db-01       | [1045,1062,1064,1045,1146]             | [1045,1062,1064]   | 5            |
| api-01      | [401,403,429,500,503]                   | [401,403,429]      | 5            |
| cache-01    | [2006,2013,2006]                        | [2006,2013,2006]   | 3            |
+-------------+------------------------------------------+--------------------+--------------+
```

This example extracts the first three error codes from each server's error log.

### Example 4: NULL and Edge Case Handling

Sample data in `experiment_results` table:
```
+--------+----------------+--------------------------------+
| exp_id | experiment_name| measurements                   |
+--------+----------------+--------------------------------+
| 1      | Test A         | [1.2,1.5,1.3,1.4,1.6]         |
| 2      | Test B         | NULL                           |
| 3      | Test C         | [2.1,2.3]                      |
| 4      | Test D         | []                             |
| 5      | Test E         | [3.1,3.2,3.3,3.4,3.5,3.6,3.7] |
+--------+----------------+--------------------------------+
```

Query:
```sql
SELECT 
    experiment_name,
    measurements,
    ARRAY_SLICE(measurements, 2, 5) AS mid_measurements,
    ARRAY_SLICE(measurements, 4, 2) AS reversed_slice,
    ARRAY_SLICE(measurements, 10, 20) AS out_of_bounds
FROM experiment_results
ORDER BY exp_id;
```

Result:
```
+----------------+--------------------------------+------------------+----------------+---------------+
| experiment_name| measurements                   | mid_measurements | reversed_slice | out_of_bounds |
+----------------+--------------------------------+------------------+----------------+---------------+
| Test A         | [1.2,1.5,1.3,1.4,1.6]         | [1.5,1.3,1.4,1.6]| []             | []            |
| Test B         | NULL                           | NULL             | NULL           | NULL          |
| Test C         | [2.1,2.3]                      | [2.3]            | []             | []            |
| Test D         | []                             | []               | []             | []            |
| Test E         | [3.1,3.2,3.3,3.4,3.5,3.6,3.7] | [3.2,3.3,3.4,3.5]| []             | []            |
+----------------+--------------------------------+------------------+----------------+---------------+
```

This example demonstrates NULL handling, empty arrays, reversed indices, and out-of-bounds slicing.

### Example 5: Using SLICE Alias with Dynamic Indices

Sample data in `customer_orders` table:
```
+-------------+---------------+----------------------------------------+-------------+
| customer_id | customer_name | order_ids                              | vip_level   |
+-------------+---------------+----------------------------------------+-------------+
| 1           | John Smith    | [1001,1002,1003,1004,1005,1006]      | 3           |
| 2           | Jane Doe      | [2001,2002,2003,2004]                 | 2           |
| 3           | Bob Wilson    | [3001,3002,3003,3004,3005,3006,3007] | 4           |
| 4           | Alice Brown   | [4001,4002]                           | 1           |
| 5           | Charlie Lee   | [5001,5002,5003,5004,5005]            | 3           |
+-------------+---------------+----------------------------------------+-------------+
```

Query:
```sql
SELECT 
    customer_name,
    order_ids,
    vip_level,
    SLICE(order_ids, 1, vip_level) AS vip_orders,
    SLICE(order_ids, -vip_level, -1) AS recent_vip_orders
FROM customer_orders
ORDER BY customer_id;
```

Result:
```
+---------------+----------------------------------------+-----------+--------------------+----------------------+
| customer_name | order_ids                              | vip_level | vip_orders         | recent_vip_orders    |
+---------------+----------------------------------------+-----------+--------------------+----------------------+
| John Smith    | [1001,1002,1003,1004,1005,1006]      | 3         | [1001,1002,1003]   | [1004,1005,1006]     |
| Jane Doe      | [2001,2002,2003,2004]                 | 2         | [2001,2002]        | [2003,2004]          |
| Bob Wilson    | [3001,3002,3003,3004,3005,3006,3007] | 4         | [3001,3002,3003,3004] | [3004,3005,3006,3007] |
| Alice Brown   | [4001,4002]                           | 1         | [4001]             | [4002]               |
| Charlie Lee   | [5001,5002,5003,5004,5005]            | 3         | [5001,5002,5003]   | [5003,5004,5005]     |
+---------------+----------------------------------------+-----------+--------------------+----------------------+
```

This example uses the SLICE alias and demonstrates dynamic slicing based on the customer's VIP level.