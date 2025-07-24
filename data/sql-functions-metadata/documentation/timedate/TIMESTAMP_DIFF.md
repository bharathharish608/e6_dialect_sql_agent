# TIMESTAMP_DIFF

Returns the difference between two timestamps

## Syntax

```sql
TIMESTAMP_DIFF( <expression1>, <expression2>, <expression3> )
```

## Arguments



## Returns

- **Type**: BIGINT NULLABLE
- **Description**: Returns the result of the TIMESTAMP_DIFF operation.
- **NULL Handling**: Function behavior with NULL inputs depends on the specific operation.

## Usage Notes

- Returns the difference between two timestamps
- Function accepts STRING, TIMESTAMP, TIMESTAMP
- Return type is BIGINT NULLABLE

## Examples

### Example 1: Basic Usage

Sample data in `employees` table:
```
+----+------------------+------------+--------+------------+----------------------+
| id | name             | department | salary | hire_date  | email                |
+----+------------------+------------+--------+------------+----------------------+
| 1  | John Doe         | Sales      | 50000  | 2020-01-15 | john.doe@company.com |
| 2  | Jane Smith       | Marketing  | 55000  | 2019-03-20 | jane.s@company.com   |
| 3  | Bob Johnson      | IT         | 60000  | 2021-06-10 | bob.j@company.com    |
| 4  | Alice Brown      | HR         | 52000  | 2020-08-05 | alice.b@company.com  |
| 5  | Charlie Wilson   | Sales      | 48000  | 2022-02-14 | charlie.w@company.com|
+----+------------------+------------+--------+------------+----------------------+
```

Query:
```sql
SELECT TIMESTAMP_DIFF('SECOND', '2023-01-01 10:00:00', '2023-01-01 10:00:30') -- Returns 30
```

Result:
```
-- Results will vary based on the function
```

This example demonstrates basic usage of the TIMESTAMP_DIFF function.

### Example 2: NULL Handling

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | Electronics| NULL   | 150            |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | NULL   | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT product_id, product_name, TIMESTAMP_DIFF(price) AS result
FROM products
ORDER BY product_id;
```

Result:
```
-- Results showing NULL handling
```

This example shows how TIMESTAMP_DIFF handles NULL values.

### Example 3: Practical Application

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@company.com  | 555-0123     | Scranton |
| 2           | Dwight     | Schrute     | d.schrute@company.com| 555-0124     | Scranton |
| 3           | Jim        | Halpert     | j.halpert@company.com| 555-0125     | Scranton |
| 4           | Pam        | Beesly      | p.beesly@company.com | 555-0126     | Scranton |
| 5           | Stanley    | Hudson      | s.hudson@company.com | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query:
```sql
-- Practical example using TIMESTAMP_DIFF
SELECT customer_id, first_name, last_name
FROM customers
WHERE customer_id <= 5;
```

Result:
```
-- Results demonstrating practical usage
```

This example shows a practical application of the TIMESTAMP_DIFF function.