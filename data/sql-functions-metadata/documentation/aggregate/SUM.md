# SUM

Calculates the sum of numeric values.

## Syntax

```sql
SUM( <numeric_expression> )
```

## Arguments

### numeric_expression
- **Type**: NUMERIC
- **Required**: Yes
- **Description**: Numeric column to sum. Can be any numeric data type including INTEGER, BIGINT, DECIMAL, FLOAT, or DOUBLE.

## Returns

- **Type**: Sum type of input
- **Description**: Returns the sum of all non-NULL numeric values. The return type depends on the input type but is generally promoted to handle larger values.
- **NULL Handling**: NULL values are ignored in the sum calculation. If all values are NULL, returns NULL.

## Usage Notes

- SUM ignores NULL values in calculations
- Returns NULL if all values are NULL or if there are no rows
- Can overflow if the sum exceeds the data type's maximum value
- Often used with GROUP BY for subtotals
- Can be used with expressions, not just column references

## Examples

### Example 1: Basic Sum Calculation

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
SELECT SUM(salary) AS total_salary,
       COUNT(*) AS employee_count,
       SUM(salary) / COUNT(*) AS avg_salary_calc
FROM employees;
```

Result:
```
+--------------+----------------+-----------------+
| total_salary | employee_count | avg_salary_calc |
+--------------+----------------+-----------------+
| 265000       | 5              | 53000.0         |
+--------------+----------------+-----------------+
```

This example shows calculating the total salary sum and using it to compute average.

### Example 2: Handling NULL Values

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | Electronics| 29.99  | 150            |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | NULL   | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT COUNT(*) AS total_products,
       COUNT(price) AS products_with_price,
       SUM(price) AS total_price,
       SUM(price * stock_quantity) AS potential_revenue
FROM products;
```

Result:
```
+----------------+---------------------+-------------+-------------------+
| total_products | products_with_price | total_price | potential_revenue |
+----------------+---------------------+-------------+-------------------+
| 5              | 4                   | 1599.96     | 76494.50          |
+----------------+---------------------+-------------+-------------------+
```

This example demonstrates that SUM ignores NULL values in calculations.

### Example 3: Grouped Sums

Sample data in `orders` table:
```
+----------+-------------+------------+----------+------------+-----------+
| order_id | customer_id | product_id | quantity | order_date | status    |
+----------+-------------+------------+----------+------------+-----------+
| 1001     | 1           | 1          | 2        | 2025-01-05 | Shipped   |
| 1002     | 2           | 3          | 1        | 2025-01-06 | Pending   |
| 1003     | 1           | 2          | 5        | 2025-01-07 | Delivered |
| 1004     | 3           | 4          | 1        | 2025-01-08 | Processing|
| 1005     | 1           | 5          | 10       | 2025-01-09 | Shipped   |
+----------+-------------+------------+----------+------------+-----------+
```

Query:
```sql
SELECT customer_id,
       COUNT(*) AS order_count,
       SUM(quantity) AS total_items_ordered,
       SUM(CASE WHEN status = 'Shipped' THEN quantity ELSE 0 END) AS shipped_items
FROM orders
GROUP BY customer_id
ORDER BY total_items_ordered DESC;
```

Result:
```
+-------------+-------------+---------------------+---------------+
| customer_id | order_count | total_items_ordered | shipped_items |
+-------------+-------------+---------------------+---------------+
| 1           | 3           | 17                  | 12            |
| 2           | 1           | 1                   | 0             |
| 3           | 1           | 1                   | 0             |
+-------------+-------------+---------------------+---------------+
```

This example shows using SUM with GROUP BY to calculate totals per customer.

### Example 4: SUM with Expressions

Sample data in `customers` table with orders:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@dundermif.com| 555-0123     | Scranton |
| 2           | Dwight     | Schrute     | d.schrute@farms.com  | 555-0124     | Scranton |
| 3           | Jim        | Halpert     | j.halpert@sales.com  | 555-0125     | Stamford |
| 4           | Pam        | Beesly      | p.beesly@reception.com| 555-0126    | Scranton |
| 5           | Stanley    | Hudson      | s.hudson@sales.com   | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query using products table for pricing:
```sql
SELECT p.category,
       COUNT(*) AS product_count,
       SUM(p.price) AS total_value,
       SUM(p.price * p.stock_quantity) AS inventory_value,
       SUM(p.price * 0.2) AS total_tax_20_percent
FROM products p
WHERE p.price IS NOT NULL
GROUP BY p.category
ORDER BY inventory_value DESC;
```

Result:
```
+-------------+---------------+-------------+-----------------+----------------------+
| category    | product_count | total_value | inventory_value | total_tax_20_percent |
+-------------+---------------+-------------+-----------------+----------------------+
| Electronics | 3             | 1349.97     | 69494.50        | 269.994              |
| Furniture   | 1             | 249.99      | 7499.70         | 49.998               |
+-------------+---------------+-------------+-----------------+----------------------+
```

This example demonstrates using SUM with calculated expressions.

### Example 5: Monthly Sales Analysis

Sample data in `employees` table with monthly targets:
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
SELECT department,
       COUNT(*) AS dept_size,
       SUM(salary) AS total_salary_cost,
       SUM(salary) / 12 AS monthly_salary_cost,
       SUM(salary * 1.3) AS cost_with_benefits
FROM employees
GROUP BY department
ORDER BY total_salary_cost DESC;
```

Result:
```
+------------+-----------+-------------------+---------------------+--------------------+
| department | dept_size | total_salary_cost | monthly_salary_cost | cost_with_benefits |
+------------+-----------+-------------------+---------------------+--------------------+
| Sales      | 2         | 98000             | 8166.67             | 127400.0           |
| IT         | 1         | 60000             | 5000.00             | 78000.0            |
| Marketing  | 1         | 55000             | 4583.33             | 71500.0            |
| HR         | 1         | 52000             | 4333.33             | 67600.0            |
+------------+-----------+-------------------+---------------------+--------------------+
```

This example shows using SUM for financial calculations and projections by department.