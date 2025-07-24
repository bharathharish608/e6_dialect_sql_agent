# COUNT

Counts the number of rows or non-null values.

## Syntax

```sql
COUNT( * )
COUNT( <expression> )
COUNT( DISTINCT <expression> )
```

## Arguments

### expression
- **Type**: ANY or *
- **Required**: Yes
- **Description**: Column to count or * for all rows. When * is used, counts all rows including those with NULL values. When a column is specified, counts only non-NULL values in that column.

## Returns

- **Type**: BIGINT
- **Description**: Returns the count of rows or non-null values as a BIGINT.
- **NULL Handling**: When counting a specific column, NULL values are not counted. When using COUNT(*), all rows are counted regardless of NULL values.

## Usage Notes

- COUNT(*) counts all rows in the result set, including rows with NULL values
- COUNT(column) counts only non-NULL values in the specified column
- COUNT(DISTINCT column) counts unique non-NULL values
- COUNT is an aggregate function and is often used with GROUP BY
- Returns 0 when no rows match the criteria

## Examples

### Example 1: Basic Row Counting

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
SELECT COUNT(*) AS total_employees,
       COUNT(DISTINCT department) AS num_departments
FROM employees;
```

Result:
```
+-----------------+-----------------+
| total_employees | num_departments |
+-----------------+-----------------+
| 5               | 4               |
+-----------------+-----------------+
```

This example shows counting all rows with COUNT(*) and counting distinct departments.

### Example 2: Handling NULL Values

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | Electronics| 29.99  | NULL           |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | NULL   | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT COUNT(*) AS all_products,
       COUNT(price) AS products_with_price,
       COUNT(stock_quantity) AS products_with_stock
FROM products;
```

Result:
```
+--------------+---------------------+---------------------+
| all_products | products_with_price | products_with_stock |
+--------------+---------------------+---------------------+
| 5            | 4                   | 4                   |
+--------------+---------------------+---------------------+
```

This example demonstrates how COUNT(*) counts all rows while COUNT(column) excludes NULL values.

### Example 3: Grouping with COUNT

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
       COUNT(DISTINCT product_id) AS unique_products_ordered
FROM orders
GROUP BY customer_id
ORDER BY order_count DESC;
```

Result:
```
+-------------+-------------+-------------------------+
| customer_id | order_count | unique_products_ordered |
+-------------+-------------+-------------------------+
| 1           | 3           | 3                       |
| 2           | 1           | 1                       |
| 3           | 1           | 1                       |
+-------------+-------------+-------------------------+
```

This example shows using COUNT with GROUP BY to analyze order patterns by customer.

### Example 4: COUNT with Filtering

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@dundermif.com| 555-0123     | Scranton |
| 2           | Dwight     | Schrute     | d.schrute@farms.com  | 555-0124     | Scranton |
| 3           | Jim        | Halpert     | j.halpert@sales.com  | NULL         | Stamford |
| 4           | Pam        | Beesly      | p.beesly@reception.com| 555-0126    | Scranton |
| 5           | Stanley    | Hudson      | NULL                 | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query:
```sql
SELECT city,
       COUNT(*) AS total_customers,
       COUNT(email) AS customers_with_email,
       COUNT(phone) AS customers_with_phone
FROM customers
GROUP BY city
ORDER BY total_customers DESC;
```

Result:
```
+----------+-----------------+----------------------+----------------------+
| city     | total_customers | customers_with_email | customers_with_phone |
+----------+-----------------+----------------------+----------------------+
| Scranton | 4               | 3                    | 3                    |
| Stamford | 1               | 1                    | 0                    |
+----------+-----------------+----------------------+----------------------+
```

This example demonstrates counting with NULL values across different columns grouped by city.

### Example 5: Complex COUNT Operations

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
SELECT department,
       COUNT(*) AS dept_size,
       COUNT(CASE WHEN salary > 50000 THEN 1 END) AS high_earners,
       COUNT(CASE WHEN YEAR(hire_date) >= 2021 THEN 1 END) AS recent_hires
FROM employees
GROUP BY department
ORDER BY dept_size DESC;
```

Result:
```
+------------+-----------+--------------+--------------+
| department | dept_size | high_earners | recent_hires |
+------------+-----------+--------------+--------------+
| Sales      | 2         | 0            | 1            |
| HR         | 1         | 1            | 0            |
| IT         | 1         | 1            | 1            |
| Marketing  | 1         | 1            | 0            |
+------------+-----------+--------------+--------------+
```

This example shows using COUNT with CASE expressions to create conditional counts within groups.