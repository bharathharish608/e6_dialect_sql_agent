# COUNTIF

Counts rows where condition is true.

## Syntax

```sql
COUNTIF( <condition> )
```

## Arguments

### condition
- **Type**: BOOLEAN
- **Required**: Yes
- **Description**: Boolean condition to count. The function counts the number of rows where this condition evaluates to TRUE.

## Returns

- **Type**: BIGINT
- **Description**: Returns the count of rows where the condition is TRUE.
- **NULL Handling**: NULL conditions are not counted (treated as FALSE).

## Usage Notes

- COUNTIF only counts rows where the condition evaluates to TRUE
- NULL conditions are not counted
- COUNTIF is equivalent to COUNT(CASE WHEN condition THEN 1 END)
- This function has an alias COUNT_IF
- Useful for conditional counting without CASE expressions

## Examples

### Example 1: Basic Conditional Counting

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
SELECT COUNTIF(salary > 50000) AS high_earners,
       COUNTIF(department = 'Sales') AS sales_employees,
       COUNTIF(YEAR(hire_date) >= 2021) AS recent_hires
FROM employees;
```

Result:
```
+--------------+-----------------+--------------+
| high_earners | sales_employees | recent_hires |
+--------------+-----------------+--------------+
| 3            | 2               | 2            |
+--------------+-----------------+--------------+
```

This example shows using COUNTIF for various conditional counts on the employees table.

### Example 2: NULL Value Handling

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
SELECT COUNTIF(price > 100) AS expensive_items,
       COUNTIF(price <= 100) AS affordable_items,
       COUNTIF(stock_quantity > 50) AS well_stocked,
       COUNTIF(stock_quantity IS NULL) AS unknown_stock
FROM products;
```

Result:
```
+-----------------+------------------+--------------+---------------+
| expensive_items | affordable_items | well_stocked | unknown_stock |
+-----------------+------------------+--------------+---------------+
| 2               | 2                | 1            | 1             |
+-----------------+------------------+--------------+---------------+
```

This example demonstrates how COUNTIF handles NULL values in conditions.

### Example 3: Grouped Conditional Counting

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
       COUNT(*) AS total_orders,
       COUNTIF(status = 'Shipped') AS shipped_orders,
       COUNTIF(status = 'Delivered') AS delivered_orders,
       COUNTIF(quantity > 2) AS bulk_orders
FROM orders
GROUP BY customer_id
ORDER BY customer_id;
```

Result:
```
+-------------+--------------+----------------+------------------+-------------+
| customer_id | total_orders | shipped_orders | delivered_orders | bulk_orders |
+-------------+--------------+----------------+------------------+-------------+
| 1           | 3            | 2              | 1                | 2           |
| 2           | 1            | 0              | 0                | 0           |
| 3           | 1            | 0              | 0                | 0           |
+-------------+--------------+----------------+------------------+-------------+
```

This example shows using COUNTIF with GROUP BY to analyze order patterns by customer.

### Example 4: Complex Conditions

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
       COUNTIF(email IS NOT NULL AND phone IS NOT NULL) AS complete_contact,
       COUNTIF(email LIKE '%@company.com') AS company_emails,
       COUNTIF(LENGTH(first_name) > 5) AS long_first_names
FROM customers
GROUP BY city
ORDER BY total_customers DESC;
```

Result:
```
+----------+-----------------+------------------+----------------+------------------+
| city     | total_customers | complete_contact | company_emails | long_first_names |
+----------+-----------------+------------------+----------------+------------------+
| Scranton | 4               | 2                | 0              | 2                |
| Stamford | 1               | 0                | 0              | 0                |
+----------+-----------------+------------------+----------------+------------------+
```

This example demonstrates using COUNTIF with complex boolean conditions.

### Example 5: Date-based Conditional Counting

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
       COUNT(*) AS total_employees,
       COUNTIF(hire_date >= '2020-01-01') AS hired_2020_or_later,
       COUNTIF(hire_date >= '2021-01-01') AS hired_2021_or_later,
       COUNTIF(MONTH(hire_date) IN (1,2,3)) AS q1_hires
FROM employees
GROUP BY department
ORDER BY department;
```

Result:
```
+------------+-----------------+---------------------+---------------------+----------+
| department | total_employees | hired_2020_or_later | hired_2021_or_later | q1_hires |
+------------+-----------------+---------------------+---------------------+----------+
| HR         | 1               | 1                   | 0                   | 0        |
| IT         | 1               | 1                   | 1                   | 0        |
| Marketing  | 1               | 0                   | 0                   | 1        |
| Sales      | 2               | 2                   | 1                   | 2        |
+------------+-----------------+---------------------+---------------------+----------+
```

This example shows using COUNTIF with date-based conditions to analyze hiring patterns.