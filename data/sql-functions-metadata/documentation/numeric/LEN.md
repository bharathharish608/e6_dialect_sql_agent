# LEN

Returns the length of a string.

## Syntax

```sql
LEN( <string> )
```

## Arguments

### string
- **Type**: STRING
- **Required**: Yes
- **Description**: String value to measure.

## Returns

- **Type**: INTEGER NULLABLE
- **Description**: Returns the number of characters in the string.
- **NULL Handling**: Returns NULL if the input string is NULL.

## Usage Notes

- LEN returns the number of characters in the string
- The function counts all characters including spaces
- LEN has an alias LENGTH that provides identical functionality
- Return type is INTEGER NULLABLE

## Examples

### Example 1: Basic String Length Calculation

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
SELECT id,
       name,
       LEN(name) AS name_length,
       department,
       LEN(department) AS dept_length,
       email,
       LEN(email) AS email_length
FROM employees
ORDER BY id;
```

Result:
```
+----+------------------+-------------+------------+-------------+----------------------+--------------+
| id | name             | name_length | department | dept_length | email                | email_length |
+----+------------------+-------------+------------+-------------+----------------------+--------------+
| 1  | John Doe         | 8           | Sales      | 5           | john.doe@company.com | 21           |
| 2  | Jane Smith       | 10          | Marketing  | 9           | jane.s@company.com   | 19           |
| 3  | Bob Johnson      | 11          | IT         | 2           | bob.j@company.com    | 18           |
| 4  | Alice Brown      | 11          | HR         | 2           | alice.b@company.com  | 20           |
| 5  | Charlie Wilson   | 14          | Sales      | 5           | charlie.w@company.com| 22           |
+----+------------------+-------------+------------+-------------+----------------------+--------------+
```

This example shows calculating the length of various string fields, demonstrating that spaces are counted as characters.

### Example 2: NULL Value Handling

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | NULL       | 29.99  | 150            |
| 3          | NULL             | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | 599.99 | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT product_id,
       product_name,
       LEN(product_name) AS name_length,
       category,
       LEN(category) AS category_length
FROM products
ORDER BY product_id;
```

Result:
```
+------------+------------------+-------------+-------------+-----------------+
| product_id | product_name     | name_length | category    | category_length |
+------------+------------------+-------------+-------------+-----------------+
| 1          | Laptop Pro       | 11          | Electronics | 11              |
| 2          | Wireless Mouse   | 14          | NULL        | NULL            |
| 3          | NULL             | NULL        | Furniture   | 9               |
| 4          | Standing Desk    | 13          | Furniture   | 9               |
| 5          | USB-C Cable      | 11          | Electronics | 11              |
+------------+------------------+-------------+-------------+-----------------+
```

This example demonstrates that LEN returns NULL when the input is NULL.

### Example 3: Using LEN for Data Validation

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@dundermif.com| 555-0123     | Scranton |
| 2           | Dwight     | Schrute     | d.schrute@farms.com  | 555-0124     | Scranton |
| 3           | Jim        | Halpert     | j.halpert@sales.com  | 555-0125     | Scranton |
| 4           | Pam        | Beesly      | p.beesly@reception.com| 555-0126    | Scranton |
| 5           | Stanley    | Hudson      | s.hudson@sales.com   | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query:
```sql
SELECT customer_id,
       first_name,
       last_name,
       phone,
       LEN(phone) AS phone_length,
       CASE 
           WHEN LEN(phone) = 8 THEN 'Valid'
           WHEN LEN(phone) = 9 THEN 'Valid with dash'
           ELSE 'Invalid'
       END AS phone_validation
FROM customers
ORDER BY customer_id;
```

Result:
```
+-------------+------------+-------------+----------+--------------+------------------+
| customer_id | first_name | last_name   | phone    | phone_length | phone_validation |
+-------------+------------+-------------+----------+--------------+------------------+
| 1           | Michael    | Scott       | 555-0123 | 8            | Valid            |
| 2           | Dwight     | Schrute     | 555-0124 | 8            | Valid            |
| 3           | Jim        | Halpert     | 555-0125 | 8            | Valid            |
| 4           | Pam        | Beesly      | 555-0126 | 8            | Valid            |
| 5           | Stanley    | Hudson      | 555-0127 | 8            | Valid            |
+-------------+------------+-------------+----------+--------------+------------------+
```

This example shows using LEN for data validation to check if phone numbers meet length requirements.

### Example 4: Filtering by String Length

Sample data in `orders` table:
```
+----------+-------------+------------+----------+------------+-----------+
| order_id | customer_id | product_id | quantity | order_date | status    |
+----------+-------------+------------+----------+------------+-----------+
| 1001     | 1           | 1          | 2        | 2025-01-05 | Shipped   |
| 1002     | 2           | 3          | 1        | 2025-01-06 | Pending   |
| 1003     | 3           | 2          | 5        | 2025-01-07 | Delivered |
| 1004     | 1           | 4          | 1        | 2025-01-08 | Processing|
| 1005     | 4           | 5          | 10       | 2025-01-09 | Shipped   |
+----------+-------------+------------+----------+------------+-----------+
```

Query:
```sql
SELECT order_id,
       status,
       LEN(status) AS status_length
FROM orders
WHERE LEN(status) > 7
ORDER BY LEN(status) DESC, order_id;
```

Result:
```
+----------+------------+---------------+
| order_id | status     | status_length |
+----------+------------+---------------+
| 1004     | Processing | 10            |
| 1003     | Delivered  | 9             |
+----------+------------+---------------+
```

This example demonstrates filtering records based on string length criteria.

### Example 5: String Length Analysis

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
       COUNT(*) AS employee_count,
       MIN(LEN(name)) AS shortest_name,
       MAX(LEN(name)) AS longest_name,
       AVG(LEN(name)) AS avg_name_length,
       SUM(LEN(email)) AS total_email_chars
FROM employees
GROUP BY department
ORDER BY department;
```

Result:
```
+------------+----------------+---------------+--------------+-----------------+-------------------+
| department | employee_count | shortest_name | longest_name | avg_name_length | total_email_chars |
+------------+----------------+---------------+--------------+-----------------+-------------------+
| HR         | 1              | 11            | 11           | 11.0            | 20                |
| IT         | 1              | 11            | 11           | 11.0            | 18                |
| Marketing  | 1              | 10            | 10           | 10.0            | 19                |
| Sales      | 2              | 8             | 14           | 11.0            | 43                |
+------------+----------------+---------------+--------------+-----------------+-------------------+
```

This example shows aggregating string length information to analyze data characteristics by department.