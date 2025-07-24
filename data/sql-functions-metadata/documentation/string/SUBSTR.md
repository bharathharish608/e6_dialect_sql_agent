# SUBSTR

Extracts a substring from a string.

## Syntax

```sql
SUBSTR( <string>, <start_position> [ , <length> ] )
```

## Arguments

### string
- **Type**: STRING
- **Required**: Yes
- **Description**: The source string from which to extract a substring.

### start_position
- **Type**: INTEGER
- **Required**: Yes
- **Description**: The starting position for extraction (1-based). If positive, counting starts from the beginning of the string. If negative, counting starts from the end of the string.

### length
- **Type**: INTEGER
- **Required**: No
- **Description**: The number of characters to extract. If omitted, extracts from the start position to the end of the string.

## Returns

- **Type**: Same as input, nullable
- **Description**: Returns the extracted substring from the source string.
- **NULL Handling**: Returns NULL if the input string is NULL.

## Usage Notes

- SUBSTR uses 1-based indexing (first character is at position 1)
- Negative start positions count from the end of the string
- If length is omitted, returns all characters from start position to end
- If start position is beyond string length, returns empty string
- SUBSTR is equivalent to SUBSTRING function

## Examples

### Example 1: Basic Substring Extraction

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
       SUBSTR(name, 1, 4) AS first_four,
       SUBSTR(name, 6) AS from_sixth,
       SUBSTR(email, 1, POSITION('@' IN email) - 1) AS username
FROM employees
ORDER BY id;
```

Result:
```
+----+------------------+------------+------------+------------+
| id | name             | first_four | from_sixth | username   |
+----+------------------+------------+------------+------------+
| 1  | John Doe         | John       | Doe        | john.doe   |
| 2  | Jane Smith       | Jane       | Smith      | jane.s     |
| 3  | Bob Johnson      | Bob        | Johnson    | bob.j      |
| 4  | Alice Brown      | Alic       | e Brown    | alice.b    |
| 5  | Charlie Wilson   | Char       | lie Wilson | charlie.w  |
+----+------------------+------------+------------+------------+
```

This example shows extracting fixed-length substrings and using SUBSTR to extract email usernames.

### Example 2: Handling NULL Values

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | NULL             | Electronics| 29.99  | 150            |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | NULL       | 599.99 | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT product_id,
       product_name,
       SUBSTR(product_name, 1, 6) AS short_name,
       category,
       SUBSTR(category, 1, 3) AS cat_code
FROM products
ORDER BY product_id;
```

Result:
```
+------------+------------------+------------+-------------+----------+
| product_id | product_name     | short_name | category    | cat_code |
+------------+------------------+------------+-------------+----------+
| 1          | Laptop Pro       | Laptop     | Electronics | Ele      |
| 2          | NULL             | NULL       | Electronics | Ele      |
| 3          | Office Chair     | Office     | Furniture   | Fur      |
| 4          | Standing Desk    | Standi     | NULL        | NULL     |
| 5          | USB-C Cable      | USB-C      | Electronics | Ele      |
+------------+------------------+------------+-------------+----------+
```

This example demonstrates that SUBSTR returns NULL when the input string is NULL.

### Example 3: Using Negative Start Positions

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
       phone,
       SUBSTR(phone, -4) AS last_four,
       SUBSTR(phone, -4, 2) AS third_fourth_from_end,
       email,
       SUBSTR(email, -4) AS domain_ext
FROM customers
ORDER BY customer_id;
```

Result:
```
+-------------+----------+-----------+-----------------------+------------------------+------------+
| customer_id | phone    | last_four | third_fourth_from_end | email                  | domain_ext |
+-------------+----------+-----------+-----------------------+------------------------+------------+
| 1           | 555-0123 | 0123      | 01                    | m.scott@dundermif.com  | .com       |
| 2           | 555-0124 | 0124      | 01                    | d.schrute@farms.com    | .com       |
| 3           | 555-0125 | 0125      | 01                    | j.halpert@sales.com    | .com       |
| 4           | 555-0126 | 0126      | 01                    | p.beesly@reception.com | .com       |
| 5           | 555-0127 | 0127      | 01                    | s.hudson@sales.com     | .com       |
+-------------+----------+-----------+-----------------------+------------------------+------------+
```

This example shows using negative start positions to extract from the end of strings.

### Example 4: Extracting Date Components

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
       order_date,
       SUBSTR(CAST(order_date AS VARCHAR), 1, 4) AS year,
       SUBSTR(CAST(order_date AS VARCHAR), 6, 2) AS month,
       SUBSTR(CAST(order_date AS VARCHAR), 9, 2) AS day,
       SUBSTR(status, 1, 3) AS status_code
FROM orders
ORDER BY order_id;
```

Result:
```
+----------+------------+------+-------+-----+-------------+
| order_id | order_date | year | month | day | status_code |
+----------+------------+------+-------+-----+-------------+
| 1001     | 2025-01-05 | 2025 | 01    | 05  | Shi         |
| 1002     | 2025-01-06 | 2025 | 01    | 06  | Pen         |
| 1003     | 2025-01-07 | 2025 | 01    | 07  | Del         |
| 1004     | 2025-01-08 | 2025 | 01    | 08  | Pro         |
| 1005     | 2025-01-09 | 2025 | 01    | 09  | Shi         |
+----------+------------+------+-------+-----+-------------+
```

This example demonstrates extracting date components and creating status codes using SUBSTR.

### Example 5: Variable Length Extraction

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
       LENGTH(name) AS name_length,
       SUBSTR(name, 1, LENGTH(name)/2) AS first_half,
       SUBSTR(name, LENGTH(name)/2 + 1) AS second_half,
       SUBSTR(name, POSITION(' ' IN name) + 1) AS last_name
FROM employees
WHERE id <= 5
ORDER BY id;
```

Result:
```
+----+------------------+-------------+------------+-------------+-----------+
| id | name             | name_length | first_half | second_half | last_name |
+----+------------------+-------------+------------+-------------+-----------+
| 1  | John Doe         | 8           | John       |  Doe        | Doe       |
| 2  | Jane Smith       | 10          | Jane       | Smith       | Smith     |
| 3  | Bob Johnson      | 11          | Bob J      | ohnson      | Johnson   |
| 4  | Alice Brown      | 11          | Alice      |  Brown      | Brown     |
| 5  | Charlie Wilson   | 14          | Charlie    |  Wilson     | Wilson    |
+----+------------------+-------------+------------+-------------+-----------+
```

This example shows using SUBSTR with calculated positions to extract variable-length substrings based on string properties.