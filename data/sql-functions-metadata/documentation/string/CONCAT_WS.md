# CONCAT_WS

Concatenates strings with a separator.

## Syntax

```sql
CONCAT_WS( <separator>, <string1>, <string2> [ , <stringN> ... ] )
```

## Arguments

### separator
- **Type**: STRING
- **Required**: Yes
- **Description**: The separator string to use between concatenated values. This separator is placed between each pair of concatenated strings but not at the beginning or end of the result.

### string1
- **Type**: STRING
- **Required**: Yes
- **Description**: The first string value to concatenate after the separator parameter.

### string2
- **Type**: STRING
- **Required**: Yes
- **Description**: The second string value to concatenate.

### stringN
- **Type**: STRING
- **Required**: No
- **Description**: Additional string values to concatenate. You can provide any number of additional string parameters.

## Returns

- **Type**: Same as first argument, nullable
- **Description**: Returns a single string that is the result of concatenating all input strings with the separator placed between each pair of strings.
- **NULL Handling**: If the separator is NULL, the result is NULL. NULL values in the string arguments are skipped.

## Usage Notes

- CONCAT_WS stands for "Concatenate With Separator"
- The separator is only placed between strings, not at the beginning or end
- NULL values in the string arguments are skipped, not treated as empty strings
- The function requires at least the separator and one string argument
- Useful for creating delimited lists or formatted strings

## Examples

### Example 1: Basic String Concatenation with Comma Separator

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
       CONCAT_WS(', ', name, department, email) AS employee_info
FROM employees
ORDER BY id;
```

Result:
```
+----+------------------+---------------------------------------------------------------+
| id | name             | employee_info                                                 |
+----+------------------+---------------------------------------------------------------+
| 1  | John Doe         | John Doe, Sales, john.doe@company.com                        |
| 2  | Jane Smith       | Jane Smith, Marketing, jane.s@company.com                    |
| 3  | Bob Johnson      | Bob Johnson, IT, bob.j@company.com                           |
| 4  | Alice Brown      | Alice Brown, HR, alice.b@company.com                         |
| 5  | Charlie Wilson   | Charlie Wilson, Sales, charlie.w@company.com                 |
+----+------------------+---------------------------------------------------------------+
```

This example demonstrates using CONCAT_WS with a comma and space separator to create a formatted employee information string.

### Example 2: Handling NULL Values

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@dundermif.com| 555-0123     | Scranton |
| 2           | Dwight     | NULL        | d.schrute@farms.com  | 555-0124     | Scranton |
| 3           | Jim        | Halpert     | NULL                 | 555-0125     | Scranton |
| 4           | NULL       | Beesly      | p.beesly@reception.com| NULL        | Scranton |
| 5           | Stanley    | Hudson      | s.hudson@sales.com   | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query:
```sql
SELECT customer_id,
       first_name,
       last_name,
       CONCAT_WS(' ', first_name, last_name) AS full_name,
       CONCAT_WS(' | ', first_name, last_name, email, phone) AS contact_info
FROM customers
ORDER BY customer_id;
```

Result:
```
+-------------+------------+-------------+----------------+---------------------------------------------------+
| customer_id | first_name | last_name   | full_name      | contact_info                                      |
+-------------+------------+-------------+----------------+---------------------------------------------------+
| 1           | Michael    | Scott       | Michael Scott  | Michael | Scott | m.scott@dundermif.com | 555-0123|
| 2           | Dwight     | NULL        | Dwight         | Dwight | d.schrute@farms.com | 555-0124         |
| 3           | Jim        | Halpert     | Jim Halpert    | Jim | Halpert | 555-0125                         |
| 4           | NULL       | Beesly      | Beesly         | Beesly | p.beesly@reception.com                   |
| 5           | Stanley    | Hudson      | Stanley Hudson | Stanley | Hudson | s.hudson@sales.com | 555-0127|
+-------------+------------+-------------+----------------+---------------------------------------------------+
```

This example shows how CONCAT_WS skips NULL values instead of treating them as empty strings, unlike CONCAT which returns NULL if any argument is NULL.

### Example 3: Creating CSV-Style Output

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | Electronics| 29.99  | 150            |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | 599.99 | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT CONCAT_WS(',', 
              product_id, 
              product_name, 
              category, 
              price, 
              stock_quantity) AS csv_row
FROM products
ORDER BY product_id;
```

Result:
```
+---------------------------------------------------+
| csv_row                                           |
+---------------------------------------------------+
| 1,Laptop Pro,Electronics,1299.99,50               |
| 2,Wireless Mouse,Electronics,29.99,150            |
| 3,Office Chair,Furniture,249.99,30                |
| 4,Standing Desk,Furniture,599.99,20               |
| 5,USB-C Cable,Electronics,19.99,200               |
+---------------------------------------------------+
```

This example demonstrates using CONCAT_WS to create CSV-formatted output with comma separators.

### Example 4: Building File Paths

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
       CONCAT_WS('/', 'orders', YEAR(order_date), MONTH(order_date), order_id) AS file_path,
       CONCAT_WS('-', 'ORD', YEAR(order_date), LPAD(order_id, 6, '0')) AS order_reference
FROM orders
ORDER BY order_id;
```

Result:
```
+----------+------------------------+------------------+
| order_id | file_path              | order_reference  |
+----------+------------------------+------------------+
| 1001     | orders/2025/1/1001     | ORD-2025-001001  |
| 1002     | orders/2025/1/1002     | ORD-2025-001002  |
| 1003     | orders/2025/1/1003     | ORD-2025-001003  |
| 1004     | orders/2025/1/1004     | ORD-2025-001004  |
| 1005     | orders/2025/1/1005     | ORD-2025-001005  |
+----------+------------------------+------------------+
```

This example shows using CONCAT_WS with different separators to build file paths and reference codes.

### Example 5: Creating Formatted Addresses

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+------------+
| customer_id | first_name | last_name   | email                | phone        | city       |
+-------------+------------+-------------+----------------------+--------------+------------+
| 1           | Michael    | Scott       | m.scott@company.com  | 555-0123     | Scranton   |
| 2           | Dwight     | Schrute     | d.schrute@company.com| 555-0124     | NULL       |
| 3           | Jim        | Halpert     | j.halpert@company.com| 555-0125     | Stamford   |
| 4           | Pam        | Beesly      | p.beesly@company.com | 555-0126     | Scranton   |
| 5           | Stanley    | Hudson      | s.hudson@company.com | 555-0127     | Scranton   |
+-------------+------------+-------------+----------------------+--------------+------------+
```

Query:
```sql
SELECT customer_id,
       CONCAT_WS(' ', first_name, last_name) AS name,
       CONCAT_WS(', ', 
                 CONCAT_WS(' ', first_name, last_name),
                 city,
                 'PA') AS mailing_label,
       CONCAT_WS(' - ', city, phone) AS location_contact
FROM customers
ORDER BY customer_id;
```

Result:
```
+-------------+------------------+----------------------------+---------------------+
| customer_id | name             | mailing_label              | location_contact    |
+-------------+------------------+----------------------------+---------------------+
| 1           | Michael Scott    | Michael Scott, Scranton, PA| Scranton - 555-0123 |
| 2           | Dwight Schrute   | Dwight Schrute, PA         | 555-0124            |
| 3           | Jim Halpert      | Jim Halpert, Stamford, PA  | Stamford - 555-0125 |
| 4           | Pam Beesly       | Pam Beesly, Scranton, PA   | Scranton - 555-0126 |
| 5           | Stanley Hudson   | Stanley Hudson, Scranton, PA| Scranton - 555-0127 |
+-------------+------------------+----------------------------+---------------------+
```

This example demonstrates nested CONCAT_WS calls and shows how NULL values are handled when creating formatted address labels.