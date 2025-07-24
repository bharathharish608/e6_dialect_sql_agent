# MIN_BY

## Description

The `MIN_BY` function returns the value(s) from one column corresponding to the minimum value(s) in another column within each group. When used with a third parameter, it returns the bottom N values based on the comparison column.

## Syntax

```sql
MIN_BY(value_column, comparison_column [, n])
```

## Parameters

- `value_column`: The column whose values will be returned (ANY type)
- `comparison_column`: The column used for comparison to find minimum values (must be COMPARABLE)
- `n` (optional): The number of bottom values to return (INTEGER). If omitted, returns single value.

## Return Type

- Without `n`: Returns the same type as `value_column`
- With `n`: Returns an ARRAY of the same type as `value_column`

## Examples

### Example 1: Find the cheapest product in each category

```sql
SELECT 
    category,
    MIN_BY(product_name, price) AS cheapest_product,
    MIN(price) AS lowest_price
FROM products
WHERE in_stock = true
GROUP BY category
ORDER BY category;
```

**Result:**
```
category    | cheapest_product    | lowest_price
------------|-------------------|-------------
Electronics | USB Cable         | 9.99
Furniture   | Desk Lamp         | 29.99
Clothing    | Basic T-Shirt     | 12.99
Books       | Paperback Novel   | 7.99
```

### Example 2: Get 3 lowest-performing sales reps per region

```sql
SELECT 
    region,
    MIN_BY(sales_rep_name, total_sales, 3) AS bottom_3_reps,
    MIN_BY(total_sales, total_sales, 3) AS bottom_3_sales
FROM sales_performance
WHERE quarter = 'Q1-2024'
GROUP BY region
ORDER BY region;
```

**Result:**
```
region     | bottom_3_reps                               | bottom_3_sales
-----------|--------------------------------------------|-----------------
East       | ['Tom Wilson', 'Sarah Jones', 'Mike Lee']  | [45000, 52000, 58000]
West       | ['Lisa Chen', 'Bob Smith', 'Amy Davis']    | [38000, 41000, 47000]
Central    | ['John Brown', 'Kate Miller']              | [55000, 61000]
```

### Example 3: Find earliest order for each product

```sql
SELECT 
    product_id,
    MIN_BY(order_id, order_date) AS first_order_id,
    MIN_BY(customer_id, order_date) AS first_customer,
    MIN_BY(quantity, order_date) AS first_order_quantity,
    MIN(order_date) AS first_order_date
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
GROUP BY product_id
HAVING MIN(order_date) >= '2024-01-01'
ORDER BY first_order_date
LIMIT 10;
```

**Result:**
```
product_id | first_order_id | first_customer | first_order_quantity | first_order_date
-----------|----------------|----------------|---------------------|------------------
P-1001     | ORD-10001     | C-5543         | 2                   | 2024-01-01
P-2055     | ORD-10003     | C-7891         | 1                   | 2024-01-01
P-3102     | ORD-10007     | C-2234         | 5                   | 2024-01-02
```

### Example 4: Get students with 3 lowest scores per class

```sql
WITH student_averages AS (
    SELECT 
        s.student_id,
        s.student_name,
        s.class_id,
        c.class_name,
        AVG(g.score) AS avg_score
    FROM students s
    JOIN grades g ON s.student_id = g.student_id
    JOIN classes c ON s.class_id = c.class_id
    WHERE g.semester = 'Fall 2024'
    GROUP BY s.student_id, s.student_name, s.class_id, c.class_name
)
SELECT 
    class_name,
    MIN_BY(student_name, avg_score, 3) AS students_needing_help,
    MIN_BY(ROUND(avg_score, 1), avg_score, 3) AS their_averages
FROM student_averages
GROUP BY class_name
ORDER BY class_name;
```

**Result:**
```
class_name        | students_needing_help                            | their_averages
------------------|------------------------------------------------|----------------
Mathematics 101   | ['David Park', 'Emma Wilson', 'Ryan Garcia']   | [65.2, 68.5, 71.0]
Physics 201       | ['Sophie Lee', 'James Chen', 'Maria Lopez']    | [58.8, 62.3, 64.7]
Chemistry 301     | ['Alex Johnson', 'Nina Patel']                 | [70.5, 72.1]
```

### Example 5: Find accounts with lowest balance per account type

```sql
SELECT 
    account_type,
    MIN_BY(account_number, balance) AS lowest_balance_account,
    MIN_BY(customer_name, balance) AS customer_with_lowest,
    MIN_BY(last_activity_date, balance) AS last_activity,
    MIN(balance) AS minimum_balance
FROM accounts a
JOIN customers c ON a.customer_id = c.customer_id
WHERE account_status = 'active'
    AND balance > 0
GROUP BY account_type
ORDER BY account_type;
```

**Result:**
```
account_type | lowest_balance_account | customer_with_lowest | last_activity      | minimum_balance
-------------|----------------------|---------------------|-------------------|----------------
Checking     | CHK-789012          | Robert Brown        | 2024-03-10        | 25.50
Savings      | SAV-345678          | Jennifer Davis      | 2024-02-28        | 100.00
Money Market | MM-567890           | Michael Wilson      | 2024-03-05        | 500.25
CD           | CD-123456           | Susan Martinez      | 2024-01-15        | 1000.00
```

## Notes

- When multiple values have the same minimum comparison value, `MIN_BY` returns one of them arbitrarily
- If all values in the comparison column are NULL, the function returns NULL
- When using the `n` parameter, if fewer than `n` non-null values exist, the array will contain all available values
- The comparison column must support ordering operations (numeric, date/time, or string types)
- This function is useful for finding associated values of minimum records without using subqueries
- NULL values in the comparison column are ignored when determining the minimum

## See Also

- [`MAX_BY`](MAX_BY.md) - Returns values corresponding to maximum values
- [`MIN`](MIN.md) - Returns the minimum value
- [`LAST_VALUE`](../window/LAST_VALUE.md) - Window function alternative for ordered data