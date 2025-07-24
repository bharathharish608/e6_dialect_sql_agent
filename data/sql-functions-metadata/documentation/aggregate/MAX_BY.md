# MAX_BY

## Description

The `MAX_BY` function returns the value(s) from one column corresponding to the maximum value(s) in another column within each group. When used with a third parameter, it returns the top N values based on the comparison column.

## Syntax

```sql
MAX_BY(value_column, comparison_column [, n])
```

## Parameters

- `value_column`: The column whose values will be returned (ANY type)
- `comparison_column`: The column used for comparison to find maximum values (must be COMPARABLE)
- `n` (optional): The number of top values to return (INTEGER). If omitted, returns single value.

## Return Type

- Without `n`: Returns the same type as `value_column`
- With `n`: Returns an ARRAY of the same type as `value_column`

## Examples

### Example 1: Find the product with highest price in each category

```sql
SELECT 
    category,
    MAX_BY(product_name, price) AS most_expensive_product,
    MAX(price) AS highest_price
FROM products
GROUP BY category
ORDER BY category;
```

**Result:**
```
category    | most_expensive_product | highest_price
------------|------------------------|---------------
Electronics | Laptop Pro             | 1299.99
Furniture   | Executive Desk         | 899.99
Clothing    | Winter Coat            | 199.99
```

### Example 2: Get top 3 highest-paid employees per department

```sql
SELECT 
    department,
    MAX_BY(employee_name, salary, 3) AS top_3_earners,
    MAX_BY(salary, salary, 3) AS top_3_salaries
FROM employees
GROUP BY department
ORDER BY department;
```

**Result:**
```
department | top_3_earners                           | top_3_salaries
-----------|----------------------------------------|----------------
Sales      | ['John Smith', 'Jane Doe', 'Bob Wilson'] | [95000, 92000, 88000]
IT         | ['Alice Chen', 'Tom Davis', 'Sara Lee']  | [120000, 115000, 110000]
HR         | ['Mike Johnson', 'Lisa Brown']            | [75000, 72000]
```

### Example 3: Find the order with maximum amount for each customer

```sql
SELECT 
    customer_id,
    MAX_BY(order_id, order_total) AS largest_order_id,
    MAX_BY(order_date, order_total) AS largest_order_date,
    MAX(order_total) AS max_order_amount
FROM orders
WHERE order_status = 'completed'
GROUP BY customer_id
HAVING MAX(order_total) > 1000
ORDER BY max_order_amount DESC
LIMIT 10;
```

**Result:**
```
customer_id | largest_order_id | largest_order_date | max_order_amount
------------|------------------|-------------------|------------------
C1234       | ORD-98765       | 2024-01-15        | 5499.99
C5678       | ORD-44332       | 2024-02-20        | 4250.00
C9101       | ORD-77889       | 2024-01-28        | 3899.50
```

### Example 4: Get products with top 2 ratings per category

```sql
WITH product_ratings AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.category,
        AVG(r.rating) AS avg_rating,
        COUNT(r.rating) AS review_count
    FROM products p
    JOIN reviews r ON p.product_id = r.product_id
    GROUP BY p.product_id, p.product_name, p.category
)
SELECT 
    category,
    MAX_BY(product_name, avg_rating, 2) AS top_rated_products,
    MAX_BY(avg_rating, avg_rating, 2) AS top_ratings,
    MAX_BY(review_count, avg_rating, 2) AS review_counts
FROM product_ratings
WHERE review_count >= 10
GROUP BY category
ORDER BY category;
```

**Result:**
```
category    | top_rated_products              | top_ratings  | review_counts
------------|--------------------------------|--------------|---------------
Electronics | ['Smartphone X', 'Tablet Pro']  | [4.8, 4.7]   | [523, 287]
Books       | ['Novel A', 'Guide B']          | [4.9, 4.6]   | [89, 156]
Home        | ['Air Purifier', 'Smart Light'] | [4.5, 4.4]   | [234, 178]
```

### Example 5: Find the latest transaction per account type

```sql
SELECT 
    account_type,
    MAX_BY(transaction_id, transaction_date) AS latest_transaction_id,
    MAX_BY(amount, transaction_date) AS latest_amount,
    MAX_BY(description, transaction_date) AS latest_description,
    MAX(transaction_date) AS latest_date
FROM transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY account_type
ORDER BY latest_date DESC;
```

**Result:**
```
account_type | latest_transaction_id | latest_amount | latest_description    | latest_date
-------------|----------------------|---------------|----------------------|-------------
Checking     | TXN-2024-98765      | -125.50       | Grocery Store        | 2024-03-15
Savings      | TXN-2024-87654      | 500.00        | Deposit              | 2024-03-14
Credit       | TXN-2024-76543      | -89.99        | Online Purchase      | 2024-03-14
Investment   | TXN-2024-65432      | 2500.00       | Stock Purchase       | 2024-03-12
```

## Notes

- When multiple values have the same maximum comparison value, `MAX_BY` returns one of them arbitrarily
- If all values in the comparison column are NULL, the function returns NULL
- When using the `n` parameter, if fewer than `n` non-null values exist, the array will contain all available values
- The comparison column must support ordering operations (numeric, date/time, or string types)
- This function is particularly useful for finding associated values of maximum records without using subqueries

## See Also

- [`MIN_BY`](MIN_BY.md) - Returns values corresponding to minimum values
- [`MAX`](MAX.md) - Returns the maximum value
- [`FIRST_VALUE`](../window/FIRST_VALUE.md) - Window function alternative for ordered data