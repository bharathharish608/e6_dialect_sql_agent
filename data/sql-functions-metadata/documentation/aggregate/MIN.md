# MIN

## Description

Returns the minimum value from a group of values. The MIN function compares values and returns the smallest one according to the data type's natural ordering. For numeric types, it returns the smallest number; for strings, it returns the lexicographically first value; for dates/timestamps, it returns the earliest date/time.

## Syntax

```sql
MIN(expression)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| expression | ANY COMPARABLE | The column or expression to find the minimum value from. Can be any comparable data type including numeric, string, date, or timestamp. |

## Returns

The function returns a value of the same data type as the input expression. If all values are NULL, the function returns NULL.

## Examples

### Example 1: Find Minimum Price
Find the lowest price from the products table.

```sql
SELECT MIN(price) AS lowest_price
FROM products;
```

**Result:**
| lowest_price |
|--------------|
| 9.99         |

### Example 2: Find Earliest Order Date
Get the earliest order date from the orders table.

```sql
SELECT MIN(order_date) AS first_order_date
FROM orders;
```

**Result:**
| first_order_date |
|------------------|
| 2023-01-15       |

### Example 3: Find Minimum String Value
Find the alphabetically first product name.

```sql
SELECT MIN(product_name) AS first_product_alphabetically
FROM products;
```

**Result:**
| first_product_alphabetically |
|------------------------------|
| Apples                       |

### Example 4: Minimum Value by Category
Find the minimum price for each product category.

```sql
SELECT 
    category,
    MIN(price) AS min_price_in_category
FROM products
GROUP BY category
ORDER BY category;
```

**Result:**
| category    | min_price_in_category |
|-------------|----------------------|
| Electronics | 49.99                |
| Food        | 9.99                 |
| Toys        | 14.99                |

### Example 5: Minimum with Multiple Columns
Find customers with their earliest transaction date and minimum transaction amount.

```sql
SELECT 
    customer_id,
    MIN(transaction_date) AS first_transaction,
    MIN(amount) AS smallest_transaction
FROM transactions
GROUP BY customer_id
HAVING MIN(amount) < 50
ORDER BY smallest_transaction;
```

**Result:**
| customer_id | first_transaction | smallest_transaction |
|-------------|------------------|---------------------|
| C003        | 2023-02-01       | 15.50              |
| C001        | 2023-01-15       | 25.00              |
| C005        | 2023-03-10       | 35.75              |

## Common Use Cases

1. **Price Analysis**: Finding the lowest price point for products or services
2. **Date Range Analysis**: Identifying the start of a time period or earliest event
3. **Inventory Management**: Determining minimum stock levels
4. **Performance Metrics**: Finding minimum response times or lowest scores
5. **Data Quality**: Identifying boundary values in datasets

## Notes

- MIN ignores NULL values unless all values are NULL
- For string comparisons, the function uses lexicographic (dictionary) ordering
- When used with GROUP BY, MIN returns the minimum value for each group
- MIN can be combined with other aggregate functions in the same query
- The function works with any comparable data type including custom types that implement comparison operators