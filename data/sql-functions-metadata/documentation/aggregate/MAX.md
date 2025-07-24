# MAX

## Description

Returns the maximum value from a group of values. The MAX function compares values and returns the largest one according to the data type's natural ordering. For numeric types, it returns the largest number; for strings, it returns the lexicographically last value; for dates/timestamps, it returns the latest date/time.

## Syntax

```sql
MAX(expression)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| expression | ANY COMPARABLE | The column or expression to find the maximum value from. Can be any comparable data type including numeric, string, date, or timestamp. |

## Returns

The function returns a value of the same data type as the input expression. If all values are NULL, the function returns NULL.

## Examples

### Example 1: Find Maximum Score
Find the highest score from the students table.

```sql
SELECT MAX(test_score) AS highest_score
FROM students;
```

**Result:**
| highest_score |
|---------------|
| 98.5          |

### Example 2: Find Latest Timestamp
Get the most recent update timestamp from the products table.

```sql
SELECT MAX(last_updated) AS latest_update
FROM products;
```

**Result:**
| latest_update           |
|-------------------------|
| 2024-03-15 14:30:00    |

### Example 3: Find Maximum Alphanumeric Value
Find the alphabetically last customer name.

```sql
SELECT MAX(customer_name) AS last_customer_alphabetically
FROM customers;
```

**Result:**
| last_customer_alphabetically |
|------------------------------|
| Zimmerman, Robert           |

### Example 4: Maximum Value by Department
Find the highest salary in each department.

```sql
SELECT 
    department,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department
ORDER BY max_salary DESC;
```

**Result:**
| department  | max_salary |
|-------------|------------|
| Engineering | 150000     |
| Sales       | 120000     |
| Marketing   | 95000      |
| Support     | 75000      |

### Example 5: Maximum with Date Filtering
Find the maximum order amount for each customer in the last quarter, including only customers with maximum orders over $1000.

```sql
SELECT 
    customer_id,
    customer_name,
    MAX(order_amount) AS largest_order,
    MAX(order_date) AS most_recent_order
FROM orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
GROUP BY customer_id, customer_name
HAVING MAX(order_amount) > 1000
ORDER BY largest_order DESC;
```

**Result:**
| customer_id | customer_name | largest_order | most_recent_order |
|-------------|---------------|---------------|-------------------|
| C101        | Tech Corp     | 5500.00       | 2024-03-14       |
| C205        | Global Inc    | 3200.50       | 2024-03-10       |
| C089        | Smart Solutions| 2100.00      | 2024-02-28       |
| C150        | Data Systems  | 1500.75       | 2024-03-12       |

## Common Use Cases

1. **Performance Analysis**: Finding peak values, highest scores, or best performance metrics
2. **Financial Analysis**: Identifying maximum revenue, highest prices, or peak transaction amounts
3. **Time Series Analysis**: Finding the most recent events or latest updates
4. **Inventory Management**: Determining maximum stock levels or highest demand periods
5. **Data Validation**: Identifying upper boundary values for data quality checks

## Notes

- MAX ignores NULL values unless all values are NULL
- For string comparisons, the function uses lexicographic (dictionary) ordering
- When used with GROUP BY, MAX returns the maximum value for each group
- MAX can be combined with other aggregate functions in the same query
- The function works with any comparable data type including custom types that implement comparison operators
- For timestamp comparisons, MAX returns the most recent (latest) timestamp