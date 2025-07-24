# ARRAY_AGG

## Description

`ARRAY_AGG` is an aggregate function that collects values from multiple rows into an array. It creates an array containing all non-NULL values from the specified column or expression across the rows in a group.

## Syntax

```sql
ARRAY_AGG([DISTINCT] expression [ORDER BY sort_expression [ASC | DESC] [, ...]])
```

### Parameters

- `expression`: The column or expression to aggregate into an array (ANY type)
- `DISTINCT` (optional): Removes duplicate values from the result array
- `ORDER BY` (optional): Sorts the values within the array

### Return Type

`ARRAY` - Returns an array containing all aggregated values

### Aliases

- `COLLECT_LIST` - Functionally equivalent to ARRAY_AGG

## Examples

### 1. Basic Array Aggregation

Collect all product names into a single array:

```sql
SELECT ARRAY_AGG(product_name) AS all_products
FROM products;
```

**Result:**
```
all_products
--------------------------------
['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
```

### 2. Array Aggregation with DISTINCT Values

Remove duplicate categories when creating an array:

```sql
SELECT ARRAY_AGG(DISTINCT category) AS unique_categories
FROM products;
```

**Result:**
```
unique_categories
--------------------------------
['Electronics', 'Accessories', 'Office Supplies']
```

### 3. Array Aggregation with ORDER BY

Create an array of employee names sorted alphabetically:

```sql
SELECT department,
       ARRAY_AGG(employee_name ORDER BY employee_name ASC) AS sorted_employees
FROM employees
GROUP BY department;
```

**Result:**
```
department     | sorted_employees
---------------|----------------------------------
Sales          | ['Alice Brown', 'Bob Smith', 'Carol White']
Engineering    | ['David Lee', 'Eve Davis', 'Frank Wilson']
Marketing      | ['Grace Taylor', 'Henry Jones']
```

### 4. Grouped Arrays

Create arrays of products grouped by category:

```sql
SELECT category,
       ARRAY_AGG(product_name) AS products,
       ARRAY_AGG(price) AS prices,
       COUNT(*) AS product_count
FROM products
GROUP BY category;
```

**Result:**
```
category        | products                          | prices              | product_count
----------------|-----------------------------------|---------------------|---------------
Electronics     | ['Laptop', 'Monitor', 'TV']       | [999.99, 299.99, 799.99] | 3
Accessories     | ['Mouse', 'Keyboard', 'Cable']    | [29.99, 79.99, 9.99]     | 3
Office Supplies | ['Desk', 'Chair', 'Lamp']         | [199.99, 149.99, 39.99]  | 3
```

### 5. Arrays of Complex Types

Aggregate structured data into arrays:

```sql
-- Create an array of order details with multiple fields
SELECT customer_id,
       ARRAY_AGG(
           STRUCT(
               order_id,
               order_date,
               total_amount
           ) ORDER BY order_date DESC
       ) AS order_history
FROM orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY customer_id;
```

**Result:**
```
customer_id | order_history
------------|------------------------------------------------------------
1001        | [{order_id: 5432, order_date: '2024-01-15', total_amount: 150.00},
            |  {order_id: 5401, order_date: '2024-01-10', total_amount: 89.99}]
1002        | [{order_id: 5425, order_date: '2024-01-12', total_amount: 299.99}]
```

## Using COLLECT_LIST Alias

The `COLLECT_LIST` function is an alias for `ARRAY_AGG` and works identically:

```sql
SELECT department,
       COLLECT_LIST(employee_id) AS employee_ids
FROM employees
GROUP BY department;
```

## Common Use Cases

1. **Data Denormalization**: Combine related rows into a single row with array columns
2. **Report Generation**: Create summary reports with lists of items
3. **JSON/API Responses**: Prepare nested data structures for APIs
4. **Analytics**: Analyze patterns in grouped data
5. **Data Migration**: Transform normalized data into denormalized format

## Performance Considerations

- Arrays can become large with many rows, impacting memory usage
- Use `DISTINCT` when duplicate values are not needed to reduce array size
- Consider using `LIMIT` in subqueries when aggregating large datasets
- Ordering within arrays adds computational overhead

## Compatibility Notes

- Standard SQL aggregate function supported by most modern databases
- Some databases may have slightly different syntax for ORDER BY within aggregates
- Array handling and syntax may vary between database systems