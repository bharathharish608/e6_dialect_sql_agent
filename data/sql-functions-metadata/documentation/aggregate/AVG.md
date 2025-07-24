# AVG

Calculates the average of numeric values.

## Syntax

```sql
AVG( <numeric_expression> )
```

## Arguments

### numeric_expression
- **Type**: NUMERIC
- **Required**: Yes
- **Description**: Numeric column to average. Can be any numeric data type.

## Returns

- **Type**: DOUBLE
- **Description**: Returns the average of all non-NULL numeric values as a DOUBLE.
- **NULL Handling**: NULL values are ignored in the average calculation. If all values are NULL, returns NULL.

## Usage Notes

- AVG ignores NULL values in calculations
- Returns NULL if all values are NULL or if there are no rows
- Result is always DOUBLE regardless of input type
- Commonly used with GROUP BY for group averages
- Can be used with expressions, not just column references

## Examples

### Example 1: Basic Average Calculation

Calculate the average salary of all employees.

**Sample Data (employees):**
```
| id | name          | department  | salary | hire_date  | email                |
|----|---------------|-------------|--------|------------|----------------------|
| 1  | John Smith    | Sales       | 50000  | 2020-01-15 | john@company.com     |
| 2  | Jane Doe      | Engineering | 75000  | 2019-03-22 | jane@company.com     |
| 3  | Bob Johnson   | Sales       | 55000  | 2021-06-10 | bob@company.com      |
| 4  | Alice Brown   | Engineering | 80000  | 2018-11-30 | alice@company.com    |
| 5  | Charlie Davis | Marketing   | 60000  | 2020-09-14 | charlie@company.com  |
```

**Query:**
```sql
SELECT AVG(salary) AS average_salary
FROM employees;
```

**Result:**
```
| average_salary |
|----------------|
| 64000.0        |
```

**Explanation:** The query calculates the average of all salaries (50000 + 75000 + 55000 + 80000 + 60000) / 5 = 64000.0.

### Example 2: NULL Value Handling

Demonstrate how AVG handles NULL values in the calculation.

**Sample Data (products):**
```
| product_id | product_name | category    | price  | stock_quantity |
|------------|--------------|-------------|--------|----------------|
| 1          | Laptop       | Electronics | 999.99 | 15             |
| 2          | Mouse        | Electronics | 29.99  | 50             |
| 3          | Desk Chair   | Furniture   | NULL   | 10             |
| 4          | Monitor      | Electronics | 299.99 | 25             |
| 5          | Keyboard     | Electronics | 79.99  | NULL           |
| 6          | Desk Lamp    | Furniture   | NULL   | 20             |
```

**Query:**
```sql
SELECT 
    AVG(price) AS average_price,
    COUNT(*) AS total_products,
    COUNT(price) AS products_with_price
FROM products;
```

**Result:**
```
| average_price | total_products | products_with_price |
|---------------|----------------|---------------------|
| 352.49        | 6              | 4                   |
```

**Explanation:** AVG ignores the two NULL price values and calculates (999.99 + 29.99 + 299.99 + 79.99) / 4 = 352.49.

### Example 3: Grouped Averages

Calculate average quantities ordered per product status.

**Sample Data (orders):**
```
| order_id | customer_id | product_id | quantity | order_date | status    |
|----------|-------------|------------|----------|------------|-----------|
| 1        | 101         | 1          | 2        | 2023-01-15 | completed |
| 2        | 102         | 2          | 5        | 2023-01-16 | completed |
| 3        | 103         | 1          | 1        | 2023-01-17 | pending   |
| 4        | 104         | 3          | 3        | 2023-01-18 | completed |
| 5        | 105         | 2          | 4        | 2023-01-19 | cancelled |
| 6        | 101         | 4          | 2        | 2023-01-20 | completed |
| 7        | 102         | 1          | 3        | 2023-01-21 | pending   |
| 8        | 103         | 5          | 6        | 2023-01-22 | completed |
```

**Query:**
```sql
SELECT 
    status,
    AVG(quantity) AS avg_quantity,
    COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY avg_quantity DESC;
```

**Result:**
```
| status    | avg_quantity | order_count |
|-----------|--------------|-------------|
| cancelled | 4.0          | 1           |
| completed | 3.6          | 5           |
| pending   | 2.0          | 2           |
```

**Explanation:** The query groups orders by status and calculates the average quantity for each status group.

### Example 4: AVG with Expressions

Calculate the average order value using expressions.

**Sample Data (orders with products):**
```
-- Using the same orders table from Example 3
-- Products table:
| product_id | product_name | category    | price  | stock_quantity |
|------------|--------------|-------------|--------|----------------|
| 1          | Laptop       | Electronics | 999.99 | 15             |
| 2          | Mouse        | Electronics | 29.99  | 50             |
| 3          | Desk Chair   | Furniture   | 249.99 | 10             |
| 4          | Monitor      | Electronics | 299.99 | 25             |
| 5          | Keyboard     | Electronics | 79.99  | 30             |
```

**Query:**
```sql
SELECT 
    p.category,
    AVG(o.quantity * p.price) AS avg_order_value,
    AVG(o.quantity) AS avg_quantity
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category;
```

**Result:**
```
| category    | avg_order_value | avg_quantity |
|-------------|-----------------|--------------|
| Electronics | 633.325         | 3.75         |
| Furniture   | 749.97          | 3.0          |
```

**Explanation:** The query calculates the average order value by multiplying quantity and price before applying AVG, grouped by product category.

### Example 5: Complex Analysis Scenario

Analyze customer cities by average order size and compare to overall average.

**Sample Data (customers):**
```
| customer_id | first_name | last_name | email              | phone        | city        |
|-------------|------------|-----------|-------------------|--------------|-------------|
| 101         | Sarah      | Wilson    | sarah@email.com   | 555-0101     | New York    |
| 102         | Mike       | Taylor    | mike@email.com    | 555-0102     | Los Angeles |
| 103         | Emma       | Davis     | emma@email.com    | 555-0103     | New York    |
| 104         | James      | Miller    | james@email.com   | 555-0104     | Chicago     |
| 105         | Lisa       | Anderson  | lisa@email.com    | 555-0105     | Los Angeles |
```

**Query:**
```sql
WITH city_averages AS (
    SELECT 
        c.city,
        AVG(o.quantity) AS city_avg_quantity,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.city
),
overall_average AS (
    SELECT AVG(quantity) AS overall_avg
    FROM orders
)
SELECT 
    ca.city,
    ca.city_avg_quantity,
    ca.order_count,
    oa.overall_avg,
    ROUND((ca.city_avg_quantity - oa.overall_avg) / oa.overall_avg * 100, 2) AS pct_diff_from_avg
FROM city_averages ca
CROSS JOIN overall_average oa
ORDER BY ca.city_avg_quantity DESC;
```

**Result:**
```
| city        | city_avg_quantity | order_count | overall_avg | pct_diff_from_avg |
|-------------|-------------------|-------------|-------------|-------------------|
| Los Angeles | 4.5               | 2           | 3.25        | 38.46             |
| Chicago     | 3.0               | 1           | 3.25        | -7.69             |
| New York    | 2.67              | 3           | 3.25        | -17.95            |
```

**Explanation:** This complex query uses CTEs to calculate city-level average order quantities and compare them to the overall average, showing the percentage difference for each city. It demonstrates how AVG can be used in analytical queries to derive business insights.