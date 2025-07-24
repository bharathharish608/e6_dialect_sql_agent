# APPROX_QUANTILES

## Description

The `APPROX_QUANTILES` function approximates quantile boundaries for numeric values, dividing the data into a specified number of equally-sized buckets. It returns an array of boundary values that separate these quantiles, useful for understanding data distribution and creating histogram buckets.

## Syntax

```sql
APPROX_QUANTILES(column, n)
```

## Parameters

- `column`: The numeric column to calculate quantiles from (NUMERIC types: INTEGER, BIGINT, DECIMAL, DOUBLE, etc.)
- `n`: The number of quantiles to create (INTEGER). Returns n+1 boundary values.

## Return Type

ARRAY of DOUBLE - An array containing n+1 values representing the boundaries of n quantiles, including minimum and maximum values

## Examples

### Example 1: Create income distribution buckets

```sql
SELECT 
    APPROX_QUANTILES(annual_income, 10) AS income_deciles,
    COUNT(*) AS total_customers,
    MIN(annual_income) AS min_income,
    MAX(annual_income) AS max_income
FROM customers
WHERE annual_income IS NOT NULL
    AND account_status = 'active';
```

**Result:**
```
income_deciles                                                                    | total_customers | min_income | max_income
---------------------------------------------------------------------------------|-----------------|------------|------------
[15000, 28000, 35000, 42000, 51000, 62000, 75000, 92000, 118000, 165000, 500000] | 125,678        | 15,000     | 500,000
```

### Example 2: Analyze response time distribution for different services

```sql
WITH service_quantiles AS (
    SELECT 
        service_name,
        APPROX_QUANTILES(response_time_ms, 4) AS quartile_boundaries,
        COUNT(*) AS request_count
    FROM api_metrics
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
    GROUP BY service_name
)
SELECT 
    service_name,
    request_count,
    quartile_boundaries[1] AS min_time,
    quartile_boundaries[2] AS q1_time,
    quartile_boundaries[3] AS median_time,
    quartile_boundaries[4] AS q3_time,
    quartile_boundaries[5] AS max_time
FROM service_quantiles
ORDER BY median_time;
```

**Result:**
```
service_name     | request_count | min_time | q1_time | median_time | q3_time | max_time
-----------------|---------------|----------|---------|-------------|---------|----------
user-service     | 45,678        | 12.5     | 45.2    | 78.6        | 125.3   | 892.7
product-service  | 67,890        | 23.4     | 67.8    | 95.4        | 145.6   | 1234.5
payment-service  | 12,345        | 45.6     | 123.4   | 234.5       | 456.7   | 2345.6
search-service   | 89,012        | 67.8     | 234.5   | 345.6       | 567.8   | 3456.7
```

### Example 3: Create age distribution buckets for marketing segments

```sql
SELECT 
    marketing_segment,
    APPROX_QUANTILES(age, 5) AS age_quintiles,
    COUNT(*) AS customer_count
FROM customer_demographics
WHERE age BETWEEN 18 AND 80
GROUP BY marketing_segment
ORDER BY marketing_segment;
```

**Result:**
```
marketing_segment | age_quintiles                    | customer_count
------------------|----------------------------------|----------------
Budget Conscious  | [18, 24, 31, 39, 52, 78]        | 34,567
Early Adopters    | [21, 26, 29, 34, 42, 65]        | 23,456
Family Focused    | [25, 32, 38, 45, 54, 72]        | 45,678
Luxury Seekers    | [28, 38, 45, 55, 65, 80]        | 12,345
Tech Savvy        | [18, 23, 28, 35, 48, 68]        | 56,789
```

### Example 4: Analyze order size distribution by day of week

```sql
WITH daily_quantiles AS (
    SELECT 
        DAYNAME(order_date) AS day_of_week,
        DAYOFWEEK(order_date) AS day_number,
        APPROX_QUANTILES(order_item_count, 10) AS item_count_deciles,
        COUNT(*) AS order_count,
        SUM(order_item_count) AS total_items
    FROM (
        SELECT 
            o.order_id,
            o.order_date,
            COUNT(oi.item_id) AS order_item_count
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
        GROUP BY o.order_id, o.order_date
    ) order_summary
    GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
)
SELECT 
    day_of_week,
    order_count,
    total_items,
    ROUND(total_items * 1.0 / order_count, 2) AS avg_items_per_order,
    item_count_deciles
FROM daily_quantiles
ORDER BY day_number;
```

**Result:**
```
day_of_week | order_count | total_items | avg_items_per_order | item_count_deciles
------------|-------------|-------------|---------------------|-------------------------------------------
Sunday      | 12,345      | 45,678      | 3.70                | [1, 1, 2, 2, 3, 4, 5, 6, 8, 12, 45]
Monday      | 18,234      | 72,345      | 3.97                | [1, 1, 2, 3, 3, 4, 5, 7, 9, 14, 52]
Tuesday     | 19,456      | 78,901      | 4.06                | [1, 2, 2, 3, 4, 4, 5, 7, 10, 15, 48]
Wednesday   | 20,123      | 82,345      | 4.09                | [1, 2, 2, 3, 4, 5, 6, 7, 10, 16, 55]
Thursday    | 21,234      | 89,012      | 4.19                | [1, 2, 3, 3, 4, 5, 6, 8, 11, 17, 58]
Friday      | 23,456      | 102,345     | 4.36                | [1, 2, 3, 4, 4, 5, 6, 8, 12, 18, 62]
Saturday    | 15,678      | 67,890      | 4.33                | [1, 2, 3, 3, 4, 5, 6, 8, 11, 17, 59]
```

### Example 5: Create percentile buckets for performance scoring

```sql
WITH score_distribution AS (
    SELECT 
        department,
        employee_id,
        performance_score
    FROM employee_reviews
    WHERE review_year = 2024
        AND review_status = 'completed'
),
department_quantiles AS (
    SELECT 
        department,
        APPROX_QUANTILES(performance_score, 100) AS percentile_boundaries
    FROM score_distribution
    GROUP BY department
)
SELECT 
    sd.department,
    COUNT(*) AS employee_count,
    dq.percentile_boundaries[11] AS p10_score,
    dq.percentile_boundaries[26] AS p25_score,
    dq.percentile_boundaries[51] AS p50_score,
    dq.percentile_boundaries[76] AS p75_score,
    dq.percentile_boundaries[91] AS p90_score,
    ROUND(AVG(sd.performance_score), 2) AS avg_score
FROM score_distribution sd
JOIN department_quantiles dq ON sd.department = dq.department
GROUP BY sd.department, dq.percentile_boundaries
ORDER BY avg_score DESC;
```

**Result:**
```
department    | employee_count | p10_score | p25_score | p50_score | p75_score | p90_score | avg_score
--------------|----------------|-----------|-----------|-----------|-----------|-----------|----------
Engineering   | 234            | 72.5      | 78.2      | 85.6      | 91.3      | 95.8      | 84.7
Sales         | 156            | 68.4      | 75.6      | 83.2      | 89.7      | 94.2      | 82.3
Product       | 89             | 70.2      | 76.8      | 82.9      | 88.5      | 93.6      | 81.8
Marketing     | 78             | 66.7      | 73.4      | 80.5      | 87.2      | 92.1      | 79.6
Support       | 123            | 64.3      | 71.2      | 78.9      | 85.6      | 90.4      | 77.8
```

## Notes

- Returns n+1 boundary values for n quantiles (includes min and max)
- The first element is always the minimum value, and the last is always the maximum
- NULL values are ignored in the calculation
- The function uses approximation algorithms for efficiency on large datasets
- Useful for creating histogram buckets or understanding data distribution
- Array indices are 1-based in most SQL systems
- For exact quantiles on smaller datasets, consider using window functions with NTILE

## See Also

- [`APPROX_PERCENTILE`](APPROX_PERCENTILE.md) - Calculate specific percentile values
- [`NTILE`](../window/NTILE.md) - Divide rows into buckets using window functions
- [`WIDTH_BUCKET`](../scalar/WIDTH_BUCKET.md) - Assign values to histogram buckets