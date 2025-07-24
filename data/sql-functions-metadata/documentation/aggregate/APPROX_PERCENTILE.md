# APPROX_PERCENTILE

## Description

The `APPROX_PERCENTILE` function approximates the percentile of numeric values in a column using the t-digest algorithm. This provides a memory-efficient way to calculate percentiles over large datasets where exact precision is not required.

## Syntax

```sql
APPROX_PERCENTILE(column, percentile [, accuracy])
```

## Parameters

- `column`: The numeric column to calculate percentiles from (NUMERIC types: INTEGER, BIGINT, DECIMAL, DOUBLE, etc.)
- `percentile`: The percentile to calculate as a decimal between 0 and 1 (DOUBLE). For example, 0.5 for median, 0.95 for 95th percentile
- `accuracy` (optional): The accuracy parameter controlling the precision (INTEGER). Higher values give better accuracy but use more memory. Default is typically 100.

## Return Type

DOUBLE - The approximate percentile value

## Examples

### Example 1: Calculate response time percentiles for API monitoring

```sql
SELECT 
    endpoint,
    COUNT(*) AS request_count,
    APPROX_PERCENTILE(response_time_ms, 0.50) AS p50_response_time,
    APPROX_PERCENTILE(response_time_ms, 0.90) AS p90_response_time,
    APPROX_PERCENTILE(response_time_ms, 0.95) AS p95_response_time,
    APPROX_PERCENTILE(response_time_ms, 0.99) AS p99_response_time,
    MAX(response_time_ms) AS max_response_time
FROM api_logs
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY endpoint
HAVING COUNT(*) > 100
ORDER BY p99_response_time DESC;
```

**Result:**
```
endpoint          | request_count | p50_response_time | p90_response_time | p95_response_time | p99_response_time | max_response_time
------------------|---------------|-------------------|-------------------|-------------------|-------------------|-------------------
/api/search       | 15,234        | 125.5             | 342.8             | 456.2             | 892.3             | 2,345.0
/api/user/profile | 8,456         | 45.2              | 98.7              | 156.3             | 423.1             | 1,234.5
/api/products     | 23,456        | 78.3              | 145.6             | 198.4             | 387.2             | 987.6
/api/checkout     | 3,456         | 234.1             | 456.2             | 567.8             | 789.0             | 1,567.8
```

### Example 2: Analyze salary distribution with custom accuracy

```sql
SELECT 
    department,
    job_level,
    COUNT(*) AS employee_count,
    ROUND(APPROX_PERCENTILE(salary, 0.25), 2) AS salary_p25,
    ROUND(APPROX_PERCENTILE(salary, 0.50), 2) AS salary_median,
    ROUND(APPROX_PERCENTILE(salary, 0.75), 2) AS salary_p75,
    ROUND(APPROX_PERCENTILE(salary, 0.90, 200), 2) AS salary_p90_high_accuracy,
    ROUND(AVG(salary), 2) AS salary_avg
FROM employees
WHERE employment_status = 'active'
GROUP BY department, job_level
HAVING COUNT(*) >= 10
ORDER BY department, job_level;
```

**Result:**
```
department  | job_level | employee_count | salary_p25  | salary_median | salary_p75  | salary_p90_high_accuracy | salary_avg
------------|-----------|----------------|-------------|---------------|-------------|-------------------------|------------
Engineering | Junior    | 45             | 65,000.00   | 72,000.00     | 78,000.00   | 85,000.00               | 71,500.00
Engineering | Senior    | 32             | 95,000.00   | 105,000.00    | 115,000.00  | 125,000.00              | 106,200.00
Engineering | Lead      | 12             | 125,000.00  | 135,000.00    | 145,000.00  | 155,000.00              | 136,800.00
Sales       | Junior    | 28             | 45,000.00   | 52,000.00     | 58,000.00   | 65,000.00               | 52,300.00
Sales       | Senior    | 18             | 65,000.00   | 75,000.00     | 85,000.00   | 95,000.00               | 76,100.00
```

### Example 3: Calculate order value percentiles by customer segment

```sql
WITH customer_orders AS (
    SELECT 
        c.customer_segment,
        o.order_total,
        o.order_date
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'completed'
        AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
)
SELECT 
    customer_segment,
    COUNT(*) AS order_count,
    ROUND(MIN(order_total), 2) AS min_order,
    ROUND(APPROX_PERCENTILE(order_total, 0.10), 2) AS p10_order,
    ROUND(APPROX_PERCENTILE(order_total, 0.50), 2) AS median_order,
    ROUND(APPROX_PERCENTILE(order_total, 0.90), 2) AS p90_order,
    ROUND(APPROX_PERCENTILE(order_total, 0.95), 2) AS p95_order,
    ROUND(MAX(order_total), 2) AS max_order
FROM customer_orders
GROUP BY customer_segment
ORDER BY median_order DESC;
```

**Result:**
```
customer_segment | order_count | min_order | p10_order | median_order | p90_order  | p95_order  | max_order
-----------------|-------------|-----------|-----------|--------------|------------|------------|------------
Premium          | 12,345      | 25.00     | 125.50    | 345.00       | 892.75     | 1,234.50   | 5,678.90
Regular          | 45,678      | 10.00     | 35.25     | 125.75       | 325.80     | 456.25     | 2,345.60
Budget           | 23,456      | 5.00      | 15.50     | 45.80        | 125.90     | 189.75     | 987.50
New              | 8,901       | 8.50      | 22.75     | 78.90        | 234.60     | 345.80     | 1,234.50
```

### Example 4: Analyze page load times across different devices

```sql
SELECT 
    device_type,
    browser,
    COUNT(*) AS page_views,
    ROUND(APPROX_PERCENTILE(load_time_seconds, 0.50), 2) AS median_load_time,
    ROUND(APPROX_PERCENTILE(load_time_seconds, 0.75), 2) AS p75_load_time,
    ROUND(APPROX_PERCENTILE(load_time_seconds, 0.95), 2) AS p95_load_time,
    ROUND(APPROX_PERCENTILE(load_time_seconds, 0.99, 500), 2) AS p99_load_time_accurate
FROM web_performance_metrics
WHERE measurement_date = CURRENT_DATE
    AND load_time_seconds < 30  -- Filter out outliers
GROUP BY device_type, browser
HAVING COUNT(*) > 1000
ORDER BY device_type, median_load_time;
```

**Result:**
```
device_type | browser | page_views | median_load_time | p75_load_time | p95_load_time | p99_load_time_accurate
------------|---------|------------|------------------|---------------|---------------|----------------------
Desktop     | Chrome  | 125,678    | 1.23             | 1.89          | 3.45          | 5.67
Desktop     | Firefox | 87,456     | 1.34             | 2.01          | 3.78          | 6.12
Desktop     | Safari  | 45,234     | 1.45             | 2.23          | 4.12          | 6.89
Mobile      | Chrome  | 234,567    | 2.34             | 3.56          | 6.78          | 10.23
Mobile      | Safari  | 189,012    | 2.45             | 3.78          | 7.12          | 11.34
Tablet      | Safari  | 34,567     | 1.89             | 2.89          | 5.23          | 8.45
```

### Example 5: Calculate percentiles for transaction amounts by hour

```sql
WITH hourly_transactions AS (
    SELECT 
        DATE_TRUNC('hour', transaction_timestamp) AS hour_bucket,
        transaction_type,
        amount
    FROM transactions
    WHERE transaction_timestamp >= NOW() - INTERVAL '24 hours'
        AND status = 'completed'
)
SELECT 
    hour_bucket,
    transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(APPROX_PERCENTILE(amount, 0.25), 2) AS amount_p25,
    ROUND(APPROX_PERCENTILE(amount, 0.50), 2) AS amount_p50,
    ROUND(APPROX_PERCENTILE(amount, 0.75), 2) AS amount_p75,
    ROUND(APPROX_PERCENTILE(amount, 0.95), 2) AS amount_p95,
    ROUND(AVG(amount), 2) AS amount_avg
FROM hourly_transactions
GROUP BY hour_bucket, transaction_type
HAVING COUNT(*) > 50
ORDER BY hour_bucket DESC, transaction_type;
```

**Result:**
```
hour_bucket          | transaction_type | transaction_count | amount_p25 | amount_p50 | amount_p75 | amount_p95 | amount_avg
--------------------|------------------|-------------------|------------|------------|------------|------------|------------
2024-03-15 14:00:00 | purchase         | 3,456             | 25.50      | 67.80      | 125.90     | 345.60     | 98.75
2024-03-15 14:00:00 | refund           | 234               | 15.25      | 45.60      | 89.75      | 234.50     | 67.80
2024-03-15 13:00:00 | purchase         | 3,234             | 22.75      | 65.40      | 118.60     | 325.80     | 92.45
2024-03-15 13:00:00 | refund           | 189               | 12.50      | 38.90      | 78.60      | 198.75     | 58.90
```

## Notes

- The t-digest algorithm provides excellent accuracy for extreme percentiles (near 0 or 1)
- Memory usage is proportional to the accuracy parameter
- For most use cases, the default accuracy provides a good balance between precision and performance
- The function handles NULL values by ignoring them
- Results are interpolated for percentiles that fall between data points
- Particularly useful for large-scale analytics where exact percentiles would be too slow or memory-intensive
- The approximation error is typically less than 1% for most percentiles

## See Also

- [`PERCENTILE_CONT`](../window/PERCENTILE_CONT.md) - Exact percentile calculation
- [`APPROX_QUANTILES`](APPROX_QUANTILES.md) - Calculate multiple quantile boundaries
- [`MEDIAN`](MEDIAN.md) - Exact median calculation (50th percentile)