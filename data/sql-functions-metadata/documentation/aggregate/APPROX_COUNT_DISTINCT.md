# APPROX_COUNT_DISTINCT

## Description

The `APPROX_COUNT_DISTINCT` function approximates the number of distinct values in a column using the HyperLogLog algorithm. This function provides a fast, memory-efficient alternative to `COUNT(DISTINCT)` for large datasets where an exact count is not required.

## Syntax

```sql
APPROX_COUNT_DISTINCT(column [, error_rate])
```

## Parameters

- `column`: The column to count distinct values from (ANY type)
- `error_rate` (optional): The desired error rate as a decimal between 0 and 1 (DOUBLE). Default is typically 0.023 (2.3% error)

## Return Type

BIGINT - The approximate count of distinct values

## Examples

### Example 1: Approximate unique visitors per day

```sql
SELECT 
    DATE(visit_timestamp) AS visit_date,
    COUNT(*) AS total_visits,
    APPROX_COUNT_DISTINCT(user_id) AS unique_visitors,
    ROUND(COUNT(*) * 1.0 / APPROX_COUNT_DISTINCT(user_id), 2) AS avg_visits_per_user
FROM website_visits
WHERE visit_timestamp >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
GROUP BY DATE(visit_timestamp)
ORDER BY visit_date DESC;
```

**Result:**
```
visit_date  | total_visits | unique_visitors | avg_visits_per_user
------------|--------------|-----------------|--------------------
2024-03-15  | 1,250,000    | 425,000         | 2.94
2024-03-14  | 1,180,000    | 398,000         | 2.96
2024-03-13  | 1,350,000    | 445,000         | 3.03
2024-03-12  | 1,100,000    | 380,000         | 2.89
2024-03-11  | 950,000      | 325,000         | 2.92
```

### Example 2: Approximate distinct products sold per category with custom error rate

```sql
SELECT 
    p.category,
    COUNT(*) AS total_items_sold,
    APPROX_COUNT_DISTINCT(oi.product_id) AS unique_products_sold,
    APPROX_COUNT_DISTINCT(oi.product_id, 0.01) AS unique_products_1pct_error,
    APPROX_COUNT_DISTINCT(o.customer_id) AS unique_customers
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY p.category
ORDER BY total_items_sold DESC;
```

**Result:**
```
category    | total_items_sold | unique_products_sold | unique_products_1pct_error | unique_customers
------------|------------------|---------------------|---------------------------|------------------
Electronics | 45,678           | 1,234               | 1,238                     | 15,432
Clothing    | 38,902           | 3,456               | 3,461                     | 12,876
Books       | 28,345           | 8,901               | 8,912                     | 9,234
Home        | 23,456           | 2,345               | 2,348                     | 8,765
```

### Example 3: Approximate unique IP addresses per hour for security monitoring

```sql
WITH hourly_traffic AS (
    SELECT 
        DATE_TRUNC('hour', request_timestamp) AS hour_bucket,
        APPROX_COUNT_DISTINCT(ip_address) AS unique_ips,
        APPROX_COUNT_DISTINCT(user_agent) AS unique_user_agents,
        COUNT(*) AS total_requests
    FROM web_logs
    WHERE request_timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', request_timestamp)
)
SELECT 
    hour_bucket,
    unique_ips,
    unique_user_agents,
    total_requests,
    ROUND(total_requests * 1.0 / unique_ips, 2) AS avg_requests_per_ip
FROM hourly_traffic
WHERE unique_ips > 1000
ORDER BY hour_bucket DESC;
```

**Result:**
```
hour_bucket          | unique_ips | unique_user_agents | total_requests | avg_requests_per_ip
--------------------|------------|-------------------|----------------|--------------------
2024-03-15 14:00:00 | 25,432     | 1,234             | 125,678        | 4.94
2024-03-15 13:00:00 | 28,901     | 1,345             | 142,345        | 4.92
2024-03-15 12:00:00 | 31,234     | 1,456             | 178,901        | 5.73
2024-03-15 11:00:00 | 27,654     | 1,234             | 134,567        | 4.87
```

### Example 4: Compare approximate vs exact counts for performance analysis

```sql
-- Using a CTE to calculate both for comparison
WITH sales_data AS (
    SELECT 
        store_region,
        COUNT(DISTINCT customer_id) AS exact_customers,
        APPROX_COUNT_DISTINCT(customer_id) AS approx_customers,
        APPROX_COUNT_DISTINCT(customer_id, 0.05) AS approx_customers_5pct,
        APPROX_COUNT_DISTINCT(customer_id, 0.01) AS approx_customers_1pct
    FROM transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
    GROUP BY store_region
)
SELECT 
    store_region,
    exact_customers,
    approx_customers,
    ROUND(ABS(exact_customers - approx_customers) * 100.0 / exact_customers, 2) AS default_error_pct,
    approx_customers_5pct,
    ROUND(ABS(exact_customers - approx_customers_5pct) * 100.0 / exact_customers, 2) AS error_5pct,
    approx_customers_1pct,
    ROUND(ABS(exact_customers - approx_customers_1pct) * 100.0 / exact_customers, 2) AS error_1pct
FROM sales_data
ORDER BY exact_customers DESC;
```

**Result:**
```
store_region | exact_customers | approx_customers | default_error_pct | approx_customers_5pct | error_5pct | approx_customers_1pct | error_1pct
-------------|-----------------|------------------|-------------------|----------------------|------------|----------------------|------------
East         | 125,432         | 124,890          | 0.43              | 125,100              | 0.26       | 125,380              | 0.04
West         | 98,765          | 97,234           | 1.55              | 98,432               | 0.34       | 98,701               | 0.06
Central      | 76,543          | 75,890           | 0.85              | 76,234               | 0.40       | 76,512               | 0.04
South        | 54,321          | 53,456           | 1.59              | 54,123               | 0.36       | 54,298               | 0.04
```

### Example 5: Approximate cardinality for data profiling

```sql
SELECT 
    'customers' AS table_name,
    APPROX_COUNT_DISTINCT(customer_id) AS approx_unique_ids,
    APPROX_COUNT_DISTINCT(email) AS approx_unique_emails,
    APPROX_COUNT_DISTINCT(phone_number) AS approx_unique_phones,
    APPROX_COUNT_DISTINCT(CONCAT(first_name, ' ', last_name)) AS approx_unique_names,
    APPROX_COUNT_DISTINCT(city) AS approx_unique_cities,
    APPROX_COUNT_DISTINCT(state) AS approx_unique_states,
    APPROX_COUNT_DISTINCT(zip_code) AS approx_unique_zips
FROM customers
UNION ALL
SELECT 
    'orders' AS table_name,
    APPROX_COUNT_DISTINCT(order_id),
    APPROX_COUNT_DISTINCT(customer_id),
    NULL AS approx_unique_phones,
    NULL AS approx_unique_names,
    APPROX_COUNT_DISTINCT(shipping_city),
    APPROX_COUNT_DISTINCT(shipping_state),
    APPROX_COUNT_DISTINCT(shipping_zip)
FROM orders
ORDER BY table_name;
```

**Result:**
```
table_name | approx_unique_ids | approx_unique_emails | approx_unique_phones | approx_unique_names | approx_unique_cities | approx_unique_states | approx_unique_zips
-----------|-------------------|---------------------|---------------------|--------------------|--------------------|---------------------|-------------------
customers  | 2,345,678         | 2,342,123           | 2,298,765           | 2,301,234          | 12,345             | 52                  | 28,901
orders     | 8,765,432         | 1,987,654           | NULL                | NULL               | 11,234             | 51                  | 25,678
```

## Notes

- HyperLogLog provides excellent accuracy for large cardinalities with minimal memory usage
- The default error rate is typically around 2.3% (0.023)
- Lower error rates require more memory but provide better accuracy
- For small datasets (< 1000 distinct values), the approximation may be less accurate
- NULL values are not counted as distinct values
- The function is deterministic - same input always produces the same approximate count
- Significantly faster than `COUNT(DISTINCT)` for large datasets, often 10-100x faster

## See Also

- [`COUNT`](COUNT.md) - Exact count of rows
- [`APPROX_PERCENTILE`](APPROX_PERCENTILE.md) - Approximate percentile calculations
- [`APPROX_QUANTILES`](APPROX_QUANTILES.md) - Approximate quantile boundaries