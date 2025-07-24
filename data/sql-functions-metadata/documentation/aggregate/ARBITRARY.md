# ARBITRARY

## Description

The `ARBITRARY` function returns an arbitrary non-null value from a group of rows. Unlike `ANY_VALUE`, this function specifically excludes NULL values and will only return NULL if all values in the group are NULL. The function is non-deterministic and may return different values for the same input across different query executions.

## Syntax

```sql
ARBITRARY(column)
```

## Parameters

- `column`: The column to select a value from (ANY type)

## Return Type

Same as the input column type

## Examples

### Example 1: Get sample product information per category

```sql
SELECT 
    category,
    COUNT(*) AS product_count,
    ARBITRARY(product_name) AS sample_product,
    ARBITRARY(price) AS sample_price,
    ARBITRARY(brand) AS sample_brand
FROM products
WHERE in_stock = true
GROUP BY category
ORDER BY product_count DESC;
```

**Result:**
```
category    | product_count | sample_product      | sample_price | sample_brand
------------|---------------|--------------------|--------------|--------------
Electronics | 1,234         | Wireless Mouse     | 29.99        | TechBrand
Clothing    | 987           | Cotton T-Shirt     | 19.99        | FashionCo
Books       | 756           | Mystery Novel      | 14.99        | BookPress
Home        | 543           | Coffee Maker       | 89.99        | HomeGoods
Sports      | 321           | Yoga Mat           | 34.99        | FitGear
```

### Example 2: Get representative customer per region for analysis

```sql
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.region,
        c.customer_since,
        COUNT(o.order_id) AS order_count,
        SUM(o.order_total) AS lifetime_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name, c.region, c.customer_since
)
SELECT 
    region,
    COUNT(*) AS customer_count,
    ARBITRARY(customer_name) AS sample_customer,
    ARBITRARY(customer_since) AS sample_join_date,
    ROUND(AVG(lifetime_value), 2) AS avg_lifetime_value
FROM customer_metrics
WHERE lifetime_value > 0
GROUP BY region
ORDER BY avg_lifetime_value DESC;
```

**Result:**
```
region     | customer_count | sample_customer | sample_join_date | avg_lifetime_value
-----------|----------------|-----------------|------------------|-------------------
West       | 12,345         | Sarah Johnson   | 2019-03-15       | 1,234.56
Northeast  | 10,234         | Michael Chen    | 2020-07-22       | 1,156.78
South      | 9,876          | Emily Davis     | 2018-11-03       | 987.65
Midwest    | 8,765          | Robert Wilson   | 2021-01-10       | 876.54
```

### Example 3: Debug data quality issues with sample values

```sql
SELECT 
    table_name,
    column_name,
    data_type,
    COUNT(*) AS null_count,
    ARBITRARY(error_value) AS sample_error_value,
    ARBITRARY(record_id) AS sample_record_id,
    ARBITRARY(error_timestamp) AS sample_error_time
FROM data_quality_issues
WHERE issue_date = CURRENT_DATE
    AND issue_type = 'invalid_format'
GROUP BY table_name, column_name, data_type
HAVING COUNT(*) > 10
ORDER BY null_count DESC;
```

**Result:**
```
table_name   | column_name  | data_type | null_count | sample_error_value | sample_record_id | sample_error_time
-------------|--------------|-----------|------------|-------------------|------------------|-------------------
orders       | phone_number | VARCHAR   | 234        | 555-CALL          | ORD-78901        | 2024-03-15 10:23:45
customers    | email        | VARCHAR   | 156        | notanemail        | CUST-34567       | 2024-03-15 09:15:22
transactions | amount       | DECIMAL   | 89         | $1,234.56         | TXN-56789        | 2024-03-15 11:45:33
products     | weight       | DOUBLE    | 67         | 10 pounds         | PROD-23456       | 2024-03-15 08:30:15
```

### Example 4: Sample configuration values per environment

```sql
WITH config_data AS (
    SELECT 
        environment,
        config_key,
        config_value,
        last_updated,
        updated_by
    FROM system_configurations
    WHERE is_active = true
)
SELECT 
    environment,
    COUNT(DISTINCT config_key) AS config_count,
    ARBITRARY(config_key) AS sample_key,
    ARBITRARY(config_value) AS sample_value,
    ARBITRARY(updated_by) AS sample_updater,
    MAX(last_updated) AS latest_update
FROM config_data
GROUP BY environment
ORDER BY environment;
```

**Result:**
```
environment | config_count | sample_key           | sample_value | sample_updater | latest_update
------------|--------------|---------------------|--------------|----------------|---------------
development | 145          | api.timeout.seconds | 30           | dev_admin      | 2024-03-15 14:30:00
production  | 132          | cache.ttl.minutes   | 60           | ops_team       | 2024-03-14 22:15:00
staging     | 138          | log.level           | INFO         | qa_engineer    | 2024-03-15 11:45:00
test        | 150          | db.pool.size        | 10           | test_admin     | 2024-03-15 09:00:00
```

### Example 5: Get sample transaction per payment method and status

```sql
SELECT 
    payment_method,
    transaction_status,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    ARBITRARY(transaction_id) AS sample_transaction,
    ARBITRARY(merchant_name) AS sample_merchant,
    ARBITRARY(transaction_date) AS sample_date
FROM payment_transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
GROUP BY payment_method, transaction_status
ORDER BY payment_method, transaction_count DESC;
```

**Result:**
```
payment_method | transaction_status | transaction_count | total_amount  | sample_transaction | sample_merchant    | sample_date
---------------|-------------------|-------------------|---------------|-------------------|-------------------|-------------
credit_card    | completed         | 45,678            | 2,345,678.90  | TXN-CC-98765      | Online Store ABC  | 2024-03-14
credit_card    | pending           | 1,234             | 56,789.12     | TXN-CC-34567      | Restaurant XYZ    | 2024-03-15
credit_card    | failed            | 567               | 12,345.67     | TXN-CC-12345      | Gas Station 123   | 2024-03-13
debit_card     | completed         | 34,567            | 1,234,567.89  | TXN-DC-87654      | Grocery Mart      | 2024-03-15
debit_card     | pending           | 890               | 23,456.78     | TXN-DC-45678      | Coffee Shop       | 2024-03-14
paypal         | completed         | 12,345            | 567,890.12    | TXN-PP-76543      | Tech Gadgets Inc  | 2024-03-12
```

## Notes

- `ARBITRARY` specifically excludes NULL values from selection
- Returns NULL only when all values in the group are NULL
- The function is non-deterministic - may return different values on repeated executions
- Useful for sampling data when you need any representative non-null value
- More predictable than `ANY_VALUE` when you need to ensure non-null results
- Often used in GROUP BY queries where you need a sample value but the specific value doesn't matter
- Can be combined with other aggregate functions to provide context

## See Also

- [`ANY_VALUE`](ANY_VALUE.md) - Returns any value including NULLs
- [`FIRST_VALUE`](../window/FIRST_VALUE.md) - Deterministic first value using window functions
- [`MAX`](MAX.md) - Returns maximum value (deterministic)