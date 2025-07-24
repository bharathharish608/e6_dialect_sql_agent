# ANY_VALUE

## Description

The `ANY_VALUE` function returns any value from a group of rows, including NULL values. This function is non-deterministic and may return different values for the same input across different query executions. It's particularly useful when you need a value from a grouped column that isn't part of the GROUP BY clause and the specific value doesn't matter.

## Syntax

```sql
ANY_VALUE(column)
```

## Parameters

- `column`: The column to select a value from (ANY type)

## Return Type

Same as the input column type

## Examples

### Example 1: Get any email address per customer when grouping by customer ID

```sql
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(order_total) AS total_spent,
    ANY_VALUE(customer_email) AS email,
    ANY_VALUE(customer_name) AS name
FROM orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
GROUP BY customer_id
HAVING total_spent > 1000
ORDER BY total_spent DESC
LIMIT 10;
```

**Result:**
```
customer_id | order_count | total_spent | email                    | name
------------|-------------|-------------|--------------------------|----------------
C12345      | 45          | 8,765.50    | john.smith@email.com     | John Smith
C67890      | 32          | 6,543.25    | sarah.jones@email.com    | Sarah Jones
C11111      | 28          | 5,432.10    | mike.wilson@email.com    | Michael Wilson
C22222      | 25          | 4,321.75    | emma.davis@email.com     | Emma Davis
C33333      | 22          | 3,876.90    | NULL                     | Robert Brown
```

### Example 2: Simplify complex aggregations with non-grouped columns

```sql
WITH product_sales AS (
    SELECT 
        p.product_id,
        p.category_id,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.quantity * oi.unit_price) AS revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE oi.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
    GROUP BY p.product_id, p.category_id
)
SELECT 
    ps.category_id,
    ANY_VALUE(c.category_name) AS category_name,
    COUNT(DISTINCT ps.product_id) AS product_count,
    SUM(ps.units_sold) AS total_units,
    SUM(ps.revenue) AS total_revenue,
    ROUND(SUM(ps.revenue) / SUM(ps.units_sold), 2) AS avg_price_per_unit
FROM product_sales ps
JOIN categories c ON ps.category_id = c.category_id
GROUP BY ps.category_id
ORDER BY total_revenue DESC;
```

**Result:**
```
category_id | category_name | product_count | total_units | total_revenue | avg_price_per_unit
------------|---------------|---------------|-------------|---------------|-------------------
CAT-101     | Electronics   | 145           | 12,345      | 987,654.32    | 79.99
CAT-102     | Clothing      | 234           | 23,456      | 654,321.10    | 27.90
CAT-103     | Home & Garden | 189           | 18,765      | 456,789.25    | 24.34
CAT-104     | Books         | 567           | 34,567      | 234,567.89    | 6.79
```

### Example 3: Handle multiple address fields in customer aggregations

```sql
SELECT 
    city,
    state,
    COUNT(DISTINCT customer_id) AS customer_count,
    COUNT(*) AS address_count,
    ANY_VALUE(zip_code) AS sample_zip,
    ANY_VALUE(country) AS country,
    ANY_VALUE(timezone) AS timezone
FROM customer_addresses
WHERE is_active = true
GROUP BY city, state
HAVING COUNT(DISTINCT customer_id) > 100
ORDER BY customer_count DESC
LIMIT 20;
```

**Result:**
```
city          | state | customer_count | address_count | sample_zip | country | timezone
--------------|-------|----------------|---------------|------------|---------|----------
New York      | NY    | 5,432          | 6,789         | 10001      | USA     | America/New_York
Los Angeles   | CA    | 4,321          | 5,234         | 90001      | USA     | America/Los_Angeles
Chicago       | IL    | 3,456          | 4,123         | NULL       | USA     | America/Chicago
Houston       | TX    | 2,345          | 2,789         | 77001      | USA     | America/Chicago
Phoenix       | AZ    | 1,876          | 2,134         | 85001      | USA     | America/Phoenix
```

### Example 4: Simplify session analytics with user attributes

```sql
WITH session_metrics AS (
    SELECT 
        session_id,
        user_id,
        DATE(session_start) AS session_date,
        COUNT(*) AS page_views,
        SUM(time_on_page) AS total_time,
        MAX(session_end) - MIN(session_start) AS session_duration
    FROM web_analytics
    WHERE session_start >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
    GROUP BY session_id, user_id, DATE(session_start)
)
SELECT 
    sm.session_date,
    COUNT(DISTINCT sm.user_id) AS unique_users,
    COUNT(DISTINCT sm.session_id) AS total_sessions,
    AVG(sm.page_views) AS avg_pages_per_session,
    AVG(sm.session_duration) AS avg_session_duration,
    ANY_VALUE(u.user_type) AS sample_user_type,
    ANY_VALUE(u.acquisition_channel) AS sample_channel
FROM session_metrics sm
LEFT JOIN users u ON sm.user_id = u.user_id
GROUP BY sm.session_date
ORDER BY sm.session_date DESC;
```

**Result:**
```
session_date | unique_users | total_sessions | avg_pages_per_session | avg_session_duration | sample_user_type | sample_channel
-------------|--------------|----------------|-----------------------|---------------------|------------------|----------------
2024-03-15   | 125,678      | 234,567        | 5.67                  | 00:12:34            | premium          | organic
2024-03-14   | 118,234      | 221,345        | 5.45                  | 00:11:45            | NULL             | paid_search
2024-03-13   | 132,456      | 245,678        | 5.89                  | 00:13:12            | free             | social
2024-03-12   | 109,876      | 198,765        | 5.23                  | 00:10:56            | premium          | direct
```

### Example 5: Aggregate product reviews with metadata

```sql
SELECT 
    product_id,
    COUNT(*) AS review_count,
    AVG(rating) AS avg_rating,
    COUNT(CASE WHEN rating >= 4 THEN 1 END) AS positive_reviews,
    COUNT(CASE WHEN rating <= 2 THEN 1 END) AS negative_reviews,
    ANY_VALUE(product_name) AS product_name,
    ANY_VALUE(category) AS category,
    ANY_VALUE(brand) AS brand
FROM product_reviews pr
JOIN products p USING (product_id)
WHERE pr.review_date >= DATE_SUB(CURRENT_DATE, INTERVAL 180 DAY)
    AND pr.verified_purchase = true
GROUP BY product_id
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC, review_count DESC
LIMIT 20;
```

**Result:**
```
product_id | review_count | avg_rating | positive_reviews | negative_reviews | product_name           | category    | brand
-----------|--------------|------------|------------------|------------------|------------------------|-------------|-------------
P-1234     | 567          | 4.85       | 523              | 12               | Wireless Headphones    | Electronics | AudioPro
P-5678     | 432          | 4.82       | 398              | 8                | Organic Coffee Beans   | Food        | NULL
P-9012     | 321          | 4.79       | 289              | 15               | Yoga Mat Premium       | Sports      | FitLife
P-3456     | 234          | 4.75       | 201              | 10               | Smart Watch           | Electronics | TechGear
P-7890     | 198          | 4.72       | 167              | 9                | Novel Collection      | Books       | BookWorld
```

## Notes

- `ANY_VALUE` can return NULL values, unlike `ARBITRARY`
- The function is non-deterministic - results may vary between executions
- Particularly useful for avoiding "not in GROUP BY" errors when the specific value doesn't matter
- Commonly used in aggregation queries where you need to include non-grouped columns
- More permissive than `ARBITRARY` as it doesn't filter out NULL values
- Can improve query performance by avoiding unnecessary grouping on columns where any value is acceptable
- Standard SQL function supported by many database systems

## See Also

- [`ARBITRARY`](ARBITRARY.md) - Returns arbitrary non-null values only
- [`FIRST_VALUE`](../window/FIRST_VALUE.md) - Deterministic first value using window functions
- [`MIN`](MIN.md) / [`MAX`](MAX.md) - Deterministic boundary values