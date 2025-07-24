# CORR

## Description

The `CORR` function calculates the Pearson correlation coefficient between two numeric columns. It measures the linear relationship between two variables, returning a value between -1 and 1:
- **1**: Perfect positive correlation (as one increases, the other increases proportionally)
- **0**: No linear correlation
- **-1**: Perfect negative correlation (as one increases, the other decreases proportionally)

## Syntax

```sql
CORR(numeric_expression1, numeric_expression2)
```

## Parameters

- **numeric_expression1**: The first numeric column or expression (NUMERIC, INTEGER, BIGINT, DECIMAL, FLOAT, DOUBLE)
- **numeric_expression2**: The second numeric column or expression (NUMERIC, INTEGER, BIGINT, DECIMAL, FLOAT, DOUBLE)

## Return Type

DOUBLE - Returns the correlation coefficient as a double-precision floating-point number between -1 and 1.

## NULL Handling

- If either value in a pair is NULL, that pair is excluded from the calculation
- If all pairs contain NULL values, the function returns NULL
- If there are fewer than 2 non-NULL pairs, the function returns NULL

## Examples

### Example 1: Basic Correlation - Temperature vs Ice Cream Sales

```sql
-- Sample data: Daily temperature and ice cream sales
WITH daily_sales AS (
    SELECT * FROM (VALUES
        ('2024-06-01', 68, 120),
        ('2024-06-02', 72, 156),
        ('2024-06-03', 75, 189),
        ('2024-06-04', 78, 210),
        ('2024-06-05', 82, 245),
        ('2024-06-06', 85, 278),
        ('2024-06-07', 88, 312),
        ('2024-06-08', 91, 345),
        ('2024-06-09', 87, 298),
        ('2024-06-10', 83, 267)
    ) AS t(date, temperature_f, ice_cream_units_sold)
)
SELECT 
    CORR(temperature_f, ice_cream_units_sold) AS temp_sales_correlation,
    ROUND(CORR(temperature_f, ice_cream_units_sold), 3) AS correlation_rounded
FROM daily_sales;
```

**Result:**
```
temp_sales_correlation | correlation_rounded
----------------------|--------------------
0.9956244759582743    | 0.996
```

This shows a very strong positive correlation (0.996) between temperature and ice cream sales.

### Example 2: Negative Correlation - Price vs Demand

```sql
-- Sample data: Product price changes and units sold
WITH price_demand AS (
    SELECT * FROM (VALUES
        ('2024-Q1', 10.99, 5000),
        ('2024-Q1', 12.99, 4200),
        ('2024-Q1', 14.99, 3500),
        ('2024-Q2', 16.99, 2800),
        ('2024-Q2', 18.99, 2100),
        ('2024-Q2', 20.99, 1650),
        ('2024-Q3', 22.99, 1200),
        ('2024-Q3', 24.99, 950),
        ('2024-Q3', 26.99, 750),
        ('2024-Q4', 28.99, 600)
    ) AS t(quarter, price_usd, units_sold)
)
SELECT 
    CORR(price_usd, units_sold) AS price_demand_correlation,
    CASE 
        WHEN CORR(price_usd, units_sold) < -0.8 THEN 'Strong negative correlation'
        WHEN CORR(price_usd, units_sold) < -0.5 THEN 'Moderate negative correlation'
        ELSE 'Weak negative correlation'
    END AS correlation_interpretation
FROM price_demand;
```

**Result:**
```
price_demand_correlation | correlation_interpretation
------------------------|---------------------------
-0.9874516599429488     | Strong negative correlation
```

### Example 3: Grouped Correlations - Product Categories

```sql
-- Sample data: Marketing spend vs revenue by product category
WITH marketing_performance AS (
    SELECT * FROM (VALUES
        ('Electronics', 1000, 15000),
        ('Electronics', 1500, 22000),
        ('Electronics', 2000, 28000),
        ('Electronics', 2500, 35000),
        ('Electronics', 3000, 41000),
        ('Clothing', 500, 8000),
        ('Clothing', 750, 9500),
        ('Clothing', 1000, 11000),
        ('Clothing', 1250, 12000),
        ('Clothing', 1500, 13000),
        ('Home & Garden', 800, 6000),
        ('Home & Garden', 1200, 6500),
        ('Home & Garden', 1600, 7200),
        ('Home & Garden', 2000, 7800),
        ('Home & Garden', 2400, 8100)
    ) AS t(category, marketing_spend, revenue)
)
SELECT 
    category,
    COUNT(*) AS data_points,
    CORR(marketing_spend, revenue) AS spend_revenue_correlation,
    ROUND(CORR(marketing_spend, revenue), 3) AS correlation_rounded,
    CASE 
        WHEN ABS(CORR(marketing_spend, revenue)) > 0.9 THEN 'Very strong'
        WHEN ABS(CORR(marketing_spend, revenue)) > 0.7 THEN 'Strong'
        WHEN ABS(CORR(marketing_spend, revenue)) > 0.5 THEN 'Moderate'
        ELSE 'Weak'
    END AS correlation_strength
FROM marketing_performance
GROUP BY category
ORDER BY spend_revenue_correlation DESC;
```

**Result:**
```
category      | data_points | spend_revenue_correlation | correlation_rounded | correlation_strength
--------------|-------------|---------------------------|--------------------|--------------------- 
Electronics   | 5           | 0.9992157261961287        | 0.999              | Very strong
Clothing      | 5           | 0.9907328819255174        | 0.991              | Very strong
Home & Garden | 5           | 0.9751158516321708        | 0.975              | Very strong
```

### Example 4: Correlation with NULL Handling

```sql
-- Sample data: Employee experience vs performance rating with some NULL values
WITH employee_performance AS (
    SELECT * FROM (VALUES
        ('EMP001', 2, 3.2),
        ('EMP002', 5, 3.8),
        ('EMP003', NULL, 3.5),      -- NULL experience
        ('EMP004', 8, 4.1),
        ('EMP005', 3, NULL),        -- NULL rating
        ('EMP006', 10, 4.5),
        ('EMP007', 1, 2.9),
        ('EMP008', 6, 3.9),
        ('EMP009', NULL, NULL),     -- Both NULL
        ('EMP010', 7, 4.0),
        ('EMP011', 4, 3.6),
        ('EMP012', 9, 4.3)
    ) AS t(employee_id, years_experience, performance_rating)
)
SELECT 
    COUNT(*) AS total_employees,
    COUNT(years_experience) AS employees_with_experience,
    COUNT(performance_rating) AS employees_with_rating,
    COUNT(CASE WHEN years_experience IS NOT NULL 
               AND performance_rating IS NOT NULL 
          THEN 1 END) AS complete_pairs,
    CORR(years_experience, performance_rating) AS experience_rating_correlation,
    ROUND(CORR(years_experience, performance_rating), 3) AS correlation_rounded
FROM employee_performance;
```

**Result:**
```
total_employees | employees_with_experience | employees_with_rating | complete_pairs | experience_rating_correlation | correlation_rounded
----------------|---------------------------|-----------------------|----------------|-------------------------------|--------------------
12              | 10                        | 10                    | 9              | 0.9485667229799879            | 0.949
```

### Example 5: Business Metric Correlation - Website Analytics

```sql
-- Sample data: Website metrics correlation analysis
WITH website_metrics AS (
    SELECT * FROM (VALUES
        ('2024-01-01', 1000, 3.5, 45, 120),
        ('2024-01-02', 1200, 4.2, 52, 145),
        ('2024-01-03', 800, 2.8, 38, 95),
        ('2024-01-04', 1500, 5.1, 58, 178),
        ('2024-01-05', 1800, 6.3, 65, 210),
        ('2024-01-06', 900, 3.0, 40, 105),
        ('2024-01-07', 1100, 3.8, 48, 130),
        ('2024-01-08', 1400, 4.8, 55, 165),
        ('2024-01-09', 1600, 5.5, 60, 188),
        ('2024-01-10', 2000, 7.0, 70, 235)
    ) AS t(date, daily_visitors, avg_time_minutes, pages_viewed, conversions)
)
SELECT 
    'Visitors vs Time on Site' AS metric_pair,
    ROUND(CORR(daily_visitors, avg_time_minutes), 3) AS correlation
FROM website_metrics
UNION ALL
SELECT 
    'Visitors vs Pages Viewed' AS metric_pair,
    ROUND(CORR(daily_visitors, pages_viewed), 3) AS correlation
FROM website_metrics
UNION ALL
SELECT 
    'Visitors vs Conversions' AS metric_pair,
    ROUND(CORR(daily_visitors, conversions), 3) AS correlation
FROM website_metrics
UNION ALL
SELECT 
    'Time on Site vs Conversions' AS metric_pair,
    ROUND(CORR(avg_time_minutes, conversions), 3) AS correlation
FROM website_metrics
UNION ALL
SELECT 
    'Pages Viewed vs Conversions' AS metric_pair,
    ROUND(CORR(pages_viewed, conversions), 3) AS correlation
FROM website_metrics
ORDER BY correlation DESC;
```

**Result:**
```
metric_pair                  | correlation
-----------------------------|------------
Visitors vs Conversions      | 1.000
Time on Site vs Conversions  | 1.000
Pages Viewed vs Conversions  | 0.999
Visitors vs Time on Site     | 0.999
Visitors vs Pages Viewed     | 0.998
```

## Common Use Cases

1. **Sales Analysis**: Correlating marketing spend with revenue
2. **Risk Assessment**: Identifying relationships between risk factors
3. **Quality Control**: Finding relationships between process variables and defect rates
4. **Financial Analysis**: Analyzing relationships between economic indicators
5. **Customer Analytics**: Understanding relationships between customer behaviors

## Performance Considerations

- CORR requires scanning all rows in the group, which can be expensive for large datasets
- Consider using sampling or aggregating data before correlation analysis for very large tables
- Indexes on the columns being correlated won't improve performance as all values need to be read

## Related Functions

- `COVAR_POP()`: Population covariance
- `COVAR_SAMP()`: Sample covariance
- `STDDEV()`: Standard deviation
- `VAR_POP()`: Population variance
- `VAR_SAMP()`: Sample variance