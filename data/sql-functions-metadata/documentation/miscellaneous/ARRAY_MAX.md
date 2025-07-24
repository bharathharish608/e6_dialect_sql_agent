# ARRAY_MAX

Returns the maximum value from an array of elements.

## Syntax

```sql
ARRAY_MAX( <array> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to find the maximum value from. Can contain any comparable data type (numbers, strings, dates, etc.)

## Returns

- **Type**: Element type of array, nullable
- **Description**: Returns the maximum value found in the array
- **NULL Handling**: Returns NULL if the input array is NULL or empty. NULL elements within the array are ignored when determining the maximum

## Usage Notes

- Works with arrays of any comparable type (INTEGER, DECIMAL, VARCHAR, DATE, TIMESTAMP, etc.)
- For string arrays, returns the lexicographically largest string
- For date/timestamp arrays, returns the most recent date/time
- NULL values within the array are ignored
- Returns NULL for empty arrays or NULL input

## Examples

### Example 1: Finding Maximum Values in Numeric Arrays

Sample data in `sales_performance` table:
```
+--------+---------------+-----------------------------------+----------------------------------+
| rep_id | rep_name      | monthly_sales                     | quarterly_bonuses                |
+--------+---------------+-----------------------------------+----------------------------------+
| 1      | Sarah Chen    | [45000,52000,48000,61000,58000] | [2500,3100,2900,3600]           |
| 2      | Mike Johnson  | [38000,41000,39000,43000,47000] | [2200,2400,2300,2600]           |
| 3      | Emily Davis   | [55000,58000,62000,59000,64000] | [3300,3500,3700,3600]           |
| 4      | James Wilson  | [42000,44000,46000,48000,51000] | [2500,2700,2800,3000]           |
| 5      | Lisa Anderson | [49000,53000,50000,54000,57000] | [2900,3200,3000,3300]           |
+--------+---------------+-----------------------------------+----------------------------------+
```

Query:
```sql
SELECT 
    rep_name,
    monthly_sales,
    ARRAY_MAX(monthly_sales) AS best_month_sales,
    quarterly_bonuses,
    ARRAY_MAX(quarterly_bonuses) AS highest_bonus
FROM sales_performance
ORDER BY best_month_sales DESC;
```

Result:
```
+---------------+-----------------------------------+------------------+----------------------------------+---------------+
| rep_name      | monthly_sales                     | best_month_sales | quarterly_bonuses                | highest_bonus |
+---------------+-----------------------------------+------------------+----------------------------------+---------------+
| Emily Davis   | [55000,58000,62000,59000,64000] | 64000            | [3300,3500,3700,3600]           | 3700          |
| Sarah Chen    | [45000,52000,48000,61000,58000] | 61000            | [2500,3100,2900,3600]           | 3600          |
| Lisa Anderson | [49000,53000,50000,54000,57000] | 57000            | [2900,3200,3000,3300]           | 3300          |
| James Wilson  | [42000,44000,46000,48000,51000] | 51000            | [2500,2700,2800,3000]           | 3000          |
| Mike Johnson  | [38000,41000,39000,43000,47000] | 47000            | [2200,2400,2300,2600]           | 2600          |
+---------------+-----------------------------------+------------------+----------------------------------+---------------+
```

This example finds the best monthly sales and highest quarterly bonus for each sales representative.

### Example 2: Working with String Arrays

Sample data in `product_catalog` table:
```
+------------+------------------+----------------------------------------------+------------------------------+
| product_id | product_name     | available_sizes                              | color_options                |
+------------+------------------+----------------------------------------------+------------------------------+
| 1          | Classic T-Shirt  | ['XS','S','M','L','XL','XXL']              | ['Black','White','Navy','Gray'] |
| 2          | Running Shoes    | ['7','8','9','10','11','12']               | ['Blue','Red','Green']       |
| 3          | Winter Jacket    | ['S','M','L','XL']                          | ['Black','Brown','Navy']     |
| 4          | Yoga Mat         | ['Standard','Extra Long']                    | ['Purple','Pink','Blue']     |
| 5          | Baseball Cap     | ['One Size']                                 | ['Red','Blue','Black','White'] |
+------------+------------------+----------------------------------------------+------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    available_sizes,
    ARRAY_MAX(available_sizes) AS largest_size,
    color_options,
    ARRAY_MAX(color_options) AS last_color_alphabetically
FROM product_catalog
ORDER BY product_id;
```

Result:
```
+------------------+----------------------------------------------+--------------+------------------------------+---------------------------+
| product_name     | available_sizes                              | largest_size | color_options                | last_color_alphabetically |
+------------------+----------------------------------------------+--------------+------------------------------+---------------------------+
| Classic T-Shirt  | ['XS','S','M','L','XL','XXL']              | XXL          | ['Black','White','Navy','Gray'] | White                     |
| Running Shoes    | ['7','8','9','10','11','12']               | 9            | ['Blue','Red','Green']       | Red                       |
| Winter Jacket    | ['S','M','L','XL']                          | XL           | ['Black','Brown','Navy']     | Navy                      |
| Yoga Mat         | ['Standard','Extra Long']                    | Standard     | ['Purple','Pink','Blue']     | Purple                    |
| Baseball Cap     | ['One Size']                                 | One Size     | ['Red','Blue','Black','White'] | White                     |
+------------------+----------------------------------------------+--------------+------------------------------+---------------------------+
```

This example shows how ARRAY_MAX works with string arrays, returning lexicographically largest values.

### Example 3: Date and Timestamp Arrays

Sample data in `project_timeline` table:
```
+------------+--------------------+--------------------------------------------------------+----------------------------------------------------+
| project_id | project_name       | milestone_dates                                        | review_timestamps                                  |
+------------+--------------------+--------------------------------------------------------+----------------------------------------------------+
| 1          | Website Upgrade    | ['2024-01-15','2024-02-28','2024-03-20','2024-04-10'] | ['2024-01-15 09:00','2024-02-28 14:30','2024-03-20 11:00'] |
| 2          | Mobile App Launch  | ['2024-02-01','2024-03-15','2024-04-30']             | ['2024-02-01 10:00','2024-03-15 15:45']          |
| 3          | Database Migration | ['2024-01-20','2024-02-10','2024-03-05']             | ['2024-01-20 08:30','2024-02-10 16:00','2024-03-05 13:15'] |
| 4          | API Integration    | ['2024-03-01','2024-04-15','2024-05-20']             | ['2024-03-01 09:30','2024-04-15 14:00']          |
| 5          | Security Audit     | ['2024-02-15','2024-03-30']                           | ['2024-02-15 10:15','2024-03-30 15:30']          |
+------------+--------------------+--------------------------------------------------------+----------------------------------------------------+
```

Query:
```sql
SELECT 
    project_name,
    milestone_dates,
    ARRAY_MAX(milestone_dates) AS final_milestone,
    review_timestamps,
    ARRAY_MAX(review_timestamps) AS last_review
FROM project_timeline
ORDER BY final_milestone DESC;
```

Result:
```
+--------------------+--------------------------------------------------------+-----------------+----------------------------------------------------+---------------------+
| project_name       | milestone_dates                                        | final_milestone | review_timestamps                                  | last_review         |
+--------------------+--------------------------------------------------------+-----------------+----------------------------------------------------+---------------------+
| API Integration    | ['2024-03-01','2024-04-15','2024-05-20']             | 2024-05-20      | ['2024-03-01 09:30','2024-04-15 14:00']          | 2024-04-15 14:00    |
| Mobile App Launch  | ['2024-02-01','2024-03-15','2024-04-30']             | 2024-04-30      | ['2024-02-01 10:00','2024-03-15 15:45']          | 2024-03-15 15:45    |
| Website Upgrade    | ['2024-01-15','2024-02-28','2024-03-20','2024-04-10'] | 2024-04-10      | ['2024-01-15 09:00','2024-02-28 14:30','2024-03-20 11:00'] | 2024-03-20 11:00    |
| Security Audit     | ['2024-02-15','2024-03-30']                           | 2024-03-30      | ['2024-02-15 10:15','2024-03-30 15:30']          | 2024-03-30 15:30    |
| Database Migration | ['2024-01-20','2024-02-10','2024-03-05']             | 2024-03-05      | ['2024-01-20 08:30','2024-02-10 16:00','2024-03-05 13:15'] | 2024-03-05 13:15    |
+--------------------+--------------------------------------------------------+-----------------+----------------------------------------------------+---------------------+
```

This example finds the latest milestone date and most recent review timestamp for each project.

### Example 4: Handling NULL Values and Empty Arrays

Sample data in `sensor_data` table:
```
+----------+--------------+----------------------------------------+-------------------------------+
| sensor_id| location     | temperature_readings                   | humidity_readings             |
+----------+--------------+----------------------------------------+-------------------------------+
| 1        | Warehouse A  | [18.5,19.2,NULL,20.1,19.8,NULL]       | [45,48,52,50,NULL]           |
| 2        | Office B     | [22.3,23.1,22.8,NULL,23.5]            | []                           |
| 3        | Lab C        | NULL                                   | [60,62,58,61]                |
| 4        | Storage D    | []                                     | NULL                         |
| 5        | Server Room  | [19.0,18.5,NULL,NULL,19.3]            | [35,NULL,38,40,NULL]         |
+----------+--------------+----------------------------------------+-------------------------------+
```

Query:
```sql
SELECT 
    location,
    temperature_readings,
    ARRAY_MAX(temperature_readings) AS max_temp,
    humidity_readings,
    ARRAY_MAX(humidity_readings) AS max_humidity,
    CASE 
        WHEN ARRAY_MAX(temperature_readings) > 23 THEN 'High Temp Alert'
        WHEN ARRAY_MAX(temperature_readings) IS NULL THEN 'No Data'
        ELSE 'Normal'
    END AS temp_status
FROM sensor_data
ORDER BY sensor_id;
```

Result:
```
+--------------+----------------------------------------+----------+-------------------------------+--------------+----------------+
| location     | temperature_readings                   | max_temp | humidity_readings             | max_humidity | temp_status    |
+--------------+----------------------------------------+----------+-------------------------------+--------------+----------------+
| Warehouse A  | [18.5,19.2,NULL,20.1,19.8,NULL]       | 20.1     | [45,48,52,50,NULL]           | 52           | Normal         |
| Office B     | [22.3,23.1,22.8,NULL,23.5]            | 23.5     | []                           | NULL         | High Temp Alert|
| Lab C        | NULL                                   | NULL     | [60,62,58,61]                | 62           | No Data        |
| Storage D    | []                                     | NULL     | NULL                         | NULL         | No Data        |
| Server Room  | [19.0,18.5,NULL,NULL,19.3]            | 19.3     | [35,NULL,38,40,NULL]         | 40           | Normal         |
+--------------+----------------------------------------+----------+-------------------------------+--------------+----------------+
```

This example demonstrates how ARRAY_MAX handles NULL values, empty arrays, and NULL arrays.

### Example 5: Complex Analysis with Array Functions

Sample data in `stock_portfolio` table:
```
+------------+---------------+----------------------------------------+----------------------------------------+
| portfolio_id| investor_name | daily_prices                           | daily_volumes                          |
+------------+---------------+----------------------------------------+----------------------------------------+
| 1          | John Smith    | [152.3,154.2,153.8,156.1,155.9,157.2] | [1000000,1200000,950000,1150000,1100000,1300000] |
| 2          | Jane Doe      | [45.6,46.2,44.9,47.1,46.8,48.3]       | [500000,550000,480000,620000,590000,700000] |
| 3          | Bob Johnson   | [89.5,90.2,88.7,91.3,90.8,92.1]       | [750000,800000,720000,850000,820000,900000] |
| 4          | Alice Chen    | [234.1,235.8,233.5,237.2,236.9,238.5] | [2000000,2100000,1950000,2200000,2150000,2300000] |
| 5          | Tom Wilson    | [67.8,68.4,66.9,69.2,68.7,70.1]       | [600000,650000,580000,700000,680000,750000] |
+------------+---------------+----------------------------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    investor_name,
    daily_prices,
    ARRAY_MAX(daily_prices) AS peak_price,
    ARRAY_MIN(daily_prices) AS low_price,
    ROUND(ARRAY_MAX(daily_prices) - ARRAY_MIN(daily_prices), 2) AS price_range,
    ARRAY_MAX(daily_volumes) AS max_volume,
    ROUND((ARRAY_MAX(daily_prices) - ARRAY_MIN(daily_prices)) / ARRAY_MIN(daily_prices) * 100, 2) AS volatility_pct
FROM stock_portfolio
ORDER BY volatility_pct DESC;
```

Result:
```
+---------------+----------------------------------------+------------+-----------+-------------+------------+----------------+
| investor_name | daily_prices                           | peak_price | low_price | price_range | max_volume | volatility_pct |
+---------------+----------------------------------------+------------+-----------+-------------+------------+----------------+
| Jane Doe      | [45.6,46.2,44.9,47.1,46.8,48.3]       | 48.3       | 44.9      | 3.40        | 700000     | 7.57           |
| Tom Wilson    | [67.8,68.4,66.9,69.2,68.7,70.1]       | 70.1       | 66.9      | 3.20        | 750000     | 4.78           |
| Bob Johnson   | [89.5,90.2,88.7,91.3,90.8,92.1]       | 92.1       | 88.7      | 3.40        | 900000     | 3.83           |
| John Smith    | [152.3,154.2,153.8,156.1,155.9,157.2] | 157.2      | 152.3     | 4.90        | 1300000    | 3.22           |
| Alice Chen    | [234.1,235.8,233.5,237.2,236.9,238.5] | 238.5      | 233.5     | 5.00        | 2300000    | 2.14           |
+---------------+----------------------------------------+------------+-----------+-------------+------------+----------------+
```

This example combines ARRAY_MAX with ARRAY_MIN to calculate price ranges and volatility percentages.