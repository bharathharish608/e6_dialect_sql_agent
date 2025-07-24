# ARRAY_MIN

Returns the minimum value from an array of elements.

## Syntax

```sql
ARRAY_MIN( <array> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to find the minimum value from. Can contain any comparable data type (numbers, strings, dates, etc.)

## Returns

- **Type**: Element type of array, nullable
- **Description**: Returns the minimum value found in the array
- **NULL Handling**: Returns NULL if the input array is NULL or empty. NULL elements within the array are ignored when determining the minimum

## Usage Notes

- Works with arrays of any comparable type (INTEGER, DECIMAL, VARCHAR, DATE, TIMESTAMP, etc.)
- For string arrays, returns the lexicographically smallest string
- For date/timestamp arrays, returns the earliest date/time
- NULL values within the array are ignored
- Returns NULL for empty arrays or NULL input

## Examples

### Example 1: Finding Minimum Values in Numeric Arrays

Sample data in `temperature_logs` table:
```
+----------+--------------+----------------------------------------+-------------------------------------+
| sensor_id| location     | daily_temps_celsius                    | daily_temps_fahrenheit              |
+----------+--------------+----------------------------------------+-------------------------------------+
| 1        | Freezer A    | [-18.5,-17.2,-19.1,-18.8,-17.5]      | [-1.3,1.04,-2.38,-1.84,0.5]        |
| 2        | Cooler B     | [2.3,3.1,1.8,2.5,3.2]                 | [36.14,37.58,35.24,36.5,37.76]     |
| 3        | Room C       | [20.5,21.2,19.8,22.1,20.9]            | [68.9,70.16,67.64,71.78,69.62]     |
| 4        | Warehouse D  | [15.2,16.8,14.5,17.2,15.9]            | [59.36,62.24,58.1,62.96,60.62]     |
| 5        | Outdoor E    | [28.5,30.2,27.8,31.5,29.2]            | [83.3,86.36,82.04,88.7,84.56]      |
+----------+--------------+----------------------------------------+-------------------------------------+
```

Query:
```sql
SELECT 
    location,
    daily_temps_celsius,
    ARRAY_MIN(daily_temps_celsius) AS min_celsius,
    daily_temps_fahrenheit,
    ARRAY_MIN(daily_temps_fahrenheit) AS min_fahrenheit,
    CASE 
        WHEN ARRAY_MIN(daily_temps_celsius) < 0 THEN 'Below Freezing'
        WHEN ARRAY_MIN(daily_temps_celsius) < 10 THEN 'Cold'
        ELSE 'Normal'
    END AS temp_category
FROM temperature_logs
ORDER BY min_celsius;
```

Result:
```
+--------------+----------------------------------------+-------------+-------------------------------------+----------------+----------------+
| location     | daily_temps_celsius                    | min_celsius | daily_temps_fahrenheit              | min_fahrenheit | temp_category  |
+--------------+----------------------------------------+-------------+-------------------------------------+----------------+----------------+
| Freezer A    | [-18.5,-17.2,-19.1,-18.8,-17.5]      | -19.1       | [-1.3,1.04,-2.38,-1.84,0.5]        | -2.38          | Below Freezing |
| Cooler B     | [2.3,3.1,1.8,2.5,3.2]                 | 1.8         | [36.14,37.58,35.24,36.5,37.76]     | 35.24          | Cold           |
| Warehouse D  | [15.2,16.8,14.5,17.2,15.9]            | 14.5        | [59.36,62.24,58.1,62.96,60.62]     | 58.1           | Normal         |
| Room C       | [20.5,21.2,19.8,22.1,20.9]            | 19.8        | [68.9,70.16,67.64,71.78,69.62]     | 67.64          | Normal         |
| Outdoor E    | [28.5,30.2,27.8,31.5,29.2]            | 27.8        | [83.3,86.36,82.04,88.7,84.56]      | 82.04          | Normal         |
+--------------+----------------------------------------+-------------+-------------------------------------+----------------+----------------+
```

This example finds minimum temperatures and categorizes locations based on their coldest readings.

### Example 2: Working with Date Arrays

Sample data in `employee_attendance` table:
```
+------------+---------------+----------------------------------------------------+----------------------------------------+
| emp_id     | employee_name | clock_in_dates                                     | vacation_dates                         |
+------------+---------------+----------------------------------------------------+----------------------------------------+
| 1          | Sarah Miller  | ['2024-03-01','2024-03-04','2024-03-05','2024-03-06'] | ['2024-01-15','2024-07-20','2024-12-25'] |
| 2          | John Davis    | ['2024-03-04','2024-03-05','2024-03-06','2024-03-07'] | ['2024-02-10','2024-08-15']          |
| 3          | Emma Wilson   | ['2024-03-01','2024-03-02','2024-03-05','2024-03-07'] | ['2024-04-01','2024-09-30']          |
| 4          | Michael Brown | ['2024-03-02','2024-03-04','2024-03-06','2024-03-07'] | ['2024-05-20','2024-11-28']          |
| 5          | Lisa Johnson  | ['2024-03-01','2024-03-02','2024-03-03','2024-03-04'] | ['2024-06-15','2024-10-10']          |
+------------+---------------+----------------------------------------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    employee_name,
    clock_in_dates,
    ARRAY_MIN(clock_in_dates) AS first_day_worked,
    ARRAY_MAX(clock_in_dates) AS last_day_worked,
    vacation_dates,
    ARRAY_MIN(vacation_dates) AS first_vacation,
    DATEDIFF(ARRAY_MAX(clock_in_dates), ARRAY_MIN(clock_in_dates)) AS days_span
FROM employee_attendance
ORDER BY first_day_worked;
```

Result:
```
+---------------+----------------------------------------------------+------------------+-----------------+----------------------------------------+----------------+-----------+
| employee_name | clock_in_dates                                     | first_day_worked | last_day_worked | vacation_dates                         | first_vacation | days_span |
+---------------+----------------------------------------------------+------------------+-----------------+----------------------------------------+----------------+-----------+
| Sarah Miller  | ['2024-03-01','2024-03-04','2024-03-05','2024-03-06'] | 2024-03-01       | 2024-03-06      | ['2024-01-15','2024-07-20','2024-12-25'] | 2024-01-15     | 5         |
| Emma Wilson   | ['2024-03-01','2024-03-02','2024-03-05','2024-03-07'] | 2024-03-01       | 2024-03-07      | ['2024-04-01','2024-09-30']          | 2024-04-01     | 6         |
| Lisa Johnson  | ['2024-03-01','2024-03-02','2024-03-03','2024-03-04'] | 2024-03-01       | 2024-03-04      | ['2024-06-15','2024-10-10']          | 2024-06-15     | 3         |
| Michael Brown | ['2024-03-02','2024-03-04','2024-03-06','2024-03-07'] | 2024-03-02       | 2024-03-07      | ['2024-05-20','2024-11-28']          | 2024-05-20     | 5         |
| John Davis    | ['2024-03-04','2024-03-05','2024-03-06','2024-03-07'] | 2024-03-04       | 2024-03-07      | ['2024-02-10','2024-08-15']          | 2024-02-10     | 3         |
+---------------+----------------------------------------------------+------------------+-----------------+----------------------------------------+----------------+-----------+
```

This example finds the earliest clock-in date and vacation date for each employee.

### Example 3: String Array Analysis

Sample data in `product_reviews` table:
```
+------------+------------------+--------------------------------------------+----------------------------------+
| product_id | product_name     | review_ratings                             | review_keywords                  |
+------------+------------------+--------------------------------------------+----------------------------------+
| 1          | Laptop Pro       | ['Excellent','Good','Fair','Excellent']   | ['fast','reliable','expensive'] |
| 2          | Budget Mouse     | ['Poor','Fair','Good','Fair']             | ['cheap','basic','adequate']    |
| 3          | Gaming Keyboard  | ['Excellent','Excellent','Good','Excellent'] | ['responsive','quality','rgb']   |
| 4          | Office Chair     | ['Good','Fair','Good','Excellent']        | ['comfortable','sturdy','pricey'] |
| 5          | Webcam HD        | ['Fair','Poor','Fair','Good']             | ['average','blurry','okay']     |
+------------+------------------+--------------------------------------------+----------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    review_ratings,
    ARRAY_MIN(review_ratings) AS worst_rating,
    ARRAY_MAX(review_ratings) AS best_rating,
    review_keywords,
    ARRAY_MIN(review_keywords) AS first_keyword_alpha,
    CASE 
        WHEN ARRAY_MIN(review_ratings) = 'Poor' THEN 'Needs Improvement'
        WHEN ARRAY_MIN(review_ratings) = 'Fair' THEN 'Mixed Reviews'
        ELSE 'Generally Positive'
    END AS review_summary
FROM product_reviews
ORDER BY product_id;
```

Result:
```
+------------------+--------------------------------------------+--------------+-------------+----------------------------------+---------------------+-------------------+
| product_name     | review_ratings                             | worst_rating | best_rating | review_keywords                  | first_keyword_alpha | review_summary    |
+------------------+--------------------------------------------+--------------+-------------+----------------------------------+---------------------+-------------------+
| Laptop Pro       | ['Excellent','Good','Fair','Excellent']   | Excellent    | Good        | ['fast','reliable','expensive'] | expensive           | Generally Positive|
| Budget Mouse     | ['Poor','Fair','Good','Fair']             | Fair         | Poor        | ['cheap','basic','adequate']    | adequate            | Mixed Reviews     |
| Gaming Keyboard  | ['Excellent','Excellent','Good','Excellent'] | Excellent    | Good        | ['responsive','quality','rgb']   | quality             | Generally Positive|
| Office Chair     | ['Good','Fair','Good','Excellent']        | Excellent    | Good        | ['comfortable','sturdy','pricey'] | comfortable         | Generally Positive|
| Webcam HD        | ['Fair','Poor','Fair','Good']             | Fair         | Poor        | ['average','blurry','okay']     | average             | Mixed Reviews     |
+------------------+--------------------------------------------+--------------+-------------+----------------------------------+---------------------+-------------------+
```

This example analyzes product reviews using string arrays to find worst ratings and alphabetically first keywords.

### Example 4: Handling NULL Values and Empty Arrays

Sample data in `inventory_tracking` table:
```
+------------+------------------+----------------------------------------+-------------------------------+
| item_id    | item_name        | daily_stock_levels                     | reorder_points                |
+------------+------------------+----------------------------------------+-------------------------------+
| 1          | Widget A         | [100,95,NULL,88,92,NULL,85]           | [50,45,40]                   |
| 2          | Gadget B         | [50,45,42,38,NULL,35]                 | []                           |
| 3          | Tool C           | NULL                                   | [25,20,15]                   |
| 4          | Part D           | []                                     | NULL                         |
| 5          | Component E      | [200,NULL,180,NULL,165,170]           | [100,NULL,80,90]             |
+------------+------------------+----------------------------------------+-------------------------------+
```

Query:
```sql
SELECT 
    item_name,
    daily_stock_levels,
    ARRAY_MIN(daily_stock_levels) AS min_stock,
    reorder_points,
    ARRAY_MIN(reorder_points) AS min_reorder_point,
    CASE 
        WHEN ARRAY_MIN(daily_stock_levels) IS NULL THEN 'No Stock Data'
        WHEN ARRAY_MIN(daily_stock_levels) < COALESCE(ARRAY_MIN(reorder_points), 0) THEN 'Below Reorder Point'
        ELSE 'Adequate Stock'
    END AS stock_status
FROM inventory_tracking
ORDER BY item_id;
```

Result:
```
+------------------+----------------------------------------+-----------+-------------------------------+-------------------+---------------------+
| item_name        | daily_stock_levels                     | min_stock | reorder_points                | min_reorder_point | stock_status        |
+------------------+----------------------------------------+-----------+-------------------------------+-------------------+---------------------+
| Widget A         | [100,95,NULL,88,92,NULL,85]           | 85        | [50,45,40]                   | 40                | Adequate Stock      |
| Gadget B         | [50,45,42,38,NULL,35]                 | 35        | []                           | NULL              | Adequate Stock      |
| Tool C           | NULL                                   | NULL      | [25,20,15]                   | 15                | No Stock Data       |
| Part D           | []                                     | NULL      | NULL                         | NULL              | No Stock Data       |
| Component E      | [200,NULL,180,NULL,165,170]           | 165       | [100,NULL,80,90]             | 80                | Adequate Stock      |
+------------------+----------------------------------------+-----------+-------------------------------+-------------------+---------------------+
```

This example demonstrates how ARRAY_MIN handles NULL values, empty arrays, and NULL arrays in inventory management.

### Example 5: Complex Financial Analysis

Sample data in `investment_funds` table:
```
+----------+----------------+----------------------------------------+----------------------------------------+
| fund_id  | fund_name      | daily_nav_values                       | expense_ratios_pct                     |
+----------+----------------+----------------------------------------+----------------------------------------+
| 1        | Growth Fund A  | [125.50,124.75,123.90,126.20,125.80] | [0.75,0.72,0.71,0.73,0.70]            |
| 2        | Value Fund B   | [98.20,97.50,96.80,98.90,98.40]      | [0.85,0.83,0.82,0.84,0.81]            |
| 3        | Bond Fund C    | [105.10,104.90,104.70,105.20,105.00] | [0.45,0.44,0.43,0.45,0.42]            |
| 4        | Tech Fund D    | [156.30,154.20,152.10,157.50,156.90] | [1.20,1.18,1.15,1.19,1.16]            |
| 5        | Mixed Fund E   | [112.40,111.80,111.20,113.10,112.70] | [0.95,0.93,0.92,0.94,0.91]            |
+----------+----------------+----------------------------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    fund_name,
    daily_nav_values,
    ARRAY_MIN(daily_nav_values) AS min_nav,
    ARRAY_MAX(daily_nav_values) AS max_nav,
    ROUND(ARRAY_MAX(daily_nav_values) - ARRAY_MIN(daily_nav_values), 2) AS nav_range,
    expense_ratios_pct,
    ARRAY_MIN(expense_ratios_pct) AS lowest_expense_ratio,
    ROUND(((ARRAY_MAX(daily_nav_values) - ARRAY_MIN(daily_nav_values)) / ARRAY_MIN(daily_nav_values)) * 100, 2) AS volatility_pct
FROM investment_funds
ORDER BY volatility_pct DESC;
```

Result:
```
+----------------+----------------------------------------+---------+---------+-----------+----------------------------------------+----------------------+----------------+
| fund_name      | daily_nav_values                       | min_nav | max_nav | nav_range | expense_ratios_pct                     | lowest_expense_ratio | volatility_pct |
+----------------+----------------------------------------+---------+---------+-----------+----------------------------------------+----------------------+----------------+
| Tech Fund D    | [156.30,154.20,152.10,157.50,156.90] | 152.10  | 157.50  | 5.40      | [1.20,1.18,1.15,1.19,1.16]            | 1.15                 | 3.55           |
| Growth Fund A  | [125.50,124.75,123.90,126.20,125.80] | 123.90  | 126.20  | 2.30      | [0.75,0.72,0.71,0.73,0.70]            | 0.70                 | 1.86           |
| Value Fund B   | [98.20,97.50,96.80,98.90,98.40]      | 96.80   | 98.90   | 2.10      | [0.85,0.83,0.82,0.84,0.81]            | 0.81                 | 2.17           |
| Mixed Fund E   | [112.40,111.80,111.20,113.10,112.70] | 111.20  | 113.10  | 1.90      | [0.95,0.93,0.92,0.94,0.91]            | 0.91                 | 1.71           |
| Bond Fund C    | [105.10,104.90,104.70,105.20,105.00] | 104.70  | 105.20  | 0.50      | [0.45,0.44,0.43,0.45,0.42]            | 0.42                 | 0.48           |
+----------------+----------------------------------------+---------+---------+-----------+----------------------------------------+----------------------+----------------+
```

This example combines ARRAY_MIN with ARRAY_MAX to analyze fund performance and calculate volatility metrics.