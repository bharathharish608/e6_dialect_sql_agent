# ARRAY_COMPACT

Removes null values from an array, returning a new array containing only non-null elements. This function is essential for data cleaning, filtering incomplete data, and preparing arrays for aggregation or analysis.

## Syntax

```sql
ARRAY_COMPACT( <array> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array from which to remove null values. Can be any array type (INTEGER[], VARCHAR[], etc.)

## Returns

- **Type**: ARRAY (same element type as input array)
- **Description**: Returns a new array containing all non-null elements from the input array, preserving their original order.
- **NULL Handling**: Returns NULL if the input array is NULL. Returns an empty array if all elements are NULL.

## Usage Notes

- Preserves the original order of non-null elements
- Does not remove duplicate values, only NULL values
- Useful for data cleaning before aggregations
- Can be combined with other array functions for complex transformations
- Memory efficient for sparse arrays with many NULLs

## Examples

### Example 1: Sales Performance Tracking with Incomplete Data

Sample data in `monthly_sales` table:
```
+-------------+------------------+---------------------------------------------------------------------------------+
| salesperson | name             | monthly_revenues                                                                |
+-------------+------------------+---------------------------------------------------------------------------------+
| 1           | John Smith       | [45000, 52000, NULL, 48000, 51000, NULL, 49000, 53000, 47000, NULL, 50000, 55000] |
| 2           | Sarah Johnson    | [38000, 41000, 39000, NULL, NULL, 42000, 40000, 43000, NULL, 44000, 41000, 45000] |
| 3           | Mike Chen        | [NULL, NULL, 35000, 37000, 36000, 38000, NULL, 39000, 40000, 41000, NULL, 42000] |
| 4           | Lisa Williams    | [55000, 58000, 57000, 59000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000] |
| 5           | Tom Davis        | [30000, NULL, NULL, NULL, 32000, 33000, 34000, NULL, 35000, 36000, NULL, 37000] |
+-------------+------------------+---------------------------------------------------------------------------------+
```

Query:
```sql
SELECT 
    salesperson,
    name,
    monthly_revenues,
    ARRAY_COMPACT(monthly_revenues) AS actual_sales_months,
    CARDINALITY(monthly_revenues) AS total_months,
    CARDINALITY(ARRAY_COMPACT(monthly_revenues)) AS months_with_sales,
    CARDINALITY(monthly_revenues) - CARDINALITY(ARRAY_COMPACT(monthly_revenues)) AS months_no_sales,
    ROUND(AVG(ARRAY_COMPACT(monthly_revenues)[i]), 2) AS avg_monthly_sales
FROM monthly_sales
CROSS JOIN UNNEST(SEQUENCE(1, CARDINALITY(ARRAY_COMPACT(monthly_revenues)))) AS t(i)
GROUP BY salesperson, name, monthly_revenues
ORDER BY salesperson;
```

Result:
```
+-------------+------------------+---------------------------------------------------------------------------------+-----------------------------------------------------------+-------------+------------------+----------------+-------------------+
| salesperson | name             | monthly_revenues                                                                | actual_sales_months                                       | total_months| months_with_sales| months_no_sales| avg_monthly_sales |
+-------------+------------------+---------------------------------------------------------------------------------+-----------------------------------------------------------+-------------+------------------+----------------+-------------------+
| 1           | John Smith       | [45000, 52000, NULL, 48000, 51000, NULL, 49000, 53000, 47000, NULL, 50000, 55000] | [45000, 52000, 48000, 51000, 49000, 53000, 47000, 50000, 55000] | 12    | 9                | 3              | 50000.00          |
| 2           | Sarah Johnson    | [38000, 41000, 39000, NULL, NULL, 42000, 40000, 43000, NULL, 44000, 41000, 45000] | [38000, 41000, 39000, 42000, 40000, 43000, 44000, 41000, 45000] | 12    | 9                | 3              | 41444.44          |
| 3           | Mike Chen        | [NULL, NULL, 35000, 37000, 36000, 38000, NULL, 39000, 40000, 41000, NULL, 42000] | [35000, 37000, 36000, 38000, 39000, 40000, 41000, 42000]      | 12    | 8                | 4              | 38500.00          |
| 4           | Lisa Williams    | [55000, 58000, 57000, 59000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000] | [55000, 58000, 57000, 59000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000] | 12 | 12            | 0              | 61000.00          |
| 5           | Tom Davis        | [30000, NULL, NULL, NULL, 32000, 33000, 34000, NULL, 35000, 36000, NULL, 37000] | [30000, 32000, 33000, 34000, 35000, 36000, 37000]             | 12    | 7                | 5              | 33857.14          |
+-------------+------------------+---------------------------------------------------------------------------------+-----------------------------------------------------------+-------------+------------------+----------------+-------------------+
```

This example analyzes sales performance by removing months with no sales data (NULL values) to calculate accurate averages.

### Example 2: Survey Response Analysis

Sample data in `survey_responses` table:
```
+-------------+------------------+----------------------------------------------------------------------+
| respondent  | demographic      | satisfaction_scores                                                  |
+-------------+------------------+----------------------------------------------------------------------+
| 1           | Age 18-25        | [8, 9, NULL, 7, NULL, 8, 9, NULL, 8, 7]                           |
| 2           | Age 26-35        | [NULL, 6, 7, 8, NULL, 7, 6, NULL, 8, 9]                           |
| 3           | Age 36-45        | [9, 10, 9, NULL, 10, NULL, 9, 10, NULL, 9]                        |
| 4           | Age 46-55        | [7, NULL, NULL, 8, 7, 8, NULL, 7, 8, NULL]                        |
| 5           | Age 56+          | [10, 9, 10, 10, NULL, 9, 10, 9, NULL, 10]                         |
+-------------+------------------+----------------------------------------------------------------------+
```

Query:
```sql
WITH cleaned_responses AS (
    SELECT 
        respondent,
        demographic,
        satisfaction_scores,
        ARRAY_COMPACT(satisfaction_scores) AS valid_scores,
        CARDINALITY(ARRAY_COMPACT(satisfaction_scores)) AS response_count
    FROM survey_responses
)
SELECT 
    demographic,
    COUNT(*) AS respondents,
    SUM(response_count) AS total_responses,
    ROUND(AVG(response_count), 1) AS avg_responses_per_person,
    ROUND(AVG(CARDINALITY(valid_scores)), 1) AS avg_valid_scores,
    ROUND(AVG(REDUCE(valid_scores, 0, (s, x) -> s + x, s -> s * 1.0 / CARDINALITY(valid_scores))), 2) AS avg_satisfaction
FROM cleaned_responses
GROUP BY demographic
ORDER BY demographic;
```

Result:
```
+------------------+-------------+-----------------+--------------------------+------------------+-------------------+
| demographic      | respondents | total_responses | avg_responses_per_person | avg_valid_scores | avg_satisfaction  |
+------------------+-------------+-----------------+--------------------------+------------------+-------------------+
| Age 18-25        | 1           | 7               | 7.0                      | 7.0              | 8.00              |
| Age 26-35        | 1           | 6               | 6.0                      | 6.0              | 7.17              |
| Age 36-45        | 1           | 7               | 7.0                      | 7.0              | 9.43              |
| Age 46-55        | 1           | 6               | 6.0                      | 6.0              | 7.50              |
| Age 56+          | 1           | 8               | 8.0                      | 8.0              | 9.63              |
+------------------+-------------+-----------------+--------------------------+------------------+-------------------+
```

This example cleans survey data by removing NULL responses before calculating satisfaction metrics.

### Example 3: Product Review Aggregation

Sample data in `product_reviews` table:
```
+------------+-------------------------+------------------------------------------------------------------------+
| product_id | product_name           | review_ratings                                                         |
+------------+-------------------------+------------------------------------------------------------------------+
| 101        | Wireless Headphones    | [5, 4, NULL, 5, 3, NULL, 4, 5, NULL, 4, 5, 4, NULL, 5]              |
| 102        | Smart Watch            | [4, 5, 5, NULL, NULL, 4, 5, 4, NULL, 5, 4, 5]                       |
| 103        | Bluetooth Speaker      | [3, NULL, 4, 3, NULL, 2, 4, NULL, 3, 4, NULL, 3]                    |
| 104        | USB-C Hub              | [5, 5, 5, 5, NULL, 5, 5, NULL, 5, 5, 5, NULL, 5]                    |
| 105        | Laptop Stand           | [NULL, NULL, 4, 5, 4, NULL, 5, 4, NULL, 4, 5, NULL]                 |
+------------+-------------------------+------------------------------------------------------------------------+
```

Query:
```sql
SELECT 
    product_id,
    product_name,
    review_ratings AS all_ratings,
    ARRAY_COMPACT(review_ratings) AS valid_ratings,
    CARDINALITY(review_ratings) AS total_rating_slots,
    CARDINALITY(ARRAY_COMPACT(review_ratings)) AS actual_ratings,
    ROUND(CAST(CARDINALITY(ARRAY_COMPACT(review_ratings)) AS FLOAT) / CARDINALITY(review_ratings) * 100, 1) AS response_rate,
    ROUND(REDUCE(ARRAY_COMPACT(review_ratings), 0, (s, x) -> s + x, s -> s * 1.0 / CARDINALITY(ARRAY_COMPACT(review_ratings))), 2) AS avg_rating,
    ARRAY_MIN(ARRAY_COMPACT(review_ratings)) AS min_rating,
    ARRAY_MAX(ARRAY_COMPACT(review_ratings)) AS max_rating
FROM product_reviews
ORDER BY avg_rating DESC;
```

Result:
```
+------------+-------------------------+------------------------------------------------------------------------+--------------------------------+-------------------+----------------+---------------+-----------+------------+------------+
| product_id | product_name           | all_ratings                                                            | valid_ratings                  | total_rating_slots| actual_ratings | response_rate | avg_rating| min_rating | max_rating |
+------------+-------------------------+------------------------------------------------------------------------+--------------------------------+-------------------+----------------+---------------+-----------+------------+------------+
| 104        | USB-C Hub              | [5, 5, 5, 5, NULL, 5, 5, NULL, 5, 5, 5, NULL, 5]                    | [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]| 12                | 10             | 83.3          | 5.00      | 5          | 5          |
| 101        | Wireless Headphones    | [5, 4, NULL, 5, 3, NULL, 4, 5, NULL, 4, 5, 4, NULL, 5]              | [5, 4, 5, 3, 4, 5, 4, 5, 4, 5]| 14                | 10             | 71.4          | 4.40      | 3          | 5          |
| 102        | Smart Watch            | [4, 5, 5, NULL, NULL, 4, 5, 4, NULL, 5, 4, 5]                       | [4, 5, 5, 4, 5, 4, 5, 4, 5]   | 12                | 9              | 75.0          | 4.56      | 4          | 5          |
| 105        | Laptop Stand           | [NULL, NULL, 4, 5, 4, NULL, 5, 4, NULL, 4, 5, NULL]                 | [4, 5, 4, 5, 4, 4, 5]          | 12                | 7              | 58.3          | 4.43      | 4          | 5          |
| 103        | Bluetooth Speaker      | [3, NULL, 4, 3, NULL, 2, 4, NULL, 3, 4, NULL, 3]                    | [3, 4, 3, 2, 4, 3, 4, 3]       | 12                | 8              | 66.7          | 3.25      | 2          | 4          |
+------------+-------------------------+------------------------------------------------------------------------+--------------------------------+-------------------+----------------+---------------+-----------+------------+------------+
```

This example processes product reviews by removing NULL values to calculate accurate rating statistics.

### Example 4: Time Series Data with Missing Values

Sample data in `sensor_readings` table:
```
+-----------+------------------+--------------------------------------------------------------------------------+
| sensor_id | location         | hourly_temperatures                                                            |
+-----------+------------------+--------------------------------------------------------------------------------+
| 1         | Warehouse A      | [22.5, 23.1, NULL, 24.2, NULL, 25.0, 24.8, NULL, 23.9, 23.5, 22.8, NULL]    |
| 2         | Warehouse B      | [18.2, 18.5, 18.8, NULL, 19.2, 19.5, NULL, 20.1, 20.5, NULL, 19.8, 19.2]    |
| 3         | Office Floor 1   | [21.0, NULL, 21.5, 22.0, NULL, 22.5, 23.0, 23.2, NULL, 22.8, 22.5, NULL]    |
| 4         | Server Room      | [16.0, 16.2, 16.1, 16.3, NULL, 16.5, 16.4, NULL, 16.2, 16.3, NULL, 16.1]    |
| 5         | Loading Dock     | [NULL, 25.5, 26.2, NULL, 27.8, 28.5, NULL, 27.2, 26.8, NULL, 25.2, 24.5]    |
+-----------+------------------+--------------------------------------------------------------------------------+
```

Query:
```sql
WITH temperature_analysis AS (
    SELECT 
        sensor_id,
        location,
        hourly_temperatures,
        ARRAY_COMPACT(hourly_temperatures) AS valid_readings,
        CARDINALITY(hourly_temperatures) AS total_hours,
        CARDINALITY(ARRAY_COMPACT(hourly_temperatures)) AS hours_with_data,
        ROUND(CAST(CARDINALITY(ARRAY_COMPACT(hourly_temperatures)) AS FLOAT) / CARDINALITY(hourly_temperatures) * 100, 1) AS data_availability
    FROM sensor_readings
)
SELECT 
    sensor_id,
    location,
    valid_readings,
    hours_with_data || '/' || total_hours AS data_coverage,
    data_availability || '%' AS availability_pct,
    ROUND(REDUCE(valid_readings, 0, (s, x) -> s + x, s -> s / CARDINALITY(valid_readings)), 1) AS avg_temp,
    ROUND(ARRAY_MIN(valid_readings), 1) AS min_temp,
    ROUND(ARRAY_MAX(valid_readings), 1) AS max_temp,
    ROUND(ARRAY_MAX(valid_readings) - ARRAY_MIN(valid_readings), 1) AS temp_range
FROM temperature_analysis
ORDER BY sensor_id;
```

Result:
```
+-----------+------------------+------------------------------------------------------------------------+---------------+------------------+----------+----------+----------+------------+
| sensor_id | location         | valid_readings                                                         | data_coverage | availability_pct | avg_temp | min_temp | max_temp | temp_range |
+-----------+------------------+------------------------------------------------------------------------+---------------+------------------+----------+----------+----------+------------+
| 1         | Warehouse A      | [22.5, 23.1, 24.2, 25.0, 24.8, 23.9, 23.5, 22.8]                    | 8/12          | 66.7%            | 23.7     | 22.5     | 25.0     | 2.5        |
| 2         | Warehouse B      | [18.2, 18.5, 18.8, 19.2, 19.5, 20.1, 20.5, 19.8, 19.2]              | 9/12          | 75.0%            | 19.3     | 18.2     | 20.5     | 2.3        |
| 3         | Office Floor 1   | [21.0, 21.5, 22.0, 22.5, 23.0, 23.2, 22.8, 22.5]                    | 8/12          | 66.7%            | 22.3     | 21.0     | 23.2     | 2.2        |
| 4         | Server Room      | [16.0, 16.2, 16.1, 16.3, 16.5, 16.4, 16.2, 16.3, 16.1]              | 9/12          | 75.0%            | 16.2     | 16.0     | 16.5     | 0.5        |
| 5         | Loading Dock     | [25.5, 26.2, 27.8, 28.5, 27.2, 26.8, 25.2, 24.5]                    | 8/12          | 66.7%            | 26.5     | 24.5     | 28.5     | 4.0        |
+-----------+------------------+------------------------------------------------------------------------+---------------+------------------+----------+----------+----------+------------+
```

This example analyzes temperature sensor data by removing missing readings to calculate statistics and data availability.

### Example 5: Employee Skill Assessment Tracking

Sample data in `skill_assessments` table:
```
+-------------+------------------+------------------------------------------------------------------------------+
| employee_id | employee_name    | quarterly_scores                                                             |
+-------------+------------------+------------------------------------------------------------------------------+
| 1           | Alice Johnson    | [85, 88, NULL, 90, 92, NULL, 94, 95, NULL, 96, 97, 98]                     |
| 2           | Bob Smith        | [75, NULL, 78, 80, NULL, 82, 84, NULL, 85, 87, NULL, 88]                   |
| 3           | Carol Williams   | [NULL, NULL, 70, 72, 75, 77, NULL, 80, 82, 85, NULL, 87]                   |
| 4           | David Chen       | [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, NULL]                        |
| 5           | Emma Davis       | [60, 65, NULL, NULL, 70, 75, NULL, 78, 82, NULL, 85, 88]                   |
+-------------+------------------+------------------------------------------------------------------------------+
```

Query:
```sql
WITH assessment_summary AS (
    SELECT 
        employee_id,
        employee_name,
        quarterly_scores,
        ARRAY_COMPACT(quarterly_scores) AS completed_assessments,
        CARDINALITY(ARRAY_COMPACT(quarterly_scores)) AS assessments_taken,
        12 - CARDINALITY(ARRAY_COMPACT(quarterly_scores)) AS assessments_missed
    FROM skill_assessments
)
SELECT 
    employee_id,
    employee_name,
    completed_assessments,
    assessments_taken || '/12' AS completion_rate,
    assessments_missed,
    completed_assessments[1] AS first_score,
    completed_assessments[CARDINALITY(completed_assessments)] AS latest_score,
    completed_assessments[CARDINALITY(completed_assessments)] - completed_assessments[1] AS improvement,
    ROUND(REDUCE(completed_assessments, 0, (s, x) -> s + x, s -> s * 1.0 / CARDINALITY(completed_assessments)), 1) AS avg_score,
    CASE 
        WHEN completed_assessments[CARDINALITY(completed_assessments)] > completed_assessments[1] THEN 'Improving'
        WHEN completed_assessments[CARDINALITY(completed_assessments)] = completed_assessments[1] THEN 'Stable'
        ELSE 'Declining'
    END AS trend
FROM assessment_summary
ORDER BY improvement DESC;
```

Result:
```
+-------------+------------------+------------------------------------------------------------------------------+-----------------+--------------------+-------------+--------------+-------------+-----------+-----------+
| employee_id | employee_name    | completed_assessments                                                        | completion_rate | assessments_missed | first_score | latest_score | improvement | avg_score | trend     |
+-------------+------------------+------------------------------------------------------------------------------+-----------------+--------------------+-------------+--------------+-------------+-----------+-----------+
| 5           | Emma Davis       | [60, 65, 70, 75, 78, 82, 85, 88]                                           | 8/12            | 4                  | 60          | 88           | 28          | 75.4      | Improving |
| 3           | Carol Williams   | [70, 72, 75, 77, 80, 82, 85, 87]                                           | 8/12            | 4                  | 70          | 87           | 17          | 78.5      | Improving |
| 1           | Alice Johnson    | [85, 88, 90, 92, 94, 95, 96, 97, 98]                                       | 9/12            | 3                  | 85          | 98           | 13          | 92.8      | Improving |
| 2           | Bob Smith        | [75, 78, 80, 82, 84, 85, 87, 88]                                           | 8/12            | 4                  | 75          | 88           | 13          | 82.4      | Improving |
| 4           | David Chen       | [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]                              | 11/12           | 1                  | 90          | 100          | 10          | 95.0      | Improving |
+-------------+------------------+------------------------------------------------------------------------------+-----------------+--------------------+-------------+--------------+-------------+-----------+-----------+
```

This example tracks employee skill progression by removing quarters where assessments were not completed, showing improvement trends over time.