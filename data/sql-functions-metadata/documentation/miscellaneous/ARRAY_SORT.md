# ARRAY_SORT

Sorts the elements of an array in ascending or descending order.

## Syntax

```sql
ARRAY_SORT( <array> [ , <ascending> [ , <nulls_first> ] ] )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to sort

### ascending
- **Type**: BOOLEAN
- **Required**: No
- **Default**: TRUE
- **Description**: If TRUE, sorts in ascending order. If FALSE, sorts in descending order

### nulls_first
- **Type**: BOOLEAN
- **Required**: No
- **Default**: FALSE
- **Description**: If TRUE, NULL values appear first. If FALSE, NULL values appear last

## Returns

- **Type**: Same as input array type
- **Description**: Returns a new array with elements sorted according to the specified order
- **NULL Handling**: Returns NULL if the input array is NULL. NULL elements within the array are sorted according to the nulls_first parameter

## Usage Notes

- Creates a new sorted array; the original array is not modified
- Supports arrays of any comparable type (numbers, strings, dates, etc.)
- String sorting is case-sensitive and uses lexicographic ordering
- The default behavior sorts in ascending order with NULLs last

## Examples

### Example 1: Basic Numeric Array Sorting

Sample data in `test_scores` table:
```
+------------+---------------+--------------------------------+
| student_id | student_name  | quiz_scores                    |
+------------+---------------+--------------------------------+
| 1          | Alice Johnson | [85,92,78,88,91]              |
| 2          | Bob Smith     | [72,68,75,80,77]              |
| 3          | Carol Davis   | [95,89,92,94,90]              |
| 4          | David Brown   | [60,65,70,68,72]              |
| 5          | Eve Wilson    | [88,85,90,87,86]              |
+------------+---------------+--------------------------------+
```

Query:
```sql
SELECT 
    student_name,
    quiz_scores,
    ARRAY_SORT(quiz_scores) AS scores_ascending,
    ARRAY_SORT(quiz_scores, FALSE) AS scores_descending
FROM test_scores
ORDER BY student_id;
```

Result:
```
+---------------+--------------------------------+--------------------+--------------------+
| student_name  | quiz_scores                    | scores_ascending   | scores_descending  |
+---------------+--------------------------------+--------------------+--------------------+
| Alice Johnson | [85,92,78,88,91]              | [78,85,88,91,92]   | [92,91,88,85,78]   |
| Bob Smith     | [72,68,75,80,77]              | [68,72,75,77,80]   | [80,77,75,72,68]   |
| Carol Davis   | [95,89,92,94,90]              | [89,90,92,94,95]   | [95,94,92,90,89]   |
| David Brown   | [60,65,70,68,72]              | [60,65,68,70,72]   | [72,70,68,65,60]   |
| Eve Wilson    | [88,85,90,87,86]              | [85,86,87,88,90]   | [90,88,87,86,85]   |
+---------------+--------------------------------+--------------------+--------------------+
```

This example sorts quiz scores in both ascending and descending order.

### Example 2: String Array Sorting

Sample data in `product_tags` table:
```
+------------+--------------+--------------------------------------------+
| product_id | product_name | tags                                       |
+------------+--------------+--------------------------------------------+
| 1          | Laptop Pro   | ['technology','portable','expensive','new'] |
| 2          | Coffee Maker | ['kitchen','appliance','electric','daily'] |
| 3          | Running Shoe | ['sports','footwear','athletic','comfort'] |
| 4          | Book Shelf   | ['furniture','storage','wood','home']      |
| 5          | Headphones   | ['audio','wireless','technology','music']  |
+------------+--------------+--------------------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    tags,
    ARRAY_SORT(tags) AS tags_sorted,
    ARRAY_SORT(tags, FALSE) AS tags_reverse
FROM product_tags
ORDER BY product_id;
```

Result:
```
+--------------+--------------------------------------------+-----------------------------------------------+-----------------------------------------------+
| product_name | tags                                       | tags_sorted                                   | tags_reverse                                  |
+--------------+--------------------------------------------+-----------------------------------------------+-----------------------------------------------+
| Laptop Pro   | ['technology','portable','expensive','new'] | ['expensive','new','portable','technology']   | ['technology','portable','new','expensive']   |
| Coffee Maker | ['kitchen','appliance','electric','daily'] | ['appliance','daily','electric','kitchen']   | ['kitchen','electric','daily','appliance']   |
| Running Shoe | ['sports','footwear','athletic','comfort'] | ['athletic','comfort','footwear','sports']   | ['sports','footwear','comfort','athletic']   |
| Book Shelf   | ['furniture','storage','wood','home']      | ['furniture','home','storage','wood']         | ['wood','storage','home','furniture']         |
| Headphones   | ['audio','wireless','technology','music']  | ['audio','music','technology','wireless']    | ['wireless','technology','music','audio']    |
+--------------+--------------------------------------------+-----------------------------------------------+-----------------------------------------------+
```

This example demonstrates lexicographic sorting of string arrays.

### Example 3: Handling NULL Values

Sample data in `sensor_readings` table:
```
+----------+-------------+----------------------------------------+
| sensor_id| sensor_type | temperature_readings                   |
+----------+-------------+----------------------------------------+
| 1        | outdoor     | [22.5,NULL,23.1,21.8,NULL,22.9]       |
| 2        | indoor      | [20.1,20.3,NULL,20.2,20.0]            |
| 3        | warehouse   | [18.5,NULL,NULL,19.2,18.8]            |
| 4        | freezer     | [-5.2,-4.8,NULL,-5.0,-4.9]            |
| 5        | greenhouse  | [25.5,26.1,NULL,25.8,NULL,26.0]       |
+----------+-------------+----------------------------------------+
```

Query:
```sql
SELECT 
    sensor_type,
    temperature_readings,
    ARRAY_SORT(temperature_readings) AS nulls_last,
    ARRAY_SORT(temperature_readings, TRUE, TRUE) AS nulls_first,
    ARRAY_SORT(temperature_readings, FALSE) AS desc_nulls_last
FROM sensor_readings
ORDER BY sensor_id;
```

Result:
```
+-------------+----------------------------------------+--------------------------------+--------------------------------+--------------------------------+
| sensor_type | temperature_readings                   | nulls_last                     | nulls_first                    | desc_nulls_last                |
+-------------+----------------------------------------+--------------------------------+--------------------------------+--------------------------------+
| outdoor     | [22.5,NULL,23.1,21.8,NULL,22.9]       | [21.8,22.5,22.9,23.1,NULL,NULL] | [NULL,NULL,21.8,22.5,22.9,23.1] | [23.1,22.9,22.5,21.8,NULL,NULL] |
| indoor      | [20.1,20.3,NULL,20.2,20.0]            | [20.0,20.1,20.2,20.3,NULL]     | [NULL,20.0,20.1,20.2,20.3]     | [20.3,20.2,20.1,20.0,NULL]     |
| warehouse   | [18.5,NULL,NULL,19.2,18.8]            | [18.5,18.8,19.2,NULL,NULL]     | [NULL,NULL,18.5,18.8,19.2]     | [19.2,18.8,18.5,NULL,NULL]     |
| freezer     | [-5.2,-4.8,NULL,-5.0,-4.9]            | [-5.2,-5.0,-4.9,-4.8,NULL]     | [NULL,-5.2,-5.0,-4.9,-4.8]     | [-4.8,-4.9,-5.0,-5.2,NULL]     |
| greenhouse  | [25.5,26.1,NULL,25.8,NULL,26.0]       | [25.5,25.8,26.0,26.1,NULL,NULL] | [NULL,NULL,25.5,25.8,26.0,26.1] | [26.1,26.0,25.8,25.5,NULL,NULL] |
+-------------+----------------------------------------+--------------------------------+--------------------------------+--------------------------------+
```

This example shows different ways to handle NULL values when sorting.

### Example 4: Date Array Sorting

Sample data in `project_milestones` table:
```
+------------+------------------+----------------------------------------------------+
| project_id | project_name     | milestone_dates                                    |
+------------+------------------+----------------------------------------------------+
| 1          | Website Redesign | ['2024-03-15','2024-01-10','2024-02-20','2024-04-05'] |
| 2          | Mobile App       | ['2024-02-01','2024-03-30','2024-01-15']         |
| 3          | Data Migration   | ['2024-04-10','2024-02-28','2024-03-20','2024-01-25'] |
| 4          | API Development  | ['2024-03-01','2024-04-15','2024-02-15']         |
| 5          | Security Audit   | ['2024-01-20','2024-02-10','2024-03-05','2024-04-20'] |
+------------+------------------+----------------------------------------------------+
```

Query:
```sql
SELECT 
    project_name,
    milestone_dates,
    ARRAY_SORT(milestone_dates) AS chronological,
    ARRAY_SORT(milestone_dates, FALSE) AS reverse_chronological,
    ARRAY_SIZE(milestone_dates) AS num_milestones
FROM project_milestones
ORDER BY project_id;
```

Result:
```
+------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+----------------+
| project_name     | milestone_dates                                    | chronological                                      | reverse_chronological                              | num_milestones |
+------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+----------------+
| Website Redesign | ['2024-03-15','2024-01-10','2024-02-20','2024-04-05'] | ['2024-01-10','2024-02-20','2024-03-15','2024-04-05'] | ['2024-04-05','2024-03-15','2024-02-20','2024-01-10'] | 4              |
| Mobile App       | ['2024-02-01','2024-03-30','2024-01-15']         | ['2024-01-15','2024-02-01','2024-03-30']         | ['2024-03-30','2024-02-01','2024-01-15']         | 3              |
| Data Migration   | ['2024-04-10','2024-02-28','2024-03-20','2024-01-25'] | ['2024-01-25','2024-02-28','2024-03-20','2024-04-10'] | ['2024-04-10','2024-03-20','2024-02-28','2024-01-25'] | 4              |
| API Development  | ['2024-03-01','2024-04-15','2024-02-15']         | ['2024-02-15','2024-03-01','2024-04-15']         | ['2024-04-15','2024-03-01','2024-02-15']         | 3              |
| Security Audit   | ['2024-01-20','2024-02-10','2024-03-05','2024-04-20'] | ['2024-01-20','2024-02-10','2024-03-05','2024-04-20'] | ['2024-04-20','2024-03-05','2024-02-10','2024-01-20'] | 4              |
+------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+----------------+
```

This example sorts project milestone dates chronologically and reverse chronologically.

### Example 5: Complex Sorting with Mixed Data

Sample data in `financial_metrics` table:
```
+------------+---------------+--------------------------------+-----------------------------+
| company_id | company_name  | revenue_millions               | profit_margins              |
+------------+---------------+--------------------------------+-----------------------------+
| 1          | TechCorp      | [120.5,135.2,118.7,142.1,NULL] | [0.15,0.18,0.14,0.20,NULL] |
| 2          | RetailMax     | [85.3,NULL,92.1,88.5,95.2]    | [0.08,NULL,0.09,0.07,0.10] |
| 3          | FinanceOne    | [200.1,195.5,210.3,205.8]     | [0.25,0.23,0.28,0.26]      |
| 4          | HealthPlus    | [NULL,75.2,78.9,82.1,85.5]    | [NULL,0.12,0.13,0.14,0.15] |
| 5          | EduTech       | [45.6,48.2,NULL,52.3,55.1]    | [0.05,0.06,NULL,0.08,0.09] |
+------------+---------------+--------------------------------+-----------------------------+
```

Query:
```sql
SELECT 
    company_name,
    revenue_millions,
    ARRAY_SORT(revenue_millions) AS revenue_sorted,
    profit_margins,
    ARRAY_SORT(profit_margins, FALSE, FALSE) AS margin_desc_nulls_last,
    ARRAY_MAX(ARRAY_SORT(revenue_millions)) AS max_revenue,
    ARRAY_MIN(ARRAY_SORT(profit_margins)) AS min_margin
FROM financial_metrics
ORDER BY company_id;
```

Result:
```
+---------------+--------------------------------+--------------------------------+-----------------------------+-------------------------+-------------+------------+
| company_name  | revenue_millions               | revenue_sorted                 | profit_margins              | margin_desc_nulls_last  | max_revenue | min_margin |
+---------------+--------------------------------+--------------------------------+-----------------------------+-------------------------+-------------+------------+
| TechCorp      | [120.5,135.2,118.7,142.1,NULL] | [118.7,120.5,135.2,142.1,NULL] | [0.15,0.18,0.14,0.20,NULL] | [0.20,0.18,0.15,0.14,NULL] | 142.1       | 0.14       |
| RetailMax     | [85.3,NULL,92.1,88.5,95.2]    | [85.3,88.5,92.1,95.2,NULL]    | [0.08,NULL,0.09,0.07,0.10] | [0.10,0.09,0.08,0.07,NULL] | 95.2        | 0.07       |
| FinanceOne    | [200.1,195.5,210.3,205.8]     | [195.5,200.1,205.8,210.3]     | [0.25,0.23,0.28,0.26]      | [0.28,0.26,0.25,0.23]     | 210.3       | 0.23       |
| HealthPlus    | [NULL,75.2,78.9,82.1,85.5]    | [75.2,78.9,82.1,85.5,NULL]    | [NULL,0.12,0.13,0.14,0.15] | [0.15,0.14,0.13,0.12,NULL] | 85.5        | 0.12       |
| EduTech       | [45.6,48.2,NULL,52.3,55.1]    | [45.6,48.2,52.3,55.1,NULL]    | [0.05,0.06,NULL,0.08,0.09] | [0.09,0.08,0.06,0.05,NULL] | 55.1        | 0.05       |
+---------------+--------------------------------+--------------------------------+-----------------------------+-------------------------+-------------+------------+
```

This example combines ARRAY_SORT with other array functions for complex financial analysis.