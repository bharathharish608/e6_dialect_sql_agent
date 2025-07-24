# ARG_MIN

## Description

The `ARG_MIN` function returns the value of the first column at the row where the second column has its minimum value. This is useful when you want to find associated values at the point where another column reaches its minimum.

## Syntax

```sql
ARG_MIN(value_column, comparison_column)
```

## Parameters

- **value_column** (ANY): The column whose value will be returned. Can be of any data type.
- **comparison_column** (COMPARABLE): The column used to determine the minimum. Must be a comparable type (numeric, date, string, etc.).

## Return Type

Returns the same data type as the `value_column`. The result is nullable and returns NULL if the input set is empty.

## Examples

### Example 1: Basic Usage - Product with Lowest Price

Find the product name with the lowest price in each category.

```sql
SELECT 
    category,
    ARG_MIN(product_name, price) AS cheapest_product,
    MIN(price) AS lowest_price
FROM products
GROUP BY category;
```

**Sample Input:**
| product_name | category | price |
|--------------|-----------|--------|
| Laptop Pro | Electronics | 1500.00 |
| Phone X | Electronics | 1200.00 |
| Tablet Mini | Electronics | 800.00 |
| Sofa Deluxe | Furniture | 2000.00 |
| Chair Basic | Furniture | 150.00 |

**Sample Output:**
| category | cheapest_product | lowest_price |
|-----------|------------------|--------------|
| Electronics | Tablet Mini | 800.00 |
| Furniture | Chair Basic | 150.00 |

### Example 2: With Dates - Earliest Record

Find the first order details for each customer.

```sql
SELECT 
    customer_id,
    ARG_MIN(order_id, order_date) AS first_order_id,
    ARG_MIN(total_amount, order_date) AS first_order_amount,
    MIN(order_date) AS first_order_date
FROM orders
GROUP BY customer_id;
```

**Sample Input:**
| order_id | customer_id | order_date | total_amount |
|----------|-------------|------------|--------------|
| 1001 | C100 | 2024-01-15 | 250.00 |
| 1002 | C100 | 2024-02-20 | 175.00 |
| 1003 | C100 | 2024-03-10 | 300.00 |
| 1004 | C101 | 2024-01-20 | 450.00 |
| 1005 | C101 | 2024-03-15 | 600.00 |

**Sample Output:**
| customer_id | first_order_id | first_order_amount | first_order_date |
|-------------|----------------|-------------------|------------------|
| C100 | 1001 | 250.00 | 2024-01-15 |
| C101 | 1004 | 450.00 | 2024-01-20 |

### Example 3: Grouped Results

Find the employee with the lowest salary in each department and location (entry-level identification).

```sql
SELECT 
    department,
    location,
    ARG_MIN(employee_name, salary) AS entry_level_employee,
    ARG_MIN(employee_id, salary) AS employee_id,
    ARG_MIN(hire_date, salary) AS hire_date,
    MIN(salary) AS min_salary
FROM employees
GROUP BY department, location
ORDER BY department, location;
```

**Sample Input:**
| employee_id | employee_name | department | location | salary | hire_date |
|-------------|---------------|------------|----------|---------|------------|
| E001 | John Smith | Sales | New York | 75000 | 2023-01-15 |
| E002 | Jane Doe | Sales | New York | 82000 | 2022-06-20 |
| E003 | Bob Johnson | Sales | London | 70000 | 2023-09-01 |
| E004 | Alice Brown | IT | New York | 95000 | 2021-03-10 |
| E005 | Charlie Davis | IT | London | 88000 | 2022-11-15 |

**Sample Output:**
| department | location | entry_level_employee | employee_id | hire_date | min_salary |
|------------|----------|---------------------|-------------|-----------|------------|
| IT | London | Charlie Davis | E005 | 2022-11-15 | 88000 |
| IT | New York | Alice Brown | E004 | 2021-03-10 | 95000 |
| Sales | London | Bob Johnson | E003 | 2023-09-01 | 70000 |
| Sales | New York | John Smith | E001 | 2023-01-15 | 75000 |

### Example 4: With NULL Handling

Demonstrate how ARG_MIN handles NULL values in both value and comparison columns.

```sql
SELECT 
    store_region,
    ARG_MIN(store_name, operating_cost) AS most_efficient_store,
    ARG_MIN(manager_name, operating_cost) AS efficient_store_manager,
    MIN(operating_cost) AS lowest_operating_cost
FROM store_performance
WHERE operating_cost IS NOT NULL
GROUP BY store_region;
```

**Sample Input:**
| store_name | manager_name | store_region | operating_cost |
|------------|--------------|--------------|----------------|
| Store A | John Manager | North | 45000 |
| Store B | NULL | North | 48000 |
| Store C | Sarah Manager | North | NULL |
| Store D | Mike Manager | South | 42000 |
| Store E | Lisa Manager | South | NULL |
| Store F | Tom Manager | South | 44000 |

**Sample Output:**
| store_region | most_efficient_store | efficient_store_manager | lowest_operating_cost |
|--------------|---------------------|------------------------|--------------------|
| North | Store A | John Manager | 45000 |
| South | Store D | Mike Manager | 42000 |

### Example 5: Complex Business Scenario

Find the supplier with the shortest delivery time for each product category in each region, along with their reliability metrics.

```sql
WITH supplier_performance AS (
    SELECT 
        s.supplier_id,
        s.supplier_name,
        s.region,
        p.product_category,
        AVG(d.delivery_days) AS avg_delivery_time,
        COUNT(d.delivery_id) AS total_deliveries,
        SUM(CASE WHEN d.on_time = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS on_time_percentage,
        AVG(d.quality_score) AS avg_quality_score
    FROM suppliers s
    JOIN deliveries d ON s.supplier_id = d.supplier_id
    JOIN products p ON d.product_id = p.product_id
    WHERE d.delivery_date >= '2024-01-01'
    GROUP BY s.supplier_id, s.supplier_name, s.region, p.product_category
)
SELECT 
    region,
    product_category,
    ARG_MIN(supplier_name, avg_delivery_time) AS fastest_supplier,
    ARG_MIN(supplier_id, avg_delivery_time) AS fastest_supplier_id,
    ARG_MIN(total_deliveries, avg_delivery_time) AS fastest_supplier_deliveries,
    ARG_MIN(on_time_percentage, avg_delivery_time) AS fastest_supplier_on_time_pct,
    ARG_MIN(avg_quality_score, avg_delivery_time) AS fastest_supplier_quality,
    MIN(avg_delivery_time) AS shortest_avg_delivery_days
FROM supplier_performance
GROUP BY region, product_category
ORDER BY region, product_category;
```

**Sample Input (supplier_performance CTE result):**
| supplier_id | supplier_name | region | product_category | avg_delivery_time | total_deliveries | on_time_percentage | avg_quality_score |
|-------------|---------------|--------|------------------|-------------------|------------------|-------------------|-------------------|
| SUP001 | FastShip Co | East | Electronics | 2.5 | 150 | 95.0 | 4.8 |
| SUP002 | QuickDeliver | East | Electronics | 3.2 | 120 | 88.0 | 4.6 |
| SUP003 | SpeedyTrans | East | Furniture | 4.1 | 80 | 92.0 | 4.7 |
| SUP004 | RapidMove | East | Furniture | 3.8 | 95 | 94.0 | 4.9 |
| SUP005 | SwiftCargo | West | Electronics | 2.8 | 200 | 91.0 | 4.5 |
| SUP006 | ExpressShip | West | Electronics | 3.0 | 180 | 93.0 | 4.7 |

**Sample Output:**
| region | product_category | fastest_supplier | fastest_supplier_id | fastest_supplier_deliveries | fastest_supplier_on_time_pct | fastest_supplier_quality | shortest_avg_delivery_days |
|--------|------------------|------------------|---------------------|----------------------------|------------------------------|-------------------------|---------------------------|
| East | Electronics | FastShip Co | SUP001 | 150 | 95.0 | 4.8 | 2.5 |
| East | Furniture | RapidMove | SUP004 | 95 | 94.0 | 4.9 | 3.8 |
| West | Electronics | SwiftCargo | SUP005 | 200 | 91.0 | 4.5 | 2.8 |

## Notes

- If multiple rows have the same minimum value in the comparison column, ARG_MIN returns the value from the first row encountered (non-deterministic).
- NULL values in the comparison column are ignored when determining the minimum.
- If all values in the comparison column are NULL, the function returns NULL.
- The function is particularly useful for finding associated data at extreme points without using subqueries or window functions.
- Performance is generally better than using subqueries with MIN() for the same purpose.

## See Also

- [ARG_MAX](ARG_MAX.md) - Returns the value at the row with maximum comparison value
- [MIN](MIN.md) - Returns the minimum value
- [FIRST_VALUE](../window/FIRST_VALUE.md) - Window function for getting first value in a partition