# ARG_MAX

## Description

The `ARG_MAX` function returns the value of the first column at the row where the second column has its maximum value. This is useful when you want to find associated values at the point where another column reaches its maximum.

## Syntax

```sql
ARG_MAX(value_column, comparison_column)
```

## Parameters

- **value_column** (ANY): The column whose value will be returned. Can be of any data type.
- **comparison_column** (COMPARABLE): The column used to determine the maximum. Must be a comparable type (numeric, date, string, etc.).

## Return Type

Returns the same data type as the `value_column`. The result is nullable and returns NULL if the input set is empty.

## Examples

### Example 1: Basic Usage - Product with Highest Price

Find the product name with the highest price in each category.

```sql
SELECT 
    category,
    ARG_MAX(product_name, price) AS most_expensive_product,
    MAX(price) AS highest_price
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
| category | most_expensive_product | highest_price |
|-----------|------------------------|---------------|
| Electronics | Laptop Pro | 1500.00 |
| Furniture | Sofa Deluxe | 2000.00 |

### Example 2: With Dates - Latest Record

Find the most recent order details for each customer.

```sql
SELECT 
    customer_id,
    ARG_MAX(order_id, order_date) AS latest_order_id,
    ARG_MAX(total_amount, order_date) AS latest_order_amount,
    MAX(order_date) AS latest_order_date
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
| customer_id | latest_order_id | latest_order_amount | latest_order_date |
|-------------|-----------------|---------------------|-------------------|
| C100 | 1003 | 300.00 | 2024-03-10 |
| C101 | 1005 | 600.00 | 2024-03-15 |

### Example 3: Grouped Results

Find the employee with the highest salary in each department and location.

```sql
SELECT 
    department,
    location,
    ARG_MAX(employee_name, salary) AS highest_paid_employee,
    ARG_MAX(employee_id, salary) AS employee_id,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department, location
ORDER BY department, location;
```

**Sample Input:**
| employee_id | employee_name | department | location | salary |
|-------------|---------------|------------|----------|---------|
| E001 | John Smith | Sales | New York | 75000 |
| E002 | Jane Doe | Sales | New York | 82000 |
| E003 | Bob Johnson | Sales | London | 70000 |
| E004 | Alice Brown | IT | New York | 95000 |
| E005 | Charlie Davis | IT | London | 88000 |

**Sample Output:**
| department | location | highest_paid_employee | employee_id | max_salary |
|------------|----------|-----------------------|-------------|------------|
| IT | London | Charlie Davis | E005 | 88000 |
| IT | New York | Alice Brown | E004 | 95000 |
| Sales | London | Bob Johnson | E003 | 70000 |
| Sales | New York | Jane Doe | E002 | 82000 |

### Example 4: With NULL Handling

Demonstrate how ARG_MAX handles NULL values in both value and comparison columns.

```sql
SELECT 
    product_category,
    ARG_MAX(product_name, rating) AS best_rated_product,
    ARG_MAX(manufacturer, rating) AS best_rated_manufacturer,
    MAX(rating) AS highest_rating
FROM product_reviews
WHERE rating IS NOT NULL
GROUP BY product_category;
```

**Sample Input:**
| product_name | manufacturer | product_category | rating |
|--------------|--------------|------------------|--------|
| Camera A1 | TechCorp | Photography | 4.8 |
| Camera B2 | PhotoPro | Photography | NULL |
| Camera C3 | NULL | Photography | 4.5 |
| Lens X1 | LensMaster | Photography | 4.9 |
| Phone Y1 | PhoneCo | Mobile | 4.6 |
| Phone Y2 | TechGiant | Mobile | NULL |

**Sample Output:**
| product_category | best_rated_product | best_rated_manufacturer | highest_rating |
|------------------|-------------------|-------------------------|----------------|
| Photography | Lens X1 | LensMaster | 4.9 |
| Mobile | Phone Y1 | PhoneCo | 4.6 |

### Example 5: Complex Business Scenario

Find the most successful sales representative (by total sales) for each product line in each quarter, along with their performance metrics.

```sql
WITH quarterly_sales AS (
    SELECT 
        product_line,
        EXTRACT(YEAR FROM sale_date) AS sale_year,
        EXTRACT(QUARTER FROM sale_date) AS sale_quarter,
        sales_rep_id,
        sales_rep_name,
        SUM(sale_amount) AS total_sales,
        COUNT(*) AS number_of_sales,
        AVG(customer_satisfaction) AS avg_satisfaction
    FROM sales_transactions
    WHERE sale_date >= '2024-01-01'
    GROUP BY 
        product_line,
        EXTRACT(YEAR FROM sale_date),
        EXTRACT(QUARTER FROM sale_date),
        sales_rep_id,
        sales_rep_name
)
SELECT 
    product_line,
    sale_year,
    sale_quarter,
    ARG_MAX(sales_rep_name, total_sales) AS top_performer,
    ARG_MAX(sales_rep_id, total_sales) AS top_performer_id,
    ARG_MAX(number_of_sales, total_sales) AS top_performer_sales_count,
    ARG_MAX(avg_satisfaction, total_sales) AS top_performer_satisfaction,
    MAX(total_sales) AS highest_sales_amount
FROM quarterly_sales
GROUP BY product_line, sale_year, sale_quarter
ORDER BY product_line, sale_year, sale_quarter;
```

**Sample Input (quarterly_sales CTE result):**
| product_line | sale_year | sale_quarter | sales_rep_id | sales_rep_name | total_sales | number_of_sales | avg_satisfaction |
|--------------|-----------|--------------|--------------|----------------|-------------|-----------------|------------------|
| Software | 2024 | 1 | SR001 | Mike Wilson | 125000 | 45 | 4.7 |
| Software | 2024 | 1 | SR002 | Sarah Lee | 98000 | 38 | 4.8 |
| Software | 2024 | 2 | SR001 | Mike Wilson | 145000 | 52 | 4.6 |
| Software | 2024 | 2 | SR002 | Sarah Lee | 155000 | 48 | 4.9 |
| Hardware | 2024 | 1 | SR003 | Tom Chen | 87000 | 25 | 4.5 |
| Hardware | 2024 | 1 | SR004 | Lisa Park | 92000 | 28 | 4.4 |

**Sample Output:**
| product_line | sale_year | sale_quarter | top_performer | top_performer_id | top_performer_sales_count | top_performer_satisfaction | highest_sales_amount |
|--------------|-----------|--------------|---------------|------------------|---------------------------|----------------------------|---------------------|
| Hardware | 2024 | 1 | Lisa Park | SR004 | 28 | 4.4 | 92000 |
| Software | 2024 | 1 | Mike Wilson | SR001 | 45 | 4.7 | 125000 |
| Software | 2024 | 2 | Sarah Lee | SR002 | 48 | 4.9 | 155000 |

## Notes

- If multiple rows have the same maximum value in the comparison column, ARG_MAX returns the value from the first row encountered (non-deterministic).
- NULL values in the comparison column are ignored when determining the maximum.
- If all values in the comparison column are NULL, the function returns NULL.
- The function is particularly useful for finding associated data at extreme points without using subqueries or window functions.
- Performance is generally better than using subqueries with MAX() for the same purpose.

## See Also

- [ARG_MIN](ARG_MIN.md) - Returns the value at the row with minimum comparison value
- [MAX](MAX.md) - Returns the maximum value
- [FIRST_VALUE](../window/FIRST_VALUE.md) - Window function for getting first value in a partition