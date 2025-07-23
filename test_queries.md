# Test SQL Queries for Enhanced Agent

## Queries Requiring EXTRACT Functions

### 1. Date/Time Extraction Queries
**Incorrect Query:**
```sql
SELECT YEAR(date_column) as year, MONTH(date_column) as month
FROM sales_table
WHERE date_column BETWEEN '2023-01-01' AND '2023-12-31'
```

**Expected e6data Syntax:**
```sql
SELECT EXTRACT(YEAR FROM date_column) as year, EXTRACT(MONTH FROM date_column) as month
FROM sales_table
WHERE date_column BETWEEN CAST('2023-01-01' AS DATE) AND CAST('2023-12-31' AS DATE)
```

### 2. Time Component Extraction
**Incorrect Query:**
```sql
SELECT HOUR(timestamp_column) as hour, MINUTE(timestamp_column) as minute
FROM events_table
WHERE DAY(timestamp_column) = 15
```

**Expected e6data Syntax:**
```sql
SELECT EXTRACT(HOUR FROM timestamp_column) as hour, EXTRACT(MINUTE FROM timestamp_column) as minute
FROM events_table
WHERE EXTRACT(DAY FROM timestamp_column) = 15
```

### 3. Quarter and Week Extraction
**Incorrect Query:**
```sql
SELECT QUARTER(date_column) as quarter, WEEK(date_column) as week_number
FROM sales_table
GROUP BY QUARTER(date_column), WEEK(date_column)
```

**Expected e6data Syntax:**
```sql
SELECT EXTRACT(QUARTER FROM date_column) as quarter, EXTRACT(WEEK FROM date_column) as week_number
FROM sales_table
GROUP BY EXTRACT(QUARTER FROM date_column), EXTRACT(WEEK FROM date_column)
```

## Queries Requiring Date Casting

### 4. Date Literal Issues
**Incorrect Query:**
```sql
SELECT * FROM orders
WHERE order_date >= '2023-01-01' AND order_date <= '2023-12-31'
```

**Expected e6data Syntax:**
```sql
SELECT * FROM orders
WHERE order_date >= CAST('2023-01-01' AS DATE) AND order_date <= CAST('2023-12-31' AS DATE)
```

### 5. Date Arithmetic Issues
**Incorrect Query:**
```sql
SELECT order_date + INTERVAL 7 DAY as delivery_date
FROM orders
```

**Expected e6data Syntax:**
```sql
SELECT order_date + INTERVAL '7' DAY as delivery_date
FROM orders
```

## Queries Requiring String Functions

### 6. String Concatenation Issues
**Incorrect Query:**
```sql
SELECT CONCAT(first_name, ' ', last_name) as full_name
FROM customers
```

**Expected e6data Syntax:**
```sql
SELECT first_name || ' ' || last_name as full_name
FROM customers
```

### 7. String Length Issues
**Incorrect Query:**
```sql
SELECT LENGTH(description) as desc_length
FROM products
```

**Expected e6data Syntax:**
```sql
SELECT CHAR_LENGTH(description) as desc_length
FROM products
```

### 8. String Position Issues
**Incorrect Query:**
```sql
SELECT INSTR(email, '@') as at_position
FROM users
```

**Expected e6data Syntax:**
```sql
SELECT POSITION('@' IN email) as at_position
FROM users
```

## Queries Requiring Aggregation Functions

### 9. Conditional Aggregation Issues
**Incorrect Query:**
```sql
SELECT SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as completed_sales
FROM orders
```

**Expected e6data Syntax:**
```sql
SELECT SUM(amount) FILTER (WHERE status = 'completed') as completed_sales
FROM orders
```

### 10. Window Function Issues
**Incorrect Query:**
```sql
SELECT customer_id, order_date, amount,
       LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_amount
FROM orders
```

**Expected e6data Syntax:**
```sql
SELECT customer_id, order_date, amount,
       LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_amount
FROM orders
```

## Queries Requiring Type Casting

### 11. Implicit Type Conversion Issues
**Incorrect Query:**
```sql
SELECT 'Total: ' + CAST(SUM(amount) AS VARCHAR) as summary
FROM sales
```

**Expected e6data Syntax:**
```sql
SELECT 'Total: ' || CAST(SUM(amount) AS VARCHAR) as summary
FROM sales
```

### 12. Numeric Extraction Issues
**Incorrect Query:**
```sql
SELECT CAST(EXTRACT(MONTH FROM date_column) AS VARCHAR) as month_str
FROM events
```

**Expected e6data Syntax:**
```sql
SELECT CAST(EXTRACT(MONTH FROM date_column) AS VARCHAR) as month_str
FROM events
```

## Complex Queries Requiring Multiple Fixes

### 13. Multi-Issue Query
**Incorrect Query:**
```sql
SELECT 
    YEAR(order_date) as year,
    MONTH(order_date) as month,
    CONCAT('Q', QUARTER(order_date)) as quarter,
    SUM(amount) as total_sales,
    LENGTH(description) as desc_len
FROM orders
WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY YEAR(order_date), MONTH(order_date), QUARTER(order_date)
ORDER BY YEAR(order_date), MONTH(order_date)
```

**Expected e6data Syntax:**
```sql
SELECT 
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    'Q' || CAST(EXTRACT(QUARTER FROM order_date) AS VARCHAR) as quarter,
    SUM(amount) as total_sales,
    CHAR_LENGTH(description) as desc_len
FROM orders
WHERE order_date BETWEEN CAST('2023-01-01' AS DATE) AND CAST('2023-12-31' AS DATE)
GROUP BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date), EXTRACT(QUARTER FROM order_date)
ORDER BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
```

### 14. Date Range with Time Components
**Incorrect Query:**
```sql
SELECT 
    DATE(timestamp_column) as event_date,
    HOUR(timestamp_column) as event_hour,
    COUNT(*) as event_count
FROM events
WHERE timestamp_column >= '2023-01-01 00:00:00' 
  AND timestamp_column < '2023-01-02 00:00:00'
GROUP BY DATE(timestamp_column), HOUR(timestamp_column)
```

**Expected e6data Syntax:**
```sql
SELECT 
    CAST(timestamp_column AS DATE) as event_date,
    EXTRACT(HOUR FROM timestamp_column) as event_hour,
    COUNT(*) as event_count
FROM events
WHERE timestamp_column >= CAST('2023-01-01 00:00:00' AS TIMESTAMP) 
  AND timestamp_column < CAST('2023-01-02 00:00:00' AS TIMESTAMP)
GROUP BY CAST(timestamp_column AS DATE), EXTRACT(HOUR FROM timestamp_column)
```

## Natural Language Queries to Test

### 15. Natural Language Examples
- "Show me sales by year and month for 2023"
- "Get the hour of day when most orders are placed"
- "Calculate quarterly revenue for the last 2 years"
- "Find the day of week with highest sales"
- "Show me the week number and corresponding sales for each week in 2023"
- "Get the minute-level breakdown of events between 2 PM and 4 PM"
- "Calculate the decade and century for each customer's birth date"
- "Show me the epoch timestamp for each order"

## Expected Agent Behavior

The enhanced agent should:
1. **Recognize date/time functions** and convert them to EXTRACT syntax
2. **Handle date literals** by adding CAST statements
3. **Fix string concatenation** to use || instead of + or CONCAT()
4. **Use proper string functions** like CHAR_LENGTH instead of LENGTH
5. **Apply conditional aggregation** using FILTER clauses
6. **Handle type casting** explicitly where needed
7. **Use proper interval syntax** for date arithmetic
8. **Avoid reserved keywords** as aliases (month, year, date, etc.) 