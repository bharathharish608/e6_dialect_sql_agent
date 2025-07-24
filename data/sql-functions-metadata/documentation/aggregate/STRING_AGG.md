# STRING_AGG

## Description

`STRING_AGG` is an aggregate function that concatenates strings from a group into a single string. It allows you to combine multiple string values from different rows into one concatenated result, with optional separators between values.

## Syntax

```sql
STRING_AGG(column [, separator] [, limit])
```

## Parameters

- **column** (STRING): The string column or expression to concatenate. Required.
- **separator** (STRING): Optional. The string to use as a delimiter between concatenated values. If not specified, values are concatenated without any separator.
- **limit** (INTEGER): Optional. Maximum number of values to concatenate. If more values exist in the group, only the first `limit` values are included.

## Return Type

STRING NULLABLE - Returns a concatenated string or NULL if all input values are NULL.

## Usage Notes

- The function accepts 1 to 3 parameters
- When used with `GROUP BY`, concatenates values within each group
- Can be combined with `ORDER BY` within the function to control the order of concatenation
- NULL values are ignored during concatenation
- If all values in a group are NULL, the result is NULL
- The separator is only added between non-NULL values

## Examples

### Example 1: Basic Concatenation with Separator

```sql
-- Sample data: employees table
-- | department | employee_name |
-- |------------|---------------|
-- | Sales      | Alice         |
-- | Sales      | Bob           |
-- | Sales      | Charlie       |
-- | HR         | David         |
-- | HR         | Eve           |

SELECT 
    department,
    STRING_AGG(employee_name, ', ') AS team_members
FROM employees
GROUP BY department;

-- Result:
-- | department | team_members        |
-- |------------|---------------------|
-- | HR         | David, Eve          |
-- | Sales      | Alice, Bob, Charlie |
```

### Example 2: Concatenation Without Separator

```sql
-- Sample data: product_codes table
-- | category | code |
-- |----------|------|
-- | A        | X1   |
-- | A        | X2   |
-- | A        | X3   |
-- | B        | Y1   |
-- | B        | Y2   |

SELECT 
    category,
    STRING_AGG(code) AS combined_codes
FROM product_codes
GROUP BY category;

-- Result:
-- | category | combined_codes |
-- |----------|----------------|
-- | A        | X1X2X3         |
-- | B        | Y1Y2           |
```

### Example 3: Using ORDER BY Clause

```sql
-- Sample data: messages table
-- | conversation_id | timestamp           | message    |
-- |-----------------|---------------------|------------|
-- | 1               | 2024-01-01 10:00:00 | Hello      |
-- | 1               | 2024-01-01 10:01:00 | How are    |
-- | 1               | 2024-01-01 10:02:00 | you?       |
-- | 2               | 2024-01-01 11:00:00 | Good       |
-- | 2               | 2024-01-01 11:01:00 | morning    |

SELECT 
    conversation_id,
    STRING_AGG(message, ' ' ORDER BY timestamp) AS full_conversation
FROM messages
GROUP BY conversation_id;

-- Result:
-- | conversation_id | full_conversation |
-- |-----------------|-------------------|
-- | 1               | Hello How are you?|
-- | 2               | Good morning      |
```

### Example 4: Using Limit Parameter

```sql
-- Sample data: tags table
-- | article_id | tag        |
-- |------------|------------|
-- | 1          | database   |
-- | 1          | sql        |
-- | 1          | tutorial   |
-- | 1          | beginner   |
-- | 1          | guide      |
-- | 2          | python     |
-- | 2          | coding     |

SELECT 
    article_id,
    STRING_AGG(tag, ', ', 3) AS top_tags
FROM tags
GROUP BY article_id;

-- Result:
-- | article_id | top_tags                  |
-- |------------|---------------------------|
-- | 1          | database, sql, tutorial   |
-- | 2          | python, coding            |
```

### Example 5: Grouped Concatenation with Multiple Columns

```sql
-- Sample data: orders table
-- | customer_id | order_date | product_name | quantity |
-- |-------------|------------|--------------|----------|
-- | 101         | 2024-01-01 | Laptop       | 1        |
-- | 101         | 2024-01-01 | Mouse        | 2        |
-- | 101         | 2024-01-02 | Keyboard     | 1        |
-- | 102         | 2024-01-01 | Monitor      | 2        |
-- | 102         | 2024-01-02 | Cable        | 5        |

SELECT 
    customer_id,
    order_date,
    STRING_AGG(
        CONCAT(product_name, ' (', CAST(quantity AS VARCHAR), ')'), 
        ', '
    ) AS items_ordered
FROM orders
GROUP BY customer_id, order_date
ORDER BY customer_id, order_date;

-- Result:
-- | customer_id | order_date | items_ordered              |
-- |-------------|------------|----------------------------|
-- | 101         | 2024-01-01 | Laptop (1), Mouse (2)      |
-- | 101         | 2024-01-02 | Keyboard (1)               |
-- | 102         | 2024-01-01 | Monitor (2)                |
-- | 102         | 2024-01-02 | Cable (5)                  |
```

## Common Use Cases

1. **Creating comma-separated lists** - Combine multiple values into a single field for reporting
2. **Building dynamic SQL** - Concatenate column names or values for dynamic query construction
3. **Data denormalization** - Transform normalized data into a more readable format
4. **Log aggregation** - Combine multiple log entries into a single message
5. **Tag management** - Create tag lists for articles, products, or other entities

## Performance Considerations

- Be mindful of the resulting string length when concatenating many values
- Using the `limit` parameter can help control memory usage for large groups
- Ordering within STRING_AGG may impact performance on large datasets
- Consider indexing the columns used in GROUP BY clauses for better performance

## Related Functions

- `CONCAT()` - Concatenates multiple strings (non-aggregate)
- `CONCAT_WS()` - Concatenates with separator (non-aggregate)
- `GROUP_CONCAT()` - Similar function in some SQL dialects
- `LISTAGG()` - Equivalent function in other database systems