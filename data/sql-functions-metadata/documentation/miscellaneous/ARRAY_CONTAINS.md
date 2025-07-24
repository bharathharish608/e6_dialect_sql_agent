# ARRAY_CONTAINS

Checks if an array contains a specific element

## Syntax

```sql
ARRAY_CONTAINS( <array>, <element> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to search in. Can be an array of any data type.

### element
- **Type**: ANY
- **Required**: Yes
- **Description**: The element to search for in the array. Must be compatible with the array's element type.

## Returns

- **Type**: BOOLEAN
- **Description**: Returns TRUE if the array contains the element, FALSE if not.
- **NULL Handling**: If the array is NULL, returns NULL. If the element is NULL, checks if the array contains a NULL value.

## Usage Notes

- The function performs exact matching of the element value
- Works with arrays of any data type (integers, strings, dates, etc.)
- Case-sensitive for string comparisons
- Supports searching for NULL values within arrays
- Efficient for checking membership without needing to iterate through the array manually

## Examples

### Example 1: Basic Contains Check

Query:
```sql
SELECT 
    ARRAY_CONTAINS([1, 2, 3, 4, 5], 3) AS contains_3,
    ARRAY_CONTAINS([1, 2, 3, 4, 5], 6) AS contains_6;
```

Result:
```
+------------+------------+
| contains_3 | contains_6 |
+------------+------------+
| TRUE       | FALSE      |
+------------+------------+
```

This example demonstrates checking whether an integer array contains specific values.

### Example 2: String Array Contains

Query:
```sql
SELECT 
    ARRAY_CONTAINS(['apple', 'banana', 'cherry'], 'banana') AS has_banana,
    ARRAY_CONTAINS(['apple', 'banana', 'cherry'], 'BANANA') AS has_uppercase,
    ARRAY_CONTAINS(['apple', 'banana', 'cherry'], 'grape') AS has_grape;
```

Result:
```
+------------+---------------+-----------+
| has_banana | has_uppercase | has_grape |
+------------+---------------+-----------+
| TRUE       | FALSE         | FALSE     |
+------------+---------------+-----------+
```

This example shows that string comparisons are case-sensitive.

### Example 3: Handling NULL Values

Query:
```sql
SELECT 
    ARRAY_CONTAINS([1, 2, NULL, 4], NULL) AS contains_null,
    ARRAY_CONTAINS([1, 2, 3, 4], NULL) AS no_null,
    ARRAY_CONTAINS(NULL, 5) AS null_array,
    ARRAY_CONTAINS([], 1) AS empty_array;
```

Result:
```
+---------------+---------+------------+-------------+
| contains_null | no_null | null_array | empty_array |
+---------------+---------+------------+-------------+
| TRUE          | FALSE   | NULL       | FALSE       |
+---------------+---------+------------+-------------+
```

This example demonstrates how ARRAY_CONTAINS handles NULL values and empty arrays.

### Example 4: Working with Date Arrays

Query:
```sql
SELECT 
    ARRAY_CONTAINS(['2024-01-01'::date, '2024-02-01'::date, '2024-03-01'::date], '2024-02-01'::date) AS has_feb,
    ARRAY_CONTAINS(['2024-01-01'::date, '2024-02-01'::date, '2024-03-01'::date], '2024-04-01'::date) AS has_apr;
```

Result:
```
+---------+---------+
| has_feb | has_apr |
+---------+---------+
| TRUE    | FALSE   |
+---------+---------+
```

This example shows ARRAY_CONTAINS working with date arrays.

### Example 5: Practical Application with Table Data

Sample data in `product_tags` table:
```
+------------+------------------+----------------------------------------+
| product_id | product_name     | tags                                   |
+------------+------------------+----------------------------------------+
| 1          | Laptop Pro       | ['electronics', 'computers', 'premium'] |
| 2          | Wireless Mouse   | ['electronics', 'accessories']          |
| 3          | Office Chair     | ['furniture', 'ergonomic', 'office']    |
| 4          | Standing Desk    | ['furniture', 'ergonomic', 'adjustable']|
| 5          | USB-C Cable      | ['electronics', 'accessories', 'cables']|
+------------+------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    tags,
    ARRAY_CONTAINS(tags, 'electronics') AS is_electronic,
    ARRAY_CONTAINS(tags, 'ergonomic') AS is_ergonomic
FROM product_tags
WHERE ARRAY_CONTAINS(tags, 'electronics') OR ARRAY_CONTAINS(tags, 'ergonomic')
ORDER BY product_id;
```

Result:
```
+------------------+----------------------------------------+---------------+--------------+
| product_name     | tags                                   | is_electronic | is_ergonomic |
+------------------+----------------------------------------+---------------+--------------+
| Laptop Pro       | ['electronics', 'computers', 'premium'] | TRUE          | FALSE        |
| Wireless Mouse   | ['electronics', 'accessories']          | TRUE          | FALSE        |
| Office Chair     | ['furniture', 'ergonomic', 'office']    | FALSE         | TRUE         |
| Standing Desk    | ['furniture', 'ergonomic', 'adjustable']| FALSE         | TRUE         |
| USB-C Cable      | ['electronics', 'accessories', 'cables']| TRUE          | FALSE        |
+------------------+----------------------------------------+---------------+--------------+
```

This example demonstrates using ARRAY_CONTAINS to filter and categorize products based on their tags.