# ARRAY_APPEND

Appends an element to the end of an array

## Syntax

```sql
ARRAY_APPEND( <array>, <element> )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to which the element will be appended. Can be an array of any data type.

### element
- **Type**: ANY
- **Required**: Yes
- **Description**: The element to append to the array. Must be compatible with the array's element type.

## Returns

- **Type**: Same as input array
- **Description**: Returns a new array with the element appended at the end. The original array is not modified.
- **NULL Handling**: If the array is NULL, returns NULL. If the element is NULL, appends NULL to the array.

## Usage Notes

- The function creates a new array with the element added at the end
- The element type must be compatible with the array's element type
- Empty arrays can have elements appended to them
- Supports nested arrays (arrays of arrays)

## Examples

### Example 1: Basic Array Append

Query:
```sql
SELECT ARRAY_APPEND([1, 2, 3], 4) AS appended_array;
```

Result:
```
+----------------+
| appended_array |
+----------------+
| [1, 2, 3, 4]   |
+----------------+
```

This example demonstrates appending a single integer to an integer array.

### Example 2: Appending Strings

Query:
```sql
SELECT ARRAY_APPEND(['apple', 'banana', 'cherry'], 'date') AS fruit_array;
```

Result:
```
+-------------------------------+
| fruit_array                   |
+-------------------------------+
| ['apple', 'banana', 'cherry', 'date'] |
+-------------------------------+
```

This example shows appending a string to a string array.

### Example 3: Handling NULL Values

Query:
```sql
SELECT 
    ARRAY_APPEND([1, 2, 3], NULL) AS array_with_null,
    ARRAY_APPEND(NULL, 5) AS null_array_result,
    ARRAY_APPEND([10, NULL, 30], 40) AS array_containing_null;
```

Result:
```
+-----------------+-------------------+-----------------------+
| array_with_null | null_array_result | array_containing_null |
+-----------------+-------------------+-----------------------+
| [1, 2, 3, NULL] | NULL              | [10, NULL, 30, 40]    |
+-----------------+-------------------+-----------------------+
```

This example demonstrates how ARRAY_APPEND handles NULL values in different scenarios.

### Example 4: Working with Empty Arrays

Query:
```sql
SELECT 
    ARRAY_APPEND([], 'first') AS single_element,
    ARRAY_APPEND(ARRAY_APPEND([], 1), 2) AS built_array;
```

Result:
```
+----------------+-------------+
| single_element | built_array |
+----------------+-------------+
| ['first']      | [1, 2]      |
+----------------+-------------+
```

This example shows appending elements to empty arrays and building arrays incrementally.

### Example 5: Practical Application with Table Data

Sample data in `user_preferences` table:
```
+---------+------------+---------------------------+
| user_id | username   | favorite_colors           |
+---------+------------+---------------------------+
| 1       | alice      | ['blue', 'green']         |
| 2       | bob        | ['red']                   |
| 3       | charlie    | ['yellow', 'purple', 'orange'] |
| 4       | diana      | []                        |
| 5       | eve        | ['black', 'white']        |
+---------+------------+---------------------------+
```

Query:
```sql
SELECT 
    username,
    favorite_colors,
    ARRAY_APPEND(favorite_colors, 'gold') AS updated_colors
FROM user_preferences
ORDER BY user_id;
```

Result:
```
+----------+---------------------------+--------------------------------+
| username | favorite_colors           | updated_colors                 |
+----------+---------------------------+--------------------------------+
| alice    | ['blue', 'green']         | ['blue', 'green', 'gold']      |
| bob      | ['red']                   | ['red', 'gold']                |
| charlie  | ['yellow', 'purple', 'orange'] | ['yellow', 'purple', 'orange', 'gold'] |
| diana    | []                        | ['gold']                       |
| eve      | ['black', 'white']        | ['black', 'white', 'gold']     |
+----------+---------------------------+--------------------------------+
```

This example demonstrates using ARRAY_APPEND to add a new favorite color to each user's existing preferences.