# ARRAY_PREPEND

Prepends an element to the beginning of an array

## Syntax

```sql
ARRAY_PREPEND( <element>, <array> )
```

## Arguments

### element
- **Type**: ANY
- **Required**: Yes
- **Description**: The element to prepend to the array. Must be compatible with the array's element type.

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to which the element will be prepended. Can be an array of any data type.

## Returns

- **Type**: Same as input array
- **Description**: Returns a new array with the element prepended at the beginning. The original array is not modified.
- **NULL Handling**: If the array is NULL, returns NULL. If the element is NULL, prepends NULL to the array.

## Usage Notes

- The function creates a new array with the element added at the beginning
- The element type must be compatible with the array's element type
- Empty arrays can have elements prepended to them
- Supports nested arrays (arrays of arrays)
- Note that the parameter order is element first, then array (opposite of ARRAY_APPEND)

## Examples

### Example 1: Basic Array Prepend

Query:
```sql
SELECT ARRAY_PREPEND(0, [1, 2, 3]) AS prepended_array;
```

Result:
```
+-----------------+
| prepended_array |
+-----------------+
| [0, 1, 2, 3]    |
+-----------------+
```

This example demonstrates prepending a single integer to an integer array.

### Example 2: Prepending Strings

Query:
```sql
SELECT ARRAY_PREPEND('apple', ['banana', 'cherry', 'date']) AS fruit_array;
```

Result:
```
+--------------------------------+
| fruit_array                    |
+--------------------------------+
| ['apple', 'banana', 'cherry', 'date'] |
+--------------------------------+
```

This example shows prepending a string to a string array.

### Example 3: Handling NULL Values

Query:
```sql
SELECT 
    ARRAY_PREPEND(NULL, [1, 2, 3]) AS array_with_null,
    ARRAY_PREPEND(5, NULL) AS null_array_result,
    ARRAY_PREPEND(10, [NULL, 20, 30]) AS array_containing_null;
```

Result:
```
+-----------------+-------------------+-----------------------+
| array_with_null | null_array_result | array_containing_null |
+-----------------+-------------------+-----------------------+
| [NULL, 1, 2, 3] | NULL              | [10, NULL, 20, 30]    |
+-----------------+-------------------+-----------------------+
```

This example demonstrates how ARRAY_PREPEND handles NULL values in different scenarios.

### Example 4: Working with Empty Arrays

Query:
```sql
SELECT 
    ARRAY_PREPEND('first', []) AS single_element,
    ARRAY_PREPEND(2, ARRAY_PREPEND(1, [])) AS built_array;
```

Result:
```
+----------------+-------------+
| single_element | built_array |
+----------------+-------------+
| ['first']      | [2, 1]      |
+----------------+-------------+
```

This example shows prepending elements to empty arrays and building arrays in reverse order.

### Example 5: Practical Application with Table Data

Sample data in `task_queue` table:
```
+---------+-------------+---------------------------+
| task_id | task_name   | processing_steps          |
+---------+-------------+---------------------------+
| 1       | Order-101   | ['pack', 'ship', 'deliver'] |
| 2       | Order-102   | ['ship', 'deliver']       |
| 3       | Order-103   | ['deliver']               |
| 4       | Order-104   | []                        |
| 5       | Order-105   | ['pack', 'ship']          |
+---------+-------------+---------------------------+
```

Query:
```sql
SELECT 
    task_name,
    processing_steps,
    ARRAY_PREPEND('validate', processing_steps) AS full_process
FROM task_queue
ORDER BY task_id;
```

Result:
```
+-------------+---------------------------+-------------------------------------+
| task_name   | processing_steps          | full_process                        |
+-------------+---------------------------+-------------------------------------+
| Order-101   | ['pack', 'ship', 'deliver'] | ['validate', 'pack', 'ship', 'deliver'] |
| Order-102   | ['ship', 'deliver']       | ['validate', 'ship', 'deliver']     |
| Order-103   | ['deliver']               | ['validate', 'deliver']             |
| Order-104   | []                        | ['validate']                        |
| Order-105   | ['pack', 'ship']          | ['validate', 'pack', 'ship']        |
+-------------+---------------------------+-------------------------------------+
```

This example demonstrates using ARRAY_PREPEND to add a validation step to the beginning of each task's processing pipeline.