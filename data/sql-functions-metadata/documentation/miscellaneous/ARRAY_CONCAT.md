# ARRAY_CONCAT

Concatenates multiple arrays

## Syntax

```sql
ARRAY_CONCAT( <array1>, <array2> [ , <arrayN> ... ] )
```

## Arguments

### array1
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The first array to concatenate. Can be an array of any data type.

### array2
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The second array to concatenate. Must have compatible element type with array1.

### arrayN
- **Type**: ARRAY
- **Required**: No
- **Description**: Additional arrays to concatenate. All arrays must have compatible element types.

## Returns

- **Type**: ARRAY
- **Description**: Returns a new array containing all elements from all input arrays in the order they appear.
- **NULL Handling**: If any input array is NULL, the entire result is NULL. NULL elements within arrays are preserved.

## Usage Notes

- Concatenates two or more arrays into a single array
- All arrays must have compatible element types
- The order of elements is preserved (elements from array1, then array2, etc.)
- Empty arrays can be concatenated and contribute no elements to the result
- The function accepts a variable number of arrays (minimum 2)

## Examples

### Example 1: Basic Array Concatenation

Query:
```sql
SELECT 
    ARRAY_CONCAT([1, 2], [3, 4]) AS two_arrays,
    ARRAY_CONCAT([1, 2], [3, 4], [5, 6]) AS three_arrays;
```

Result:
```
+-------------+-------------------+
| two_arrays  | three_arrays      |
+-------------+-------------------+
| [1, 2, 3, 4] | [1, 2, 3, 4, 5, 6] |
+-------------+-------------------+
```

This example shows concatenating two and three integer arrays.

### Example 2: String Array Concatenation

Query:
```sql
SELECT 
    ARRAY_CONCAT(['hello', 'world'], ['from', 'SQL']) AS greeting,
    ARRAY_CONCAT(['a'], ['b', 'c'], ['d', 'e', 'f']) AS letters;
```

Result:
```
+----------------------------+---------------------------+
| greeting                   | letters                   |
+----------------------------+---------------------------+
| ['hello', 'world', 'from', 'SQL'] | ['a', 'b', 'c', 'd', 'e', 'f'] |
+----------------------------+---------------------------+
```

This example demonstrates concatenating string arrays of different sizes.

### Example 3: Handling NULL Values and Empty Arrays

Query:
```sql
SELECT 
    ARRAY_CONCAT([1, 2, NULL], [3, 4]) AS with_null_element,
    ARRAY_CONCAT([1, 2], NULL) AS null_array,
    ARRAY_CONCAT([], [1, 2, 3]) AS empty_first,
    ARRAY_CONCAT([1, 2, 3], []) AS empty_last,
    ARRAY_CONCAT([], []) AS both_empty;
```

Result:
```
+-------------------+------------+-------------+-------------+------------+
| with_null_element | null_array | empty_first | empty_last  | both_empty |
+-------------------+------------+-------------+-------------+------------+
| [1, 2, NULL, 3, 4] | NULL       | [1, 2, 3]   | [1, 2, 3]   | []         |
+-------------------+------------+-------------+-------------+------------+
```

This example shows how ARRAY_CONCAT handles NULL values and empty arrays.

### Example 4: Concatenating Multiple Arrays

Query:
```sql
SELECT 
    ARRAY_CONCAT(
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ) AS concatenated_sequence;
```

Result:
```
+--------------------------------+
| concatenated_sequence          |
+--------------------------------+
| [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
+--------------------------------+
```

This example demonstrates concatenating five arrays into a single sequence.

### Example 5: Practical Application with Table Data

Sample data in `regional_sales` table:
```
+-----------+------------+-------------------------+-------------------------+
| region_id | region     | q1_top_products         | q2_top_products         |
+-----------+------------+-------------------------+-------------------------+
| 1         | North      | ['laptop', 'mouse']     | ['keyboard', 'monitor'] |
| 2         | South      | ['tablet', 'stylus']    | ['laptop', 'webcam']    |
| 3         | East       | ['phone', 'charger']    | ['headset', 'tablet']   |
| 4         | West       | ['monitor', 'cable']    | ['mouse', 'keyboard']   |
| 5         | Central    | ['printer']             | ['scanner', 'laptop']   |
+-----------+------------+-------------------------+-------------------------+
```

Query:
```sql
SELECT 
    region,
    q1_top_products,
    q2_top_products,
    ARRAY_CONCAT(q1_top_products, q2_top_products) AS h1_all_products,
    ARRAY_CONCAT(q1_top_products, q2_top_products, ['bonus-item']) AS products_with_bonus
FROM regional_sales
ORDER BY region_id;
```

Result:
```
+----------+-------------------------+-------------------------+----------------------------------------+-----------------------------------------------+
| region   | q1_top_products         | q2_top_products         | h1_all_products                        | products_with_bonus                           |
+----------+-------------------------+-------------------------+----------------------------------------+-----------------------------------------------+
| North    | ['laptop', 'mouse']     | ['keyboard', 'monitor'] | ['laptop', 'mouse', 'keyboard', 'monitor'] | ['laptop', 'mouse', 'keyboard', 'monitor', 'bonus-item'] |
| South    | ['tablet', 'stylus']    | ['laptop', 'webcam']    | ['tablet', 'stylus', 'laptop', 'webcam'] | ['tablet', 'stylus', 'laptop', 'webcam', 'bonus-item'] |
| East     | ['phone', 'charger']    | ['headset', 'tablet']   | ['phone', 'charger', 'headset', 'tablet'] | ['phone', 'charger', 'headset', 'tablet', 'bonus-item'] |
| West     | ['monitor', 'cable']    | ['mouse', 'keyboard']   | ['monitor', 'cable', 'mouse', 'keyboard'] | ['monitor', 'cable', 'mouse', 'keyboard', 'bonus-item'] |
| Central  | ['printer']             | ['scanner', 'laptop']   | ['printer', 'scanner', 'laptop']      | ['printer', 'scanner', 'laptop', 'bonus-item'] |
+----------+-------------------------+-------------------------+----------------------------------------+-----------------------------------------------+
```

This example demonstrates using ARRAY_CONCAT to combine quarterly product lists and add bonus items for each region.