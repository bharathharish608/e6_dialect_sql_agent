# ARRAY_POSITION

Returns the position of an element in an array (1-based)

## Syntax

```sql
ARRAY_POSITION( <element>, <array> )
```

## Arguments

### element
- **Type**: ANY
- **Required**: Yes
- **Description**: The element to find in the array. Must be compatible with the array's element type.

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array to search in. Can be an array of any data type.

## Returns

- **Type**: INTEGER
- **Description**: Returns the 1-based position of the first occurrence of the element in the array. Returns 0 if the element is not found.
- **NULL Handling**: If the array is NULL, returns NULL. If the element is NULL, searches for NULL in the array.

## Usage Notes

- Returns the position of the first occurrence only (if element appears multiple times)
- Positions are 1-based (first element is at position 1, not 0)
- Returns 0 if the element is not found in the array
- Case-sensitive for string comparisons
- Supports searching for NULL values within arrays

## Examples

### Example 1: Basic Position Finding

Query:
```sql
SELECT 
    ARRAY_POSITION(3, [1, 2, 3, 4, 5]) AS position_of_3,
    ARRAY_POSITION(6, [1, 2, 3, 4, 5]) AS position_of_6,
    ARRAY_POSITION(1, [1, 2, 3, 4, 5]) AS position_of_1;
```

Result:
```
+---------------+---------------+---------------+
| position_of_3 | position_of_6 | position_of_1 |
+---------------+---------------+---------------+
| 3             | 0             | 1             |
+---------------+---------------+---------------+
```

This example shows finding positions of elements, including when an element is not found (returns 0).

### Example 2: Finding First Occurrence in Arrays with Duplicates

Query:
```sql
SELECT 
    ARRAY_POSITION(2, [1, 2, 3, 2, 5]) AS first_2,
    ARRAY_POSITION('a', ['b', 'a', 'c', 'a', 'd']) AS first_a,
    ARRAY_POSITION(10, [10, 20, 10, 30, 10]) AS first_10;
```

Result:
```
+---------+---------+----------+
| first_2 | first_a | first_10 |
+---------+---------+----------+
| 2       | 2       | 1        |
+---------+---------+----------+
```

This example demonstrates that ARRAY_POSITION returns the position of the first occurrence when duplicates exist.

### Example 3: Handling NULL Values

Query:
```sql
SELECT 
    ARRAY_POSITION(NULL, [1, 2, NULL, 4]) AS null_position,
    ARRAY_POSITION(5, NULL) AS null_array,
    ARRAY_POSITION(NULL, [1, 2, 3]) AS no_null_found,
    ARRAY_POSITION(1, []) AS empty_array;
```

Result:
```
+---------------+------------+---------------+-------------+
| null_position | null_array | no_null_found | empty_array |
+---------------+------------+---------------+-------------+
| 3             | NULL       | 0             | 0           |
+---------------+------------+---------------+-------------+
```

This example shows how ARRAY_POSITION handles NULL values and empty arrays.

### Example 4: Working with String Arrays

Query:
```sql
SELECT 
    ARRAY_POSITION('banana', ['apple', 'banana', 'cherry']) AS banana_pos,
    ARRAY_POSITION('BANANA', ['apple', 'banana', 'cherry']) AS uppercase_pos,
    ARRAY_POSITION('grape', ['apple', 'banana', 'cherry']) AS not_found;
```

Result:
```
+------------+---------------+-----------+
| banana_pos | uppercase_pos | not_found |
+------------+---------------+-----------+
| 2          | 0             | 0         |
+------------+---------------+-----------+
```

This example demonstrates case-sensitive string matching in ARRAY_POSITION.

### Example 5: Practical Application with Table Data

Sample data in `workflow_steps` table:
```
+-------------+------------------+--------------------------------------------+
| workflow_id | workflow_name    | step_sequence                              |
+-------------+------------------+--------------------------------------------+
| 1           | Order Processing | ['received', 'validated', 'packed', 'shipped'] |
| 2           | User Onboarding  | ['signup', 'verify', 'profile', 'welcome']  |
| 3           | Bug Fix          | ['reported', 'triaged', 'assigned', 'fixed', 'tested'] |
| 4           | Content Review   | ['draft', 'review', 'approved', 'published'] |
| 5           | Payment Process  | ['initiated', 'authorized', 'captured']     |
+-------------+------------------+--------------------------------------------+
```

Query:
```sql
SELECT 
    workflow_name,
    step_sequence,
    ARRAY_POSITION('validated', step_sequence) AS validation_step,
    ARRAY_POSITION('approved', step_sequence) AS approval_step,
    CASE 
        WHEN ARRAY_POSITION('approved', step_sequence) > 0 
        THEN 'Has Approval Step'
        ELSE 'No Approval Required'
    END AS approval_status
FROM workflow_steps
ORDER BY workflow_id;
```

Result:
```
+------------------+--------------------------------------------+-----------------+---------------+----------------------+
| workflow_name    | step_sequence                              | validation_step | approval_step | approval_status      |
+------------------+--------------------------------------------+-----------------+---------------+----------------------+
| Order Processing | ['received', 'validated', 'packed', 'shipped'] | 2               | 0             | No Approval Required |
| User Onboarding  | ['signup', 'verify', 'profile', 'welcome']  | 0               | 0             | No Approval Required |
| Bug Fix          | ['reported', 'triaged', 'assigned', 'fixed', 'tested'] | 0               | 0             | No Approval Required |
| Content Review   | ['draft', 'review', 'approved', 'published'] | 0               | 3             | Has Approval Step    |
| Payment Process  | ['initiated', 'authorized', 'captured']     | 0               | 0             | No Approval Required |
+------------------+--------------------------------------------+-----------------+---------------+----------------------+
```

This example demonstrates using ARRAY_POSITION to find specific steps in workflow sequences and make decisions based on their presence and position.