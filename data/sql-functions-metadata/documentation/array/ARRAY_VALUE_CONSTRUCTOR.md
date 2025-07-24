# ARRAY_VALUE_CONSTRUCTOR

Constructs an array from values using ARRAY[...] syntax

## Syntax

```sql
ARRAY[<expression1>, <expression2> [ , <expressionN> ... ]]
```

## Arguments

### expression1, expression2, ..., expressionN
- **Type**: Variable number of values of compatible types
- **Required**: No (can create empty arrays)
- **Description**: Values to include in the array. All values must be of compatible types that can be coerced to a common type.

## Returns

- **Type**: ARRAY
- **Description**: Returns an array containing all the provided values in the order specified.
- **NULL Handling**: NULL values are preserved as array elements.

## Usage Notes

- Constructs an array from values using ARRAY[...] syntax
- All values must be of compatible types (e.g., all numeric, all strings)
- Empty arrays can be created using ARRAY[]
- Arrays can be nested to create multi-dimensional arrays
- NULL values are allowed and preserved in the array

## Examples

### Example 1: Basic Integer Array Construction

Sample data in `employees` table:
```
+----+------------------+------------+--------+------------+----------------------+
| id | name             | department | salary | hire_date  | email                |
+----+------------------+------------+--------+------------+----------------------+
| 1  | John Doe         | Sales      | 50000  | 2020-01-15 | john.doe@company.com |
| 2  | Jane Smith       | Marketing  | 55000  | 2019-03-20 | jane.s@company.com   |
| 3  | Bob Johnson      | IT         | 60000  | 2021-06-10 | bob.j@company.com    |
| 4  | Alice Brown      | HR         | 52000  | 2020-08-05 | alice.b@company.com  |
| 5  | Charlie Wilson   | Sales      | 48000  | 2022-02-14 | charlie.w@company.com|
+----+------------------+------------+--------+------------+----------------------+
```

Query:
```sql
SELECT 
    id,
    name,
    ARRAY[id, 100, 200, 300] AS id_array,
    ARRAY[1, 2, 3, 4, 5] AS simple_int_array
FROM employees
WHERE id <= 3;
```

Result:
```
+----+-------------+----------------+------------------+
| id | name        | id_array       | simple_int_array |
+----+-------------+----------------+------------------+
| 1  | John Doe    | [1,100,200,300]| [1,2,3,4,5]      |
| 2  | Jane Smith  | [2,100,200,300]| [1,2,3,4,5]      |
| 3  | Bob Johnson | [3,100,200,300]| [1,2,3,4,5]      |
+----+-------------+----------------+------------------+
```

This example demonstrates creating arrays with integer values, including both literal values and column references.

### Example 2: String Array Construction

Query:
```sql
SELECT 
    id,
    name,
    department,
    ARRAY[name, department, email] AS contact_info,
    ARRAY['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] AS workdays
FROM employees
WHERE id IN (1, 2);
```

Result:
```
+----+------------+------------+------------------------------------------------+--------------------------------------------------+
| id | name       | department | contact_info                                   | workdays                                         |
+----+------------+------------+------------------------------------------------+--------------------------------------------------+
| 1  | John Doe   | Sales      | ['John Doe','Sales','john.doe@company.com']   | ['Monday','Tuesday','Wednesday','Thursday','Friday'] |
| 2  | Jane Smith | Marketing  | ['Jane Smith','Marketing','jane.s@company.com']| ['Monday','Tuesday','Wednesday','Thursday','Friday'] |
+----+------------+------------+------------------------------------------------+--------------------------------------------------+
```

This example shows creating arrays with string values from both columns and literals.

### Example 3: Mixed But Compatible Types

Query:
```sql
SELECT 
    id,
    name,
    salary,
    ARRAY[salary, salary * 0.1, salary * 1.1] AS salary_calculations,
    ARRAY[1, 2.5, 3.14, 100] AS numeric_array
FROM employees
WHERE id <= 3;
```

Result:
```
+----+-------------+--------+-------------------------+------------------+
| id | name        | salary | salary_calculations     | numeric_array    |
+----+-------------+--------+-------------------------+------------------+
| 1  | John Doe    | 50000  | [50000,5000,55000]      | [1,2.5,3.14,100] |
| 2  | Jane Smith  | 55000  | [55000,5500,60500]      | [1,2.5,3.14,100] |
| 3  | Bob Johnson | 60000  | [60000,6000,66000]      | [1,2.5,3.14,100] |
+----+-------------+--------+-------------------------+------------------+
```

This example demonstrates arrays with mixed numeric types (integers and decimals) that are automatically coerced to a common type.

### Example 4: Empty Array

Query:
```sql
SELECT 
    id,
    name,
    ARRAY[] AS empty_array,
    CASE 
        WHEN department = 'Sales' THEN ARRAY[id, 1, 2]
        ELSE ARRAY[]
    END AS conditional_array
FROM employees
WHERE id <= 4;
```

Result:
```
+----+-------------+-------------+------------------+
| id | name        | empty_array | conditional_array |
+----+-------------+-------------+------------------+
| 1  | John Doe    | []          | [1,1,2]          |
| 2  | Jane Smith  | []          | []               |
| 3  | Bob Johnson | []          | []               |
| 4  | Alice Brown | []          | []               |
+----+-------------+-------------+------------------+
```

This example shows how to create empty arrays and use them in conditional logic.

### Example 5: Nested Arrays

Query:
```sql
SELECT 
    id,
    name,
    ARRAY[ARRAY[1, 2], ARRAY[3, 4], ARRAY[5, 6]] AS matrix_2x3,
    ARRAY[
        ARRAY['Name', name],
        ARRAY['Dept', department],
        ARRAY['Email', email]
    ] AS employee_details
FROM employees
WHERE id = 1;
```

Result:
```
+----+----------+----------------------+-------------------------------------------------------------------------+
| id | name     | matrix_2x3           | employee_details                                                        |
+----+----------+----------------------+-------------------------------------------------------------------------+
| 1  | John Doe | [[1,2],[3,4],[5,6]]  | [['Name','John Doe'],['Dept','Sales'],['Email','john.doe@company.com']] |
+----+----------+----------------------+-------------------------------------------------------------------------+
```

This example demonstrates creating nested arrays (arrays of arrays) for multi-dimensional data structures.