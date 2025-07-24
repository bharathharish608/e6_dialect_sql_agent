# SET_SEARCH

## Description

The `SET_SEARCH` function checks whether a value exists within a set of values. It's useful for testing membership in a collection without writing multiple OR conditions. This function provides a cleaner and more efficient way to check if a value matches any item in a given set.

## Syntax

```sql
SET_SEARCH(value, set)
```

## Parameters

- `value` (ANY): The value to search for in the set
- `set` (ARRAY or comma-separated values): The set of values to search within

## Returns

- Type: `BOOLEAN`
- Description: TRUE if the value is found in the set, FALSE otherwise

## Examples

### Example 1: Search for Product Category
Check if a product belongs to specific categories.

```sql
SELECT product_name, category,
       SET_SEARCH(category, ARRAY['Electronics', 'Computers', 'Mobile']) AS is_tech_product
FROM products;
```

**Sample Results:**
```
product_name        | category      | is_tech_product
-------------------|---------------|----------------
iPhone 14          | Mobile        | true
Office Chair       | Furniture     | false
Laptop Pro         | Computers     | true
Book: SQL Guide    | Books         | false
Wireless Mouse     | Electronics   | true
```

### Example 2: Validate Status Codes
Check if status codes are in the success range.

```sql
SELECT request_id, status_code,
       SET_SEARCH(status_code, ARRAY[200, 201, 202, 204]) AS is_success
FROM api_logs;
```

**Sample Results:**
```
request_id | status_code | is_success
-----------|-------------|------------
REQ-001    | 200         | true
REQ-002    | 404         | false
REQ-003    | 201         | true
REQ-004    | 500         | false
REQ-005    | 204         | true
```

### Example 3: Filter Allowed Countries
Check if users are from allowed countries.

```sql
SELECT user_id, country,
       SET_SEARCH(country, ARRAY['USA', 'Canada', 'UK', 'Australia']) AS is_allowed_region
FROM users
WHERE SET_SEARCH(country, ARRAY['USA', 'Canada', 'UK', 'Australia']);
```

**Sample Results:**
```
user_id | country   | is_allowed_region
--------|-----------|------------------
U001    | USA       | true
U003    | Canada    | true
U005    | UK        | true
U008    | Australia | true
```

### Example 4: Check Department Access
Verify if employees belong to departments with special privileges.

```sql
SELECT employee_name, department,
       SET_SEARCH(department, ARRAY['Executive', 'Finance', 'HR']) AS has_sensitive_access
FROM employees;
```

**Sample Results:**
```
employee_name    | department   | has_sensitive_access
----------------|--------------|---------------------
John Smith      | Engineering  | false
Sarah Johnson   | Finance      | true
Mike Williams   | Sales        | false
Lisa Brown      | HR           | true
Tom Davis       | Executive    | true
```

### Example 5: Validate File Extensions
Check if uploaded files have allowed extensions.

```sql
SELECT filename,
       LOWER(REGEXP_EXTRACT(filename, '\.([^.]+)$', 1)) AS extension,
       SET_SEARCH(LOWER(REGEXP_EXTRACT(filename, '\.([^.]+)$', 1)), 
                  ARRAY['jpg', 'png', 'gif', 'pdf']) AS is_allowed_type
FROM uploads;
```

**Sample Results:**
```
filename           | extension | is_allowed_type
------------------|-----------|----------------
report.pdf        | pdf       | true
photo.jpg         | jpg       | true
document.docx     | docx      | false
image.PNG         | png       | true
script.exe        | exe       | false
```

## Common Use Cases

1. **Access Control**: Check if users/roles are in allowed lists
2. **Data Validation**: Validate values against predefined sets
3. **Filtering**: Efficiently filter records based on multiple criteria
4. **Configuration**: Check if settings match allowed values
5. **Categorization**: Group items based on set membership

## Notes

- More readable and maintainable than multiple OR conditions
- Performance is typically better than multiple OR comparisons for larger sets
- NULL values in the search value return NULL (not FALSE)
- NULL values in the set are handled according to SQL NULL semantics
- Case sensitivity depends on the data type and collation
- Some implementations may support different syntax for the set parameter