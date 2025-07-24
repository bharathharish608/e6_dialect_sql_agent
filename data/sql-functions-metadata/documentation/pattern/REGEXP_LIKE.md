# REGEXP_LIKE

## Description

`REGEXP_LIKE` checks if a string matches a regular expression pattern. Unlike `REGEXP_CONTAINS`, this function checks if the entire string conforms to the pattern (though the pattern can be written to match partial strings using appropriate regex syntax).

## Syntax

```sql
REGEXP_LIKE(string_expression, regex_pattern)
```

## Parameters

- **string_expression**: The string to match against the pattern (VARCHAR)
- **regex_pattern**: The regular expression pattern (VARCHAR)

## Return Type

BOOLEAN - Returns TRUE if the string matches the pattern, FALSE otherwise

## Examples

### Example 1: Validate email format
```sql
SELECT REGEXP_LIKE('john.doe@example.com', '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$') AS is_valid_email;
-- Result: TRUE

SELECT REGEXP_LIKE('invalid.email@', '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$') AS is_valid_email;
-- Result: FALSE
```

### Example 2: Match specific date formats
```sql
SELECT REGEXP_LIKE('2024-03-15', '^\d{4}-\d{2}-\d{2}$') AS is_iso_date;
-- Result: TRUE

SELECT REGEXP_LIKE('03/15/2024', '^\d{2}/\d{2}/\d{4}$') AS is_us_date;
-- Result: TRUE

SELECT REGEXP_LIKE('15-Mar-2024', '^\d{2}-[A-Za-z]{3}-\d{4}$') AS is_custom_date;
-- Result: TRUE
```

### Example 3: Validate product codes
```sql
SELECT REGEXP_LIKE('PROD-2024-A1B2', '^PROD-\d{4}-[A-Z0-9]{4}$') AS is_valid_product_code;
-- Result: TRUE

SELECT REGEXP_LIKE('prod-2024-a1b2', '^PROD-\d{4}-[A-Z0-9]{4}$') AS is_valid_product_code;
-- Result: FALSE (lowercase doesn't match)
```

### Example 4: Match patterns with optional components
```sql
-- Match phone numbers with optional country code
SELECT REGEXP_LIKE('+1-555-123-4567', '^\+?1?-?\d{3}-\d{3}-\d{4}$') AS is_phone;
-- Result: TRUE

SELECT REGEXP_LIKE('555-123-4567', '^\+?1?-?\d{3}-\d{3}-\d{4}$') AS is_phone;
-- Result: TRUE

SELECT REGEXP_LIKE('1-555-123-4567', '^\+?1?-?\d{3}-\d{3}-\d{4}$') AS is_phone;
-- Result: TRUE
```

### Example 5: Using wildcards and quantifiers
```sql
-- Match strings that start with 'log_' and end with '.txt'
SELECT REGEXP_LIKE('log_20240315.txt', '^log_.*\.txt$') AS is_log_file;
-- Result: TRUE

-- Match strings containing only alphanumeric characters
SELECT REGEXP_LIKE('User123', '^[a-zA-Z0-9]+$') AS is_alphanumeric;
-- Result: TRUE

SELECT REGEXP_LIKE('User-123', '^[a-zA-Z0-9]+$') AS is_alphanumeric;
-- Result: FALSE (contains hyphen)
```

## Common Use Cases

1. **Input Validation**: Ensure data conforms to expected formats before processing
2. **Data Quality Checks**: Identify records that don't match expected patterns
3. **ETL Processing**: Filter or route data based on pattern matching
4. **User Input Verification**: Validate form inputs like email, phone, postal codes

## Notes

- Use `^` and `$` anchors to match the entire string
- Without anchors, the pattern can match anywhere in the string
- The function is case-sensitive by default; use `(?i)` for case-insensitive matching
- Common pattern elements:
  - `\d` matches any digit
  - `\w` matches word characters (letters, digits, underscore)
  - `\s` matches whitespace
  - `.` matches any character
  - `*` means zero or more
  - `+` means one or more
  - `?` means zero or one

## See Also

- [REGEXP_CONTAINS](REGEXP_CONTAINS.md) - For checking if a pattern exists anywhere in a string
- [REGEXP_EXTRACT](REGEXP_EXTRACT.md) - For extracting matching portions
- [REGEXP_REPLACE](REGEXP_REPLACE.md) - For replacing patterns with new text