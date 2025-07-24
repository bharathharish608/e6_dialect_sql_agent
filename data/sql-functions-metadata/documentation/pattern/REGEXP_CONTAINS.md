# REGEXP_CONTAINS

## Description

`REGEXP_CONTAINS` checks if a string contains any match for the specified regular expression pattern. It returns TRUE if at least one match is found anywhere in the string, FALSE otherwise.

## Syntax

```sql
REGEXP_CONTAINS(string_expression, regex_pattern)
```

## Parameters

- **string_expression**: The string to search in (VARCHAR)
- **regex_pattern**: The regular expression pattern to search for (VARCHAR)

## Return Type

BOOLEAN - Returns TRUE if the pattern is found, FALSE otherwise

## Examples

### Example 1: Check for email pattern
```sql
SELECT REGEXP_CONTAINS('Contact us at support@example.com', '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}') AS has_email;
-- Result: TRUE

SELECT REGEXP_CONTAINS('Contact us at our website', '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}') AS has_email;
-- Result: FALSE
```

### Example 2: Check for phone number pattern
```sql
SELECT REGEXP_CONTAINS('Call us at (555) 123-4567', '\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}') AS has_phone;
-- Result: TRUE

SELECT REGEXP_CONTAINS('Our phone is 555.123.4567', '\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}') AS has_phone;
-- Result: FALSE (different format)
```

### Example 3: Check for specific word boundaries
```sql
SELECT REGEXP_CONTAINS('The cat is in the category', '\bcat\b') AS has_word_cat;
-- Result: TRUE (matches 'cat' as a whole word)

SELECT REGEXP_CONTAINS('The category is important', '\bcat\b') AS has_word_cat;
-- Result: FALSE ('cat' is part of 'category', not a whole word)
```

### Example 4: Check for numeric patterns
```sql
SELECT REGEXP_CONTAINS('Order #12345 shipped', '#[0-9]{5}') AS has_order_number;
-- Result: TRUE

SELECT REGEXP_CONTAINS('Product SKU: ABC-789-XYZ', '[A-Z]{3}-[0-9]{3}-[A-Z]{3}') AS has_sku;
-- Result: TRUE
```

### Example 5: Case-insensitive pattern matching
```sql
SELECT REGEXP_CONTAINS('Hello World', '(?i)hello') AS case_insensitive_match;
-- Result: TRUE

SELECT REGEXP_CONTAINS('HELLO WORLD', '(?i)hello') AS case_insensitive_match;
-- Result: TRUE
```

## Common Use Cases

1. **Data Validation**: Verify if input contains valid email addresses, phone numbers, or URLs
2. **Text Analysis**: Check for presence of specific patterns in log files or text data
3. **Data Quality**: Identify records containing specific formats or patterns
4. **Security**: Detect potentially malicious patterns in user input

## Notes

- The function uses Java-style regular expressions
- Special regex characters (like `.`, `*`, `+`, etc.) need to be escaped with `\` if you want to match them literally
- For case-insensitive matching, use the `(?i)` flag at the beginning of your pattern
- The function returns TRUE as soon as it finds the first match; it doesn't need to scan the entire string

## See Also

- [REGEXP_LIKE](REGEXP_LIKE.md) - For matching entire strings against a pattern
- [REGEXP_EXTRACT](REGEXP_EXTRACT.md) - For extracting matching substrings
- [REGEXP_COUNT](REGEXP_COUNT.md) - For counting pattern occurrences