# REGEXP_REPLACE

## Description

`REGEXP_REPLACE` replaces all substrings in a string that match a regular expression pattern with a replacement string. You can use capturing groups in the pattern and reference them in the replacement string.

## Syntax

```sql
REGEXP_REPLACE(string_expression, regex_pattern [, replacement])
```

## Parameters

- **string_expression**: The string to perform replacements on (VARCHAR)
- **regex_pattern**: The regular expression pattern to match (VARCHAR)
- **replacement** (optional): The replacement string (VARCHAR, default is empty string)

## Return Type

VARCHAR - Returns the string with all matching substrings replaced

## Examples

### Example 1: Remove or replace special characters
```sql
-- Remove all non-alphanumeric characters
SELECT REGEXP_REPLACE('Phone: (555) 123-4567', '[^0-9a-zA-Z]', '') AS cleaned_phone;
-- Result: 'Phone5551234567'

-- Replace multiple spaces with single space
SELECT REGEXP_REPLACE('Hello    World     Test', '\s+', ' ') AS normalized_text;
-- Result: 'Hello World Test'

-- Remove HTML tags
SELECT REGEXP_REPLACE('<p>Hello <b>World</b></p>', '<[^>]+>', '') AS text_only;
-- Result: 'Hello World'
```

### Example 2: Format phone numbers
```sql
-- Format 10-digit phone number
SELECT REGEXP_REPLACE('5551234567', '(\d{3})(\d{3})(\d{4})', '($1) $2-$3') AS formatted_phone;
-- Result: '(555) 123-4567'

-- Standardize different phone formats
SELECT REGEXP_REPLACE('555.123.4567', '[\.\-\s]', '-') AS standardized_phone;
-- Result: '555-123-4567'

SELECT REGEXP_REPLACE('(555) 123 4567', '[\(\)\s\-]', '') AS digits_only;
-- Result: '5551234567'
```

### Example 3: Mask sensitive data
```sql
-- Mask credit card number (keep last 4 digits)
SELECT REGEXP_REPLACE('1234-5678-9012-3456', '\d{4}-\d{4}-\d{4}-(\d{4})', 'XXXX-XXXX-XXXX-$1') AS masked_cc;
-- Result: 'XXXX-XXXX-XXXX-3456'

-- Mask email address (keep domain)
SELECT REGEXP_REPLACE('john.doe@example.com', '^[^@]+(@.+)$', 'XXXXX$1') AS masked_email;
-- Result: 'XXXXX@example.com'

-- Partially mask SSN
SELECT REGEXP_REPLACE('123-45-6789', '(\d{3})-(\d{2})-(\d{4})', 'XXX-XX-$3') AS masked_ssn;
-- Result: 'XXX-XX-6789'
```

### Example 4: Text transformation
```sql
-- Convert camelCase to snake_case
SELECT REGEXP_REPLACE('getUserName', '([a-z])([A-Z])', '$1_$2') AS snake_case;
-- Result: 'get_User_Name' (then LOWER() can be applied)

-- Add spaces before capital letters
SELECT REGEXP_REPLACE('XMLHttpRequest', '([A-Z])', ' $1') AS spaced_text;
-- Result: ' X M L Http Request' (then TRIM() can be applied)

-- Replace multiple patterns (e.g., normalize line endings)
SELECT REGEXP_REPLACE(REGEXP_REPLACE('Line1\r\nLine2\rLine3', '\r\n', '\n'), '\r', '\n') AS normalized_text;
-- Result: 'Line1\nLine2\nLine3'
```

### Example 5: Advanced replacements with backreferences
```sql
-- Swap first and last name
SELECT REGEXP_REPLACE('Doe, John', '(\w+),\s*(\w+)', '$2 $1') AS swapped_name;
-- Result: 'John Doe'

-- Convert date format from MM/DD/YYYY to YYYY-MM-DD
SELECT REGEXP_REPLACE('03/15/2024', '(\d{2})/(\d{2})/(\d{4})', '$3-$1-$2') AS iso_date;
-- Result: '2024-03-15'

-- Add currency symbol and format
SELECT REGEXP_REPLACE('Price: 1234.56', 'Price:\s*(\d+\.?\d*)', 'Price: $$1') AS formatted_price;
-- Result: 'Price: $1234.56'

-- Extract and reformat log entry
SELECT REGEXP_REPLACE('[ERROR] 2024-03-15 10:30:45 - Connection failed', 
                     '\[(\w+)\]\s*(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})\s*-\s*(.+)', 
                     '$2T$3 [$1] $4') AS reformatted_log;
-- Result: '2024-03-15T10:30:45 [ERROR] Connection failed'
```

## Common Use Cases

1. **Data Cleaning**: Remove unwanted characters or normalize formats
2. **Data Masking**: Hide sensitive information while preserving structure
3. **Format Conversion**: Transform data between different formats
4. **Text Normalization**: Standardize variations in text representation
5. **Template Processing**: Replace placeholders with actual values

## Notes

- The replacement happens globally (all matches are replaced)
- Use `$1`, `$2`, etc. to reference capturing groups in the replacement string
- `$0` refers to the entire match
- To include a literal `$` in the replacement, use `$$`
- If no replacement string is provided, matches are removed (replaced with empty string)
- The function processes the string from left to right; earlier replacements can affect later matches

## See Also

- [REGEXP_EXTRACT](REGEXP_EXTRACT.md) - For extracting matching substrings
- [REGEXP_CONTAINS](REGEXP_CONTAINS.md) - For checking pattern existence
- [REGEXP_LIKE](REGEXP_LIKE.md) - For pattern matching validation