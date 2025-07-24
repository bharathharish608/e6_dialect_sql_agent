# REGEXP_EXTRACT

## Description

`REGEXP_EXTRACT` extracts a substring from a string that matches a regular expression pattern. You can optionally specify which capturing group to return when the pattern contains multiple groups.

## Syntax

```sql
REGEXP_EXTRACT(string_expression, regex_pattern [, group_number])
```

## Parameters

- **string_expression**: The string to extract from (VARCHAR)
- **regex_pattern**: The regular expression pattern with optional capturing groups (VARCHAR)
- **group_number** (optional): Which capturing group to return (INTEGER, default is 0 for the entire match)

## Return Type

VARCHAR - Returns the matched substring, or NULL if no match is found

## Examples

### Example 1: Extract email domains
```sql
SELECT REGEXP_EXTRACT('Contact: john.doe@example.com', '@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 1) AS domain;
-- Result: 'example.com'

SELECT REGEXP_EXTRACT('Email: support@company.org', '[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 1) AS domain;
-- Result: 'company.org'
```

### Example 2: Extract numeric values
```sql
-- Extract order number
SELECT REGEXP_EXTRACT('Order #12345 has been shipped', '#(\d+)', 1) AS order_number;
-- Result: '12345'

-- Extract price
SELECT REGEXP_EXTRACT('Price: $49.99', '\$(\d+\.?\d*)', 1) AS price;
-- Result: '49.99'

-- Extract percentage
SELECT REGEXP_EXTRACT('Discount: 25% off', '(\d+)%', 1) AS discount_percent;
-- Result: '25'
```

### Example 3: Extract date components
```sql
-- Extract year, month, day from ISO date
SELECT REGEXP_EXTRACT('2024-03-15', '(\d{4})-(\d{2})-(\d{2})', 1) AS year,
       REGEXP_EXTRACT('2024-03-15', '(\d{4})-(\d{2})-(\d{2})', 2) AS month,
       REGEXP_EXTRACT('2024-03-15', '(\d{4})-(\d{2})-(\d{2})', 3) AS day;
-- Result: year='2024', month='03', day='15'

-- Extract entire match (group 0)
SELECT REGEXP_EXTRACT('Date: 2024-03-15', '\d{4}-\d{2}-\d{2}', 0) AS full_date;
-- Result: '2024-03-15'
```

### Example 4: Extract from structured text
```sql
-- Extract values from key-value pairs
SELECT REGEXP_EXTRACT('user_id=12345; session_id=abc123', 'user_id=([^;]+)', 1) AS user_id;
-- Result: '12345'

-- Extract from log entries
SELECT REGEXP_EXTRACT('[2024-03-15 10:30:45] ERROR: Connection timeout', '\[([\d-]+)\s+([\d:]+)\]\s+(\w+):', 1) AS log_date,
       REGEXP_EXTRACT('[2024-03-15 10:30:45] ERROR: Connection timeout', '\[([\d-]+)\s+([\d:]+)\]\s+(\w+):', 2) AS log_time,
       REGEXP_EXTRACT('[2024-03-15 10:30:45] ERROR: Connection timeout', '\[([\d-]+)\s+([\d:]+)\]\s+(\w+):', 3) AS log_level;
-- Result: log_date='2024-03-15', log_time='10:30:45', log_level='ERROR'
```

### Example 5: Extract with complex patterns
```sql
-- Extract URL components
SELECT REGEXP_EXTRACT('https://www.example.com:8080/path/to/page?id=123', 
                     '(https?)://([^:/]+)(:\d+)?(/[^?]*)', 1) AS protocol,
       REGEXP_EXTRACT('https://www.example.com:8080/path/to/page?id=123', 
                     '(https?)://([^:/]+)(:\d+)?(/[^?]*)', 2) AS domain,
       REGEXP_EXTRACT('https://www.example.com:8080/path/to/page?id=123', 
                     '(https?)://([^:/]+)(:\d+)?(/[^?]*)', 4) AS path;
-- Result: protocol='https', domain='www.example.com', path='/path/to/page'

-- Extract version numbers
SELECT REGEXP_EXTRACT('Version 2.4.1-beta', 'Version\s+(\d+)\.(\d+)\.(\d+)(?:-(\w+))?', 0) AS full_version,
       REGEXP_EXTRACT('Version 2.4.1-beta', 'Version\s+(\d+)\.(\d+)\.(\d+)(?:-(\w+))?', 1) AS major,
       REGEXP_EXTRACT('Version 2.4.1-beta', 'Version\s+(\d+)\.(\d+)\.(\d+)(?:-(\w+))?', 2) AS minor,
       REGEXP_EXTRACT('Version 2.4.1-beta', 'Version\s+(\d+)\.(\d+)\.(\d+)(?:-(\w+))?', 3) AS patch,
       REGEXP_EXTRACT('Version 2.4.1-beta', 'Version\s+(\d+)\.(\d+)\.(\d+)(?:-(\w+))?', 4) AS release;
-- Result: full_version='Version 2.4.1-beta', major='2', minor='4', patch='1', release='beta'
```

## Common Use Cases

1. **Data Parsing**: Extract structured data from unstructured text
2. **Log Analysis**: Parse log entries to extract timestamps, levels, and messages
3. **URL Processing**: Extract domains, paths, and query parameters from URLs
4. **Data Transformation**: Convert free-form text into structured columns
5. **Information Extraction**: Pull specific values from narrative text

## Notes

- Group 0 returns the entire match
- Groups are numbered starting from 1 for the first capturing group
- Non-capturing groups `(?:...)` don't count toward group numbering
- If the pattern doesn't match, the function returns NULL
- If the specified group number doesn't exist, the function returns NULL
- Capturing groups are defined by parentheses `()` in the pattern

## See Also

- [REGEXP_CONTAINS](REGEXP_CONTAINS.md) - For checking pattern existence
- [REGEXP_REPLACE](REGEXP_REPLACE.md) - For replacing patterns
- [REGEXP_COUNT](REGEXP_COUNT.md) - For counting pattern occurrences