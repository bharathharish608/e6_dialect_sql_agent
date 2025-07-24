# LIKE

## Description

The `LIKE` operator performs pattern matching using SQL wildcards. It compares a string value against a pattern containing special wildcard characters: `%` (matches any sequence of characters) and `_` (matches any single character). The comparison is case-sensitive.

## Syntax

```sql
string LIKE pattern
```

## Parameters

- `string` (VARCHAR): The string value to test
- `pattern` (VARCHAR): The pattern to match against, containing wildcards

## Wildcards

- `%` - Matches any sequence of zero or more characters
- `_` - Matches exactly one character
- `\` - Escape character (use `\\` to match literal `\`, `\%` for literal `%`, `\_` for literal `_`)

## Returns

- Type: `BOOLEAN`
- Description: TRUE if the string matches the pattern, FALSE otherwise

## Examples

### Example 1: Match Beginning of String
Find all products starting with "Apple".

```sql
SELECT product_name 
FROM products 
WHERE product_name LIKE 'Apple%';
```

**Sample Results:**
```
product_name
------------
Apple iPhone 14
Apple MacBook Pro
Apple Watch Series 8
```

### Example 2: Match End of String
Find all email addresses ending with "gmail.com".

```sql
SELECT email 
FROM users 
WHERE email LIKE '%gmail.com';
```

**Sample Results:**
```
email
-----
john.doe@gmail.com
jane.smith@gmail.com
admin@gmail.com
```

### Example 3: Match Pattern with Single Character Wildcard
Find all 5-letter words starting with 'S' and ending with 'E'.

```sql
SELECT word 
FROM dictionary 
WHERE word LIKE 'S___E';
```

**Sample Results:**
```
word
----
STONE
SPACE
STAKE
```

### Example 4: Match Pattern Anywhere in String
Find all addresses containing "Main St".

```sql
SELECT address 
FROM locations 
WHERE address LIKE '%Main St%';
```

**Sample Results:**
```
address
-------
123 Main Street, Suite 100
456 E Main St
789 Main St Apt 5B
```

### Example 5: Escape Special Characters
Find filenames containing literal underscore.

```sql
SELECT filename 
FROM files 
WHERE filename LIKE '%\_%';
```

**Sample Results:**
```
filename
--------
user_profile.jpg
report_2023_Q4.pdf
backup_database.sql
```

## Common Use Cases

1. **Search Filters**: Implement user-friendly search functionality
2. **Data Validation**: Check if values follow specific patterns
3. **Categorization**: Group data based on naming conventions
4. **File Management**: Filter files by extension or naming pattern
5. **Database Queries**: Find records with partial matches

## Notes

- `LIKE` is case-sensitive; use `ILIKE` for case-insensitive matching
- Performance can be slower than exact matches, especially with leading wildcards (`%pattern`)
- Indexes may not be used effectively with leading wildcard patterns
- NULL values never match any LIKE pattern, including `'%'`
- To match literal `%` or `_` characters, escape them with backslash (`\%`, `\_`)