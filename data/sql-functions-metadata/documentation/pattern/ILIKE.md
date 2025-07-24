# ILIKE

## Description

The `ILIKE` operator performs case-insensitive pattern matching using SQL wildcards. It works exactly like `LIKE` but ignores case differences between the string and pattern. It uses the same wildcards: `%` (matches any sequence of characters) and `_` (matches any single character).

## Syntax

```sql
string ILIKE pattern
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
- Description: TRUE if the string matches the pattern (case-insensitive), FALSE otherwise

## Examples

### Example 1: Case-Insensitive Product Search
Find products containing "phone" regardless of case.

```sql
SELECT product_name, price 
FROM products 
WHERE product_name ILIKE '%phone%';
```

**Sample Results:**
```
product_name           | price
----------------------|-------
iPhone 14 Pro         | 999
Samsung Galaxy Phone  | 899
PHONECASE Premium     | 29
Smartphone Stand      | 15
```

### Example 2: Case-Insensitive Email Domain Match
Find all email variations for a domain.

```sql
SELECT email, username 
FROM users 
WHERE email ILIKE '%@EXAMPLE.COM';
```

**Sample Results:**
```
email                  | username
----------------------|----------
john@Example.com      | john_doe
ADMIN@EXAMPLE.COM     | admin
support@example.com   | support
Test@ExAmPlE.cOm     | testuser
```

### Example 3: Case-Insensitive Name Prefix
Find all names starting with "mc" or "MC".

```sql
SELECT full_name, department 
FROM employees 
WHERE full_name ILIKE 'mc%';
```

**Sample Results:**
```
full_name         | department
-----------------|------------
McDonald, John   | Sales
McBride, Sarah   | Engineering
MCCARTHY, Tom    | Marketing
Mcdonald, Lisa   | HR
```

### Example 4: Case-Insensitive File Extension
Find all image files regardless of extension case.

```sql
SELECT filename, file_size 
FROM uploads 
WHERE filename ILIKE '%.jpg' 
   OR filename ILIKE '%.png' 
   OR filename ILIKE '%.gif';
```

**Sample Results:**
```
filename          | file_size
-----------------|----------
photo.JPG        | 2048576
image.png        | 1024000
Banner.PNG       | 3072000
animation.GIF    | 512000
```

### Example 5: Case-Insensitive Pattern with Underscores
Find all codes matching pattern regardless of case.

```sql
SELECT code, description 
FROM inventory 
WHERE code ILIKE 'PR_D_%';
```

**Sample Results:**
```
code      | description
----------|-------------
PROD_A    | Product A
prod_b    | Product B
PrOd_C    | Product C
PRoD_X    | Product X
```

## Common Use Cases

1. **User Search**: Implement forgiving search that works regardless of user's capitalization
2. **Email Validation**: Match email domains case-insensitively
3. **Tag/Category Matching**: Find items by tags regardless of case variations
4. **File Operations**: Match file extensions case-insensitively
5. **Name Matching**: Find names with various capitalization styles

## Notes

- `ILIKE` is PostgreSQL-specific; other databases may use `LOWER(column) LIKE LOWER(pattern)`
- Performance is generally slower than case-sensitive `LIKE`
- Indexes on the column may not be used effectively, especially with leading wildcards
- NULL values never match any ILIKE pattern, including `'%'`
- The case-insensitive comparison uses the database's collation rules