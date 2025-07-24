# REGEXP_EXTRACT_ALL

## Description

The `REGEXP_EXTRACT_ALL` function extracts all non-overlapping substrings from a string that match a regular expression pattern. It returns an array containing all matches. When a capture group is specified, it returns the captured groups instead of the full matches.

## Syntax

```sql
REGEXP_EXTRACT_ALL(string, pattern)
REGEXP_EXTRACT_ALL(string, pattern, group)
```

## Parameters

- `string` (VARCHAR): The input string to search for matches
- `pattern` (VARCHAR): The regular expression pattern to match against
- `group` (INTEGER, optional): The capture group number to extract (0 for entire match, 1 for first group, etc.)

## Returns

- Type: `ARRAY<VARCHAR>`
- Description: An array containing all matched substrings or captured groups

## Examples

### Example 1: Extract All Words
Extract all words from a sentence.

```sql
SELECT REGEXP_EXTRACT_ALL('The quick brown fox jumps over the lazy dog', '\w+') AS words;
```

**Result:**
```
words
-----
['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

### Example 2: Extract All Numbers
Extract all numeric values from a string.

```sql
SELECT REGEXP_EXTRACT_ALL('Order #12345: 3 items at $19.99 each, total: $59.97', '\d+\.?\d*') AS numbers;
```

**Result:**
```
numbers
-------
['12345', '3', '19.99', '59.97']
```

### Example 3: Extract Email Addresses
Extract all email addresses from text.

```sql
SELECT REGEXP_EXTRACT_ALL(
    'Contact us at support@example.com or sales@example.org for assistance',
    '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
) AS emails;
```

**Result:**
```
emails
------
['support@example.com', 'sales@example.org']
```

### Example 4: Extract Captured Groups
Extract domain names from URLs using capture groups.

```sql
SELECT REGEXP_EXTRACT_ALL(
    'Visit https://www.example.com and http://docs.example.org',
    'https?://(?:www\.)?([a-zA-Z0-9.-]+)',
    1
) AS domains;
```

**Result:**
```
domains
-------
['example.com', 'docs.example.org']
```

### Example 5: Extract Hashtags
Extract all hashtags from social media text.

```sql
SELECT REGEXP_EXTRACT_ALL(
    'Check out our #newproduct launch! #innovation #tech #startup',
    '#(\w+)',
    1
) AS hashtags;
```

**Result:**
```
hashtags
--------
['newproduct', 'innovation', 'tech', 'startup']
```

## Common Use Cases

1. **Text Mining**: Extract specific patterns like phone numbers, postal codes, or IDs
2. **Log Analysis**: Extract timestamps, IP addresses, or error codes from log files
3. **Data Validation**: Find all instances of invalid data patterns
4. **Content Parsing**: Extract mentions, hashtags, or URLs from social media content
5. **Data Transformation**: Split complex strings into structured arrays

## Notes

- The function returns an empty array if no matches are found
- Regular expression syntax follows standard POSIX or PCRE patterns (implementation-dependent)
- When using capture groups, group 0 represents the entire match
- The matches are non-overlapping; once a portion of the string is matched, it won't be included in subsequent matches
- Case sensitivity depends on the regex pattern (use `(?i)` for case-insensitive matching)