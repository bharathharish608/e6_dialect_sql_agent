# REGEXP_SPLIT

## Description

The `REGEXP_SPLIT` function splits a string into an array of substrings using a regular expression pattern as the delimiter. The pattern matches are used as separators and are not included in the resulting array.

## Syntax

```sql
REGEXP_SPLIT(string, pattern)
```

## Parameters

- `string` (VARCHAR): The input string to split
- `pattern` (VARCHAR): The regular expression pattern to use as delimiter

## Returns

- Type: `ARRAY<VARCHAR>`
- Description: An array of substrings split by the pattern

## Examples

### Example 1: Split by Multiple Delimiters
Split a string using multiple delimiter characters.

```sql
SELECT REGEXP_SPLIT('apple,banana;orange|grape', '[,;|]') AS fruits;
```

**Result:**
```
fruits
------
['apple', 'banana', 'orange', 'grape']
```

### Example 2: Split by Whitespace
Split text by any whitespace character (spaces, tabs, newlines).

```sql
SELECT REGEXP_SPLIT('Hello    World	Tab
Newline', '\s+') AS words;
```

**Result:**
```
words
-----
['Hello', 'World', 'Tab', 'Newline']
```

### Example 3: Split Camel Case String
Split a camelCase or PascalCase string into words.

```sql
SELECT REGEXP_SPLIT('getUserAccountDetails', '(?=[A-Z])') AS words;
```

**Result:**
```
words
-----
['get', 'User', 'Account', 'Details']
```

### Example 4: Split Version Numbers
Split version numbers or dotted notation.

```sql
SELECT REGEXP_SPLIT('192.168.1.100', '\.') AS ip_parts;
```

**Result:**
```
ip_parts
--------
['192', '168', '1', '100']
```

### Example 5: Split Mixed Delimiters with Numbers
Split a string containing various delimiters and preserve numbers.

```sql
SELECT REGEXP_SPLIT('item1-qty:5,item2-qty:3;item3-qty:10', '[-:,;]') AS parts;
```

**Result:**
```
parts
-----
['item1', 'qty', '5', 'item2', 'qty', '3', 'item3', 'qty', '10']
```

## Common Use Cases

1. **CSV/TSV Processing**: Split comma or tab-separated values
2. **Path Parsing**: Split file paths or URLs into components
3. **Log Parsing**: Split log entries by delimiters
4. **Text Tokenization**: Break text into words or tokens
5. **Data Cleaning**: Split concatenated fields into separate values

## Notes

- Empty strings may be included in the result if the pattern matches at the beginning, end, or creates consecutive matches
- If the pattern doesn't match anywhere in the string, the entire string is returned as a single-element array
- The delimiter pattern itself is not included in the resulting array
- Use `\` to escape special regex characters when matching literal characters
- For simple single-character splits, this function may have more overhead than simpler split functions