# REGEXP_COUNT

## Description

`REGEXP_COUNT` counts the number of times a regular expression pattern occurs in a string. It returns the total count of non-overlapping matches found.

## Syntax

```sql
REGEXP_COUNT(string_expression, regex_pattern)
```

## Parameters

- **string_expression**: The string to search in (VARCHAR)
- **regex_pattern**: The regular expression pattern to count (VARCHAR)

## Return Type

BIGINT - Returns the number of matches found (0 if no matches)

## Examples

### Example 1: Count word occurrences
```sql
-- Count occurrences of a specific word
SELECT REGEXP_COUNT('The cat in the hat sat on the mat', '\bthe\b') AS the_count;
-- Result: 3 (case-sensitive, matches 'the' three times)

-- Case-insensitive word count
SELECT REGEXP_COUNT('The cat in the hat sat on the mat', '(?i)\bthe\b') AS the_count_ci;
-- Result: 4 (matches 'The' and 'the')

-- Count all words
SELECT REGEXP_COUNT('Hello World from SQL', '\b\w+\b') AS word_count;
-- Result: 4
```

### Example 2: Count numeric patterns
```sql
-- Count digits
SELECT REGEXP_COUNT('Order #12345 shipped on 2024-03-15', '\d') AS digit_count;
-- Result: 13 (counts individual digits)

-- Count numbers (sequences of digits)
SELECT REGEXP_COUNT('Items: 25, 30, 45, and 100', '\d+') AS number_count;
-- Result: 4

-- Count decimal numbers
SELECT REGEXP_COUNT('Prices: $10.99, $25.50, $5, $100.00', '\d+\.?\d*') AS price_count;
-- Result: 4
```

### Example 3: Count email addresses and URLs
```sql
-- Count email addresses in text
SELECT REGEXP_COUNT('Contact john@example.com or mary@company.org for support@help.com', 
                   '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}') AS email_count;
-- Result: 3

-- Count URLs
SELECT REGEXP_COUNT('Visit https://example.com and http://test.org or www.sample.net', 
                   'https?://[^\s]+|www\.[^\s]+') AS url_count;
-- Result: 3
```

### Example 4: Count pattern variations
```sql
-- Count different date formats
SELECT REGEXP_COUNT('Dates: 2024-03-15, 03/15/2024, 15-Mar-2024', 
                   '\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-[A-Za-z]{3}-\d{4}') AS date_count;
-- Result: 3

-- Count phone number variations
SELECT REGEXP_COUNT('Call (555) 123-4567 or 555.123.4567 or 555-123-4567', 
                   '\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}') AS phone_count;
-- Result: 3

-- Count different currency amounts
SELECT REGEXP_COUNT('Total: $100.50 + €75.25 + £50 = $225.75', 
                   '[$€£]\d+\.?\d*') AS currency_count;
-- Result: 4
```

### Example 5: Advanced pattern counting
```sql
-- Count sentences (ending with . ! or ?)
SELECT REGEXP_COUNT('Hello world. How are you? I am fine! Thanks.', 
                   '[^.!?]+[.!?]') AS sentence_count;
-- Result: 4

-- Count CSV fields
SELECT REGEXP_COUNT('name,age,city,country', '[^,]+') AS field_count;
-- Result: 4

-- Count repeated characters
SELECT REGEXP_COUNT('Mississippi', 's{2}') AS double_s_count;
-- Result: 2

-- Count lines in multi-line text
SELECT REGEXP_COUNT('Line 1\nLine 2\nLine 3\nLine 4', '^.+$') AS line_count;
-- Result: 1 (without multiline flag)

-- Count specific tag occurrences in HTML
SELECT REGEXP_COUNT('<div>Hello</div><div>World</div><span>Test</span>', '<div>') AS div_count;
-- Result: 2
```

## Common Use Cases

1. **Text Analysis**: Count word frequencies, sentence counts, or specific terms
2. **Data Validation**: Verify expected number of pattern occurrences
3. **Log Analysis**: Count error occurrences, specific events, or patterns
4. **Quality Checks**: Ensure data meets format requirements (e.g., correct number of fields)
5. **Content Analysis**: Analyze text for specific patterns or keywords

## Notes

- The function counts non-overlapping matches
- Empty matches are not counted
- The pattern is applied globally across the entire string
- For overlapping patterns, only the first match at each position is counted
- Use word boundaries `\b` to match whole words only
- Performance may vary with complex patterns on large strings

## See Also

- [REGEXP_CONTAINS](REGEXP_CONTAINS.md) - For checking if a pattern exists
- [REGEXP_EXTRACT](REGEXP_EXTRACT.md) - For extracting matching substrings
- [REGEXP_REPLACE](REGEXP_REPLACE.md) - For replacing pattern occurrences