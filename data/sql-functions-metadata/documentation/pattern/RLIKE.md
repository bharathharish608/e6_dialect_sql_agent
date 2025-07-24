# RLIKE

## Description

The `RLIKE` operator performs pattern matching using regular expressions. It's functionally equivalent to `REGEXP_LIKE` and returns TRUE if the string matches the regular expression pattern. Unlike `LIKE`, it supports the full power of regular expressions for complex pattern matching.

## Syntax

```sql
string RLIKE pattern
```

## Parameters

- `string` (VARCHAR): The string value to test against the pattern
- `pattern` (VARCHAR): The regular expression pattern to match

## Returns

- Type: `BOOLEAN`
- Description: TRUE if the string matches the regular expression pattern, FALSE otherwise

## Examples

### Example 1: Match Valid Email Addresses
Validate email addresses using regex pattern.

```sql
SELECT email, 
       email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' AS is_valid
FROM user_emails;
```

**Sample Results:**
```
email                    | is_valid
------------------------|----------
john.doe@example.com    | true
invalid.email@         | false
user@domain.co.uk      | true
@example.com           | false
test.user+tag@gmail.com| true
```

### Example 2: Match Phone Number Patterns
Find various phone number formats.

```sql
SELECT phone_number,
       phone_number RLIKE '^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$' AS is_valid_phone
FROM contacts;
```

**Sample Results:**
```
phone_number      | is_valid_phone
-----------------|---------------
(555) 123-4567   | true
+1-555-123-4567  | true
555.123.4567     | true
12345            | false
555 123 4567     | true
```

### Example 3: Match IP Addresses
Validate IPv4 addresses.

```sql
SELECT server_ip,
       server_ip RLIKE '^([0-9]{1,3}\.){3}[0-9]{1,3}$' AS is_ip_format
FROM servers
WHERE server_ip RLIKE '^([0-9]{1,3}\.){3}[0-9]{1,3}$';
```

**Sample Results:**
```
server_ip       | is_ip_format
----------------|-------------
192.168.1.1     | true
10.0.0.255      | true
172.16.0.1      | true
256.1.1.1       | true
8.8.8.8         | true
```

### Example 4: Match Credit Card Patterns
Identify potential credit card numbers.

```sql
SELECT transaction_text,
       transaction_text RLIKE '[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}' AS contains_cc
FROM transaction_logs;
```

**Sample Results:**
```
transaction_text                          | contains_cc
-----------------------------------------|------------
Payment with card 4111-1111-1111-1111    | true
Order #12345 processed                   | false
CC: 4222 2222 2222 2222 approved       | true
Transaction ID: 9876543210              | false
Used card ending in 1234                | false
```

### Example 5: Match URL Patterns
Find valid URLs in text.

```sql
SELECT message,
       message RLIKE 'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\\s]*)?' AS contains_url
FROM comments;
```

**Sample Results:**
```
message                                          | contains_url
------------------------------------------------|-------------
Check out https://www.example.com/products      | true
Visit our site at http://demo.co               | true
Email us at support@example.com                | false
Go to https://github.com/user/repo for code   | true
The meeting is at 3:00 PM                      | false
```

## Common Use Cases

1. **Data Validation**: Validate formats for emails, phones, SSNs, etc.
2. **Pattern Detection**: Find specific patterns in logs or text
3. **Data Quality**: Identify records that don't match expected formats
4. **Security Scanning**: Detect potential sensitive data patterns
5. **Text Analysis**: Complex pattern matching beyond simple wildcards

## Notes

- `RLIKE` is equivalent to `REGEXP_LIKE` and `REGEXP` in many databases
- Regular expression syntax may vary by database implementation (POSIX vs PCRE)
- More powerful but potentially slower than `LIKE` for simple patterns
- Case sensitivity depends on the database and collation settings
- Use anchors (`^` and `$`) to match entire strings rather than substrings
- NULL values always return NULL (not TRUE or FALSE) when used with RLIKE