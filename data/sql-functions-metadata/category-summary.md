# SQL Functions Category Summary

This document provides a comprehensive summary of all categories found in the SQL functions metadata JSON files.

## File-Level Categories

The following categories are used at the file level (top-level "category" field in each JSON file):

1. **AGGREGATE** - aggregate-functions.json
2. **ARRAY** - array-functions.json
3. **BITWISE_AND_URL** - bitwise-url-functions.json
4. **CONVERSION_AND_SYSTEM** - conversion-system-functions.json
5. **ENCODING_AND_HASH** - encoding-hash-functions.json
6. **GEOSPATIAL** - geospatial-functions.json
7. **MISCELLANEOUS** - miscellaneous-functions.json
8. **NUMERIC** - numeric-functions-1.json, numeric-functions-2.json
9. **REGEX_AND_PATTERN** - regex-pattern-functions.json
10. **STRING** - string-functions-1.json, string-functions-2.json
11. **TIMEDATE** - datetime-functions-1.json, datetime-functions-2.json

## Function-Level Categories

Within each file, individual functions are categorized using the following categories:

### 1. AGGREGATE
- Used for aggregation functions like COUNT, SUM, AVG, MIN, MAX, etc.
- File: aggregate-functions.json
- Example functions: COUNT, SUM, AVG, MIN, MAX, STRING_AGG, ARRAY_AGG, ARG_MAX, ARG_MIN

### 2. ARRAY
- Used for array manipulation functions
- File: array-functions.json
- Example functions: ARRAY_VALUE_CONSTRUCTOR, ARRAY_SLICE, ARRAY_SORT, ARRAY_APPEND

### 3. NUMERIC
- Used for numeric/mathematical operations
- Files: numeric-functions-1.json, numeric-functions-2.json, bitwise-url-functions.json, encoding-hash-functions.json, geospatial-functions.json
- Example functions: PLUS, MINUS, DIVIDE, POW, MOD, COS, SIN, TAN, BITWISE_AND, MD5, SHA256

### 4. STRING
- Used for string manipulation functions
- Files: string-functions-1.json, string-functions-2.json, encoding-hash-functions.json, conversion-system-functions.json
- Example functions: CONCAT, SUBSTR, UPPER, LOWER, REPLACE, SPLIT, FROM_BASE64, TO_CHAR

### 5. USER_DEFINED_FUNCTION
- Used for user-defined or custom functions
- Found across multiple files
- Example functions: ARRAY_SLICE, BITWISE_OR, TO_BOOLEAN, NOW, UUID, E, INFINITY

### 6. TIMEDATE
- Used for date and time functions
- Files: datetime-functions-1.json, datetime-functions-2.json, conversion-system-functions.json
- Example functions: CURRENT_DATE, DATE_ADD, DATE_DIFF, TO_DATE, TO_TIMESTAMP

### 7. SYSTEM
- Used for system-level functions
- Files: conversion-system-functions.json, miscellaneous-functions.json
- Example functions: CAST, TRY_CAST, COALESCE, CASE, IF, VERSION, SLEEP, JSON_FORMAT

### 8. PATTERN
- Used for pattern matching functions
- File: regex-pattern-functions.json
- Example functions: LIKE, NOT LIKE, ILIKE, RLIKE

### 9. MAP
- Used for map/dictionary operations
- File: miscellaneous-functions.json
- Example functions: MAP_KEYS, MAP_VALUES, MAP_CONCAT

### 10. TABLE
- Used for table-generating functions
- File: miscellaneous-functions.json
- Example functions: EXPLODE

## Category Distribution by File

| File | File-Level Category | Function-Level Categories Used |
|------|-------------------|------------------------------|
| aggregate-functions.json | AGGREGATE | AGGREGATE |
| array-functions.json | ARRAY | ARRAY, USER_DEFINED_FUNCTION |
| bitwise-url-functions.json | BITWISE_AND_URL | NUMERIC, USER_DEFINED_FUNCTION |
| conversion-system-functions.json | CONVERSION_AND_SYSTEM | SYSTEM, STRING, NUMERIC, TIMEDATE, USER_DEFINED_FUNCTION |
| datetime-functions-1.json | TIMEDATE | TIMEDATE, USER_DEFINED_FUNCTION |
| datetime-functions-2.json | TIMEDATE | TIMEDATE, USER_DEFINED_FUNCTION |
| encoding-hash-functions.json | ENCODING_AND_HASH | NUMERIC, USER_DEFINED_FUNCTION, STRING |
| geospatial-functions.json | GEOSPATIAL | NUMERIC, USER_DEFINED_FUNCTION |
| miscellaneous-functions.json | MISCELLANEOUS | USER_DEFINED_FUNCTION, SYSTEM, MAP, TABLE |
| numeric-functions-1.json | NUMERIC | NUMERIC, USER_DEFINED_FUNCTION |
| numeric-functions-2.json | NUMERIC | NUMERIC, USER_DEFINED_FUNCTION |
| regex-pattern-functions.json | REGEX_AND_PATTERN | USER_DEFINED_FUNCTION, PATTERN |
| string-functions-1.json | STRING | STRING, NUMERIC, USER_DEFINED_FUNCTION |
| string-functions-2.json | STRING | STRING, USER_DEFINED_FUNCTION |

## Summary Statistics

- **Total unique file-level categories**: 11
- **Total unique function-level categories**: 10
- **Most common function-level category**: USER_DEFINED_FUNCTION (appears in 12 files)
- **Files with single function category**: aggregate-functions.json (only AGGREGATE)
- **Files with most diverse categories**: conversion-system-functions.json (5 different categories)

## Notes

1. The USER_DEFINED_FUNCTION category is used extensively across all files, suggesting it's a catch-all for functions that don't fit neatly into other categories.
2. Some functions logically belong to multiple categories (e.g., LEN/LENGTH are categorized as NUMERIC but are string functions).
3. The file-level categories generally group related functionality, while function-level categories provide more specific classification.