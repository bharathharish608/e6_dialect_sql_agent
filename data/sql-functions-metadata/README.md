# E6 SQL Functions Metadata

This directory contains JSON metadata files for all SQL functions supported by the E6 distributed SQL engine. The functions are organized into logical groups with approximately 20 functions per file.

## File Organization

1. **string-functions-1.json** - Basic string manipulation functions (CONCAT, SUBSTR, TRIM, etc.)
2. **string-functions-2.json** - Advanced string functions (INSTR, SOUNDEX, pattern matching, etc.)
3. **datetime-functions-1.json** - Date and time functions (CURRENT_DATE, DATE_ADD, etc.)
4. **datetime-functions-2.json** - More date/time functions (DATE_PART, WEEKDAY, etc.)
5. **numeric-functions-1.json** - Basic math functions (arithmetic, logarithms, etc.)
6. **numeric-functions-2.json** - Trigonometric and advanced math functions
7. **array-functions.json** - Array manipulation functions
8. **conversion-system-functions.json** - Type conversion and system functions
9. **encoding-hash-functions.json** - Encoding, decoding, and hash functions
10. **aggregate-functions.json** - Aggregate functions (COUNT, SUM, AVG, etc.)
11. **regex-pattern-functions.json** - Regular expression and pattern matching
12. **geospatial-functions.json** - H3 and Bing tile geospatial functions
13. **bitwise-url-functions.json** - Bitwise operations and URL parsing
14. **miscellaneous-functions.json** - Other functions (JSON, MAP, special functions)

## JSON Structure

Each JSON file follows this structure:

```json
{
  "category": "CATEGORY_NAME",
  "functions": [
    {
      "name": "FUNCTION_NAME",
      "sqlKind": "SQL_KIND",
      "category": "FUNCTION_CATEGORY",
      "returnType": "RETURN_TYPE_DESCRIPTION",
      "parameters": {
        "count": "NUMBER_OR_RANGE",
        "types": "PARAMETER_TYPES",
        "description": "PARAMETER_DESCRIPTION"
      },
      "description": "FUNCTION_DESCRIPTION",
      "aliases": ["ALIAS1", "ALIAS2"],
      "examples": [
        "SQL_EXAMPLE_1",
        "SQL_EXAMPLE_2"
      ]
    }
  ]
}
```

## Usage

These JSON files are designed to be used for:

1. **Documentation Generation** - Feed to AI systems or documentation generators
2. **Function Discovery** - Search and browse available functions
3. **IDE Integration** - Provide autocomplete and function signatures
4. **Testing** - Ensure all functions are properly documented
5. **API Generation** - Create function metadata APIs

## Next Steps

To generate comprehensive documentation for each function:

1. Use the provided AI prompt template with each function's metadata
2. Enrich with additional examples from unit tests
3. Add performance considerations and best practices
4. Include compatibility notes with other SQL databases
5. Generate in multiple formats (Markdown, HTML, PDF)

## Statistics

- Total functions documented: ~200+
- Categories: 14
- Functions with aliases: ~30
- Aggregate functions: ~20
- String functions: ~40
- Date/Time functions: ~40
- Numeric/Math functions: ~40

## Contributing

When adding new functions:

1. Identify the appropriate category file
2. Add the function metadata following the existing structure
3. Include at least 2 meaningful examples
4. List all known aliases
5. Ensure accurate parameter and return type descriptions