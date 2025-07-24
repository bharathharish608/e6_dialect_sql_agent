# Prompt: Extract SQL Functions from Source Code to Structured JSON

**Task**: Analyze a Java source file containing SQL function definitions and extract all function metadata into well-organized JSON files.

**Input**: 
- Java source file (e.g., `SqlStdOperatorTablePlus.java`) containing SQL function definitions using Apache Calcite's SqlFunction class
- Functions defined as static final fields like:
  ```java
  public static final SqlFunction FUNCTION_NAME = new SqlFunction(...);
  ```

**Required Output**:
1. Multiple JSON files, each containing ~20 functions, organized by category
2. A README.md file listing all created files

**JSON Structure for Each File**:
```json
{
  "category": "CATEGORY_NAME",
  "functions": [
    {
      "name": "FUNCTION_NAME",
      "sqlKind": "SQL_KIND_VALUE",
      "category": "FUNCTION_CATEGORY",
      "returnType": "Human-readable return type description",
      "parameters": {
        "count": "Number or range (e.g., '2', '1 or 2', 'Variable')",
        "types": "Parameter types (e.g., 'STRING, INTEGER, [STRING]')",
        "description": "Human-readable parameter description"
      },
      "description": "Clear description of what the function does",
      "aliases": ["ALIAS1", "ALIAS2"],
      "examples": [
        "SELECT FUNCTION(...) -- Comment explaining result",
        "SELECT FUNCTION(...) FROM table -- Real-world usage"
      ]
    }
  ]
}
```

**Extraction Rules**:

1. **Function Name**: Extract from the variable name (e.g., `CONCAT` from `public static final SqlFunction CONCAT`)

2. **SQL Kind**: Extract from SqlKind enum parameter (e.g., `SqlKind.OTHER_FUNCTION`)

3. **Category**: Extract from SqlFunctionCategory parameter (e.g., `SqlFunctionCategory.STRING`)

4. **Return Type**: 
   - Parse from ReturnTypes parameter (e.g., `ReturnTypes.VARCHAR_2000_NULLABLE`)
   - Convert to human-readable format (e.g., "VARCHAR(2000) NULLABLE")

5. **Parameters**:
   - Parse from OperandTypes parameter
   - Common patterns:
     - `OperandTypes.STRING` → "1 parameter, STRING type"
     - `OperandTypes.STRING_STRING` → "2 parameters, both STRING"
     - `OperandTypes.VARIADIC` → "Variable number of parameters"
     - `SqlOperandTypes.STRING_OPTIONAL_STRING` → "1 or 2 parameters, STRING and optional STRING"

6. **Aliases**: Identify functions with same SqlKind but different names

7. **Examples**: Create 2+ meaningful SQL examples showing:
   - Basic usage with literal values
   - Real-world usage with table columns
   - Include comments showing expected results

**Organization Guidelines**:

1. **Group by Category**:
   - String functions (CONCAT, SUBSTR, TRIM, etc.)
   - Date/Time functions (DATE_ADD, CURRENT_TIMESTAMP, etc.)
   - Numeric functions (math, trigonometry)
   - Array functions
   - Aggregate functions
   - Type conversion functions
   - System functions
   - Specialized functions (encoding, geospatial, etc.)

2. **File Naming**:
   - `string-functions-1.json`, `string-functions-2.json` (when > 20 functions)
   - `datetime-functions-1.json`
   - `numeric-functions-1.json`
   - `array-functions.json`
   - `aggregate-functions.json`
   - etc.

3. **Special Cases**:
   - Binary operators (PLUS, MINUS) should include operator symbol
   - Functions with custom classes need type inference
   - Window functions, table functions noted specially

**Quality Requirements**:

1. Every function must have:
   - Accurate parameter count and types
   - Clear, concise description
   - At least 2 examples
   - Proper categorization

2. Identify and group:
   - Function aliases (same behavior, different names)
   - Related functions (e.g., DATE_ADD and DATEADD)
   - Function families (e.g., all trigonometric functions)

3. Human-readable formatting:
   - Convert technical type names to readable format
   - Use consistent terminology
   - Provide context in descriptions

**Expected Output Stats**:
- Total functions: 200+
- Files created: 10-15
- Categories: ~10-15
- Functions per file: ~20

This structured approach enables:
- Easy programmatic processing
- AI-powered documentation generation
- Function discovery and search
- API development
- Testing coverage verification