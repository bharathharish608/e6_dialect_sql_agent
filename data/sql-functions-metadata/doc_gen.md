# SQL Function Documentation Generation Prompt

## Objective
Generate comprehensive documentation for SQL functions following a consistent structure and format. Each function's documentation should be complete, self-contained, and include practical examples with clear visualizations.
Make sure you get the details of the function from the json file because that represents my current engine's support
### Guideline for Usage Notes
- Dont add any entries that is not 100% clear to you.
- Never use words like etc. Always fill it in detail
- Do not make blank statements. Add only verifiable statements 

### Guideline for Expression and Target type
- Never use blank/generic information. 
- Clearly and explicitly mention all the types supported

## Documentation Structure

### 1. Function Name & Brief Description
Start with the function name in a clear heading, followed by a concise one-line description of what the function does.

### 2. Syntax
- Present the function syntax in a consistent format that aligns with the underlying SQL grammar
- Use angle brackets for required parameters: `<parameter>`
- Use square brackets for optional parameters: `[ , <parameter> ]`
- Show ellipsis for variadic parameters: `[ , <parameterN> ... ]`
- Example format: `FUNCTION_NAME( <param1> [ , <param2> ] [ , <paramN> ... ] )`

### 3. Arguments
List each argument individually with detailed explanations:
- **Parameter Name**: The exact name as it appears in the syntax
- **Data Type**: Accepted data types for this parameter
- **Required/Optional**: Clearly state if the parameter is required or optional
- **Description**: Detailed explanation of what the parameter does
- **Default Value**: If applicable, mention the default value
- **Constraints**: Any restrictions or special conditions

### 4. Returns
Clearly specify:
- **Return Type**: The data type of the returned value
- **Return Behavior**: How the function processes inputs to produce the output
- **NULL Handling**: How the function behaves with NULL inputs
- **Error Conditions**: When the function might return errors

### 5. Usage Notes
Include important details about function behavior:
- NULL handling specifics
- Performance considerations
- Alternative functions or operators that achieve similar results
- Edge cases and special behaviors
- Version compatibility notes (if applicable)
- Best practices for using the function


### 6. Examples
Provide multiple comprehensive examples demonstrating different use cases:

#### Example Format Requirements:
1. **Consistent Schema**: Use the following standard tables across all function documentation:
   - `employees` table (id, name, department, salary, hire_date, email)
   - `products` table (product_id, product_name, category, price, stock_quantity)
   - `orders` table (order_id, customer_id, product_id, quantity, order_date, status)
   - `customers` table (customer_id, first_name, last_name, email, phone, city)

2. **Example Structure**:
   - Title describing the use case
   - ASCII table representation with sample data (5-10 rows)
   - SQL query using the function
   - Result displayed in ASCII table format
   - Brief explanation of what happened

3. **ASCII Table Format**:
   ```
   +--------+------------+-------------+--------+
   | col1   | col2       | col3        | col4   |
   +--------+------------+-------------+--------+
   | value1 | value2     | value3      | value4 |
   | value1 | value2     | value3      | value4 |
   +--------+------------+-------------+--------+
   ```

4. **Multiple Examples Per Function**:
   - Basic usage example
   - Example with NULL values
   - Example with edge cases
   - Example combining with other functions
   - Example demonstrating common use cases in real scenarios

## Example Template

```markdown
# FUNCTION_NAME

Returns/performs [brief description of what the function does].

## Syntax

```sql
FUNCTION_NAME( <required_param> [ , <optional_param> ] )
```

## Arguments

### required_param
- **Type**: VARCHAR, INTEGER, etc.
- **Required**: Yes
- **Description**: Detailed explanation of what this parameter does and how it affects the function's behavior.

### optional_param
- **Type**: VARCHAR, INTEGER, etc.
- **Required**: No
- **Description**: Detailed explanation of the optional parameter.
- **Default**: Default value if not specified

## Returns

- **Type**: [Return data type]
- **Description**: [What the function returns and how it's calculated]
- **NULL Handling**: [How NULL inputs affect the output]

## Usage Notes

- [Important behavior note 1]
- [Performance consideration]
- [Alternative approaches]
- [Best practices]

## Examples

### Example 1: Basic Usage

Sample data in `employees` table:
```
+----+------------------+------------+--------+------------+----------------------+
| id | name             | department | salary | hire_date  | email                |
+----+------------------+------------+--------+------------+----------------------+
| 1  | John Doe         | Sales      | 50000  | 2020-01-15 | john.doe@company.com |
| 2  | Jane Smith       | Marketing  | 55000  | 2019-03-20 | jane.s@company.com   |
| 3  | Bob Johnson      | IT         | 60000  | 2021-06-10 | bob.j@company.com    |
+----+------------------+------------+--------+------------+----------------------+
```

Query:
```sql
SELECT id, name, FUNCTION_NAME(param1, param2) AS result
FROM employees
WHERE condition;
```

Result:
```
+----+------------------+----------+
| id | name             | result   |
+----+------------------+----------+
| 1  | John Doe         | output1  |
| 2  | Jane Smith       | output2  |
| 3  | Bob Johnson      | output3  |
+----+------------------+----------+
```

This example demonstrates [explanation of what the example shows].

### Example 2: Handling NULL Values

[Continue with more examples following the same pattern]
```

## Guidelines for Documentation Generation

1. **Consistency**: Maintain the same structure and format across all function documentation
2. **Completeness**: Include all sections even if some have minimal content
3. **Clarity**: Use clear, technical language without ambiguity
4. **Practicality**: Provide examples that reflect real-world usage
5. **Accuracy**: Ensure all syntax and examples are correct and executable
6. **Visual Clarity**: Use ASCII tables that are properly aligned and easy to read

## Additional Requirements

- Always use the standard schema tables defined above for consistency
- Include at least 3-5 examples per function
- Ensure examples progress from simple to complex use cases
- Test that all SQL examples would execute correctly
- Include performance implications where relevant
- Mention any database-specific variations if applicable