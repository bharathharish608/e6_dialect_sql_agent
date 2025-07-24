#!/usr/bin/env python3
"""
Generate SQL function documentation from JSON metadata files.
"""

import json
import os
import glob
from typing import Dict, List, Any


def get_syntax(func: Dict[str, Any]) -> str:
    """Generate function syntax based on parameter information."""
    name = func['name']
    params = func.get('parameters', {})
    count = params.get('count', '1')
    types = params.get('types', '')
    
    if count == '1':
        return f"{name}( <expression> )"
    elif count == '2':
        if 'separator' in types.lower() or name == 'CONCAT_WS':
            return f"{name}( <separator>, <string> )"
        elif name in ['TRY_CAST', 'CAST']:
            return f"{name}( <expression> AS <target_type> )"
        else:
            return f"{name}( <expression1>, <expression2> )"
    elif count == '3':
        return f"{name}( <expression1>, <expression2>, <expression3> )"
    elif count == '2 or 3':
        return f"{name}( <expression1>, <expression2> [ , <expression3> ] )"
    elif count == '1 or 2':
        return f"{name}( <expression> [ , <parameter2> ] )"
    elif 'Variable' in count or 'or more' in count:
        return f"{name}( <expression1>, <expression2> [ , <expressionN> ... ] )"
    else:
        return f"{name}( <parameters> )"


def get_parameter_details(func: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract parameter details from function metadata."""
    params = func.get('parameters', {})
    count = params.get('count', '1')
    types = params.get('types', '')
    desc = params.get('description', '')
    
    param_list = []
    
    if func['name'] in ['TRY_CAST', 'CAST']:
        param_list.append({
            'name': 'expression',
            'type': 'ANY',
            'required': 'Yes',
            'description': 'Value to cast to the target type.'
        })
        param_list.append({
            'name': 'target_type',
            'type': 'TYPE',
            'required': 'Yes',
            'description': 'Target type for the cast operation.'
        })
    elif func['name'] == 'CONCAT_WS':
        param_list.append({
            'name': 'separator',
            'type': 'STRING',
            'required': 'Yes',
            'description': 'Separator string to use between concatenated values.'
        })
        param_list.append({
            'name': 'string',
            'type': 'STRING',
            'required': 'Yes',
            'description': 'String value to concatenate.'
        })
        param_list.append({
            'name': 'stringN',
            'type': 'STRING',
            'required': 'No',
            'description': 'Additional string values to concatenate.'
        })
    elif count == '1':
        param_list.append({
            'name': 'expression',
            'type': types.split(',')[0].strip() if types else 'ANY',
            'required': 'Yes',
            'description': desc if desc else 'Input expression.'
        })
    elif count == '2':
        type_parts = types.split(',') if types else ['ANY', 'ANY']
        param_list.append({
            'name': 'expression1' if 'expression' not in desc.lower() else desc.split(',')[0].strip().lower().replace(' ', '_'),
            'type': type_parts[0].strip() if len(type_parts) > 0 else 'ANY',
            'required': 'Yes',
            'description': desc.split(',')[0].strip() if ',' in desc else 'First parameter.'
        })
        param_list.append({
            'name': 'expression2' if 'expression' not in desc.lower() else desc.split(',')[1].strip().lower().replace(' ', '_'),
            'type': type_parts[1].strip() if len(type_parts) > 1 else 'ANY',
            'required': 'Yes',
            'description': desc.split(',')[1].strip() if ',' in desc else 'Second parameter.'
        })
    elif 'Variable' in count or 'or more' in count:
        param_list.append({
            'name': 'expression1',
            'type': types.split(',')[0].strip() if types else 'ANY',
            'required': 'Yes',
            'description': 'First expression.'
        })
        param_list.append({
            'name': 'expression2',
            'type': types.split(',')[0].strip() if types else 'ANY',
            'required': 'Yes',
            'description': 'Second expression.'
        })
        param_list.append({
            'name': 'expressionN',
            'type': types.split(',')[0].strip() if types else 'ANY',
            'required': 'No',
            'description': 'Additional expressions.'
        })
    
    return param_list


def generate_examples(func: Dict[str, Any]) -> str:
    """Generate example sections for the function."""
    examples = []
    name = func['name']
    
    # Basic example
    examples.append(f"""### Example 1: Basic Usage

Sample data in `employees` table:
```
+----+------------------+------------+--------+------------+----------------------+
| id | name             | department | salary | hire_date  | email                |
+----+------------------+------------+--------+------------+----------------------+
| 1  | John Doe         | Sales      | 50000  | 2020-01-15 | john.doe@company.com |
| 2  | Jane Smith       | Marketing  | 55000  | 2019-03-20 | jane.s@company.com   |
| 3  | Bob Johnson      | IT         | 60000  | 2021-06-10 | bob.j@company.com    |
| 4  | Alice Brown      | HR         | 52000  | 2020-08-05 | alice.b@company.com  |
| 5  | Charlie Wilson   | Sales      | 48000  | 2022-02-14 | charlie.w@company.com|
+----+------------------+------------+--------+------------+----------------------+
```

Query:
```sql
{func.get('examples', ['-- Example query'])[0] if func.get('examples') else f"SELECT {name}(column) FROM employees"}
```

Result:
```
-- Results will vary based on the function
```

This example demonstrates basic usage of the {name} function.""")

    # NULL handling example
    examples.append(f"""### Example 2: NULL Handling

Sample data in `products` table:
```
+------------+------------------+------------+--------+----------------+
| product_id | product_name     | category   | price  | stock_quantity |
+------------+------------------+------------+--------+----------------+
| 1          | Laptop Pro       | Electronics| 1299.99| 50             |
| 2          | Wireless Mouse   | Electronics| NULL   | 150            |
| 3          | Office Chair     | Furniture  | 249.99 | 30             |
| 4          | Standing Desk    | Furniture  | NULL   | 20             |
| 5          | USB-C Cable      | Electronics| 19.99  | 200            |
+------------+------------------+------------+--------+----------------+
```

Query:
```sql
SELECT product_id, product_name, {name}(price) AS result
FROM products
ORDER BY product_id;
```

Result:
```
-- Results showing NULL handling
```

This example shows how {name} handles NULL values.""")

    return '\n\n'.join(examples)


def generate_function_doc(func: Dict[str, Any]) -> str:
    """Generate complete documentation for a single function."""
    name = func['name']
    description = func.get('description', 'Function description.')
    return_type = func.get('returnType', 'Type varies')
    
    # Get parameters
    params = get_parameter_details(func)
    
    # Build parameter sections
    param_sections = []
    for param in params:
        param_sections.append(f"""### {param['name']}
- **Type**: {param['type']}
- **Required**: {param['required']}
- **Description**: {param['description']}""")
    
    # Generate the complete documentation
    doc = f"""# {name}

{description}

## Syntax

```sql
{get_syntax(func)}
```

## Arguments

{chr(10).join(param_sections)}

## Returns

- **Type**: {return_type}
- **Description**: Returns the result of the {name} operation.
- **NULL Handling**: Function behavior with NULL inputs depends on the specific operation.

## Usage Notes

- {description}
- Function accepts {func.get('parameters', {}).get('types', 'various types')}
- Return type is {return_type}

## Examples

{generate_examples(func)}

### Example 3: Practical Application

Sample data in `customers` table:
```
+-------------+------------+-------------+----------------------+--------------+----------+
| customer_id | first_name | last_name   | email                | phone        | city     |
+-------------+------------+-------------+----------------------+--------------+----------+
| 1           | Michael    | Scott       | m.scott@company.com  | 555-0123     | Scranton |
| 2           | Dwight     | Schrute     | d.schrute@company.com| 555-0124     | Scranton |
| 3           | Jim        | Halpert     | j.halpert@company.com| 555-0125     | Scranton |
| 4           | Pam        | Beesly      | p.beesly@company.com | 555-0126     | Scranton |
| 5           | Stanley    | Hudson      | s.hudson@company.com | 555-0127     | Scranton |
+-------------+------------+-------------+----------------------+--------------+----------+
```

Query:
```sql
-- Practical example using {name}
SELECT customer_id, first_name, last_name
FROM customers
WHERE customer_id <= 5;
```

Result:
```
-- Results demonstrating practical usage
```

This example shows a practical application of the {name} function."""

    return doc


def get_category_folder(category: str) -> str:
    """Map category to folder name."""
    mapping = {
        'STRING': 'string',
        'NUMERIC': 'numeric',
        'TIMEDATE': 'timedate',
        'ARRAY': 'array',
        'AGGREGATE': 'aggregate',
        'SYSTEM': 'system',
        'CONVERSION_AND_SYSTEM': 'conversion',
        'ENCODING_AND_HASH': 'encoding-hash',
        'PATTERN': 'pattern',
        'REGEX_AND_PATTERN': 'pattern',
        'MISCELLANEOUS': 'miscellaneous',
        'BITWISE_AND_URL': 'bitwise-url',
        'GEOSPATIAL': 'geospatial',
        'USER_DEFINED_FUNCTION': 'miscellaneous',
        'MAP': 'miscellaneous',
        'TABLE': 'miscellaneous'
    }
    return mapping.get(category, 'miscellaneous')


def main():
    """Main function to generate all documentation."""
    json_files = glob.glob('/Users/aravindhs/e6x/docs/sql-functions-metadata/*.json')
    
    # Skip non-function files
    skip_files = ['category-summary.json']
    
    total_functions = 0
    
    for json_file in json_files:
        if os.path.basename(json_file) in skip_files:
            continue
            
        print(f"Processing {json_file}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        file_category = data.get('category', 'MISCELLANEOUS')
        functions = data.get('functions', [])
        
        for func in functions:
            # Skip aliases - they'll reference the main function
            if func['name'] in func.get('aliases', []):
                continue
                
            # Determine the category folder
            func_category = func.get('category', file_category)
            folder = get_category_folder(func_category)
            
            # Generate documentation
            doc_content = generate_function_doc(func)
            
            # Write to file
            output_path = f"/Users/aravindhs/e6x/docs/sql-functions-metadata/documentation/{folder}/{func['name']}.md"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(doc_content)
            
            total_functions += 1
            print(f"  Generated documentation for {func['name']} in {folder}/")
    
    print(f"\nTotal functions documented: {total_functions}")


if __name__ == "__main__":
    main()