#!/usr/bin/env python3
"""
Process all SQL functions and prepare them for documentation generation.
"""

import json
import glob
import os

def main():
    json_files = glob.glob('/Users/aravindhs/e6x/docs/sql-functions-metadata/*.json')
    
    # Skip non-function files
    skip_files = ['category-summary.json', 'generate_docs.py', 'process_all_functions.py']
    
    all_functions = []
    
    for json_file in json_files:
        if any(skip in json_file for skip in skip_files):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        file_category = data.get('category', 'MISCELLANEOUS')
        functions = data.get('functions', [])
        
        for func in functions:
            # Skip aliases - we'll only document the main function
            is_alias = False
            for other_func in functions:
                if func['name'] in other_func.get('aliases', []):
                    is_alias = True
                    break
            
            if is_alias:
                continue
                
            # Determine the category folder
            func_category = func.get('category', file_category)
            
            # Map categories to folder names
            category_map = {
                'STRING': 'string',
                'NUMERIC': 'numeric',
                'TIMEDATE': 'timedate',
                'ARRAY': 'array',
                'AGGREGATE': 'aggregate',
                'SYSTEM': 'system',
                'ENCODING_AND_HASH': 'encoding-hash',
                'PATTERN': 'pattern',
                'USER_DEFINED_FUNCTION': 'miscellaneous',
                'MAP': 'miscellaneous',
                'TABLE': 'miscellaneous',
                'BITWISE_AND_URL': 'bitwise-url',
                'GEOSPATIAL': 'geospatial',
                'MISCELLANEOUS': 'miscellaneous'
            }
            
            folder = category_map.get(func_category, 'miscellaneous')
            
            all_functions.append({
                'name': func['name'],
                'folder': folder,
                'metadata': func,
                'source_file': os.path.basename(json_file)
            })
    
    # Save the complete list
    with open('/Users/aravindhs/e6x/docs/sql-functions-metadata/all_functions.json', 'w') as f:
        json.dump(all_functions, f, indent=2)
    
    print(f"Total functions to document: {len(all_functions)}")
    
    # Group by folder
    by_folder = {}
    for func in all_functions:
        folder = func['folder']
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(func['name'])
    
    print("\nFunctions by category:")
    for folder, funcs in sorted(by_folder.items()):
        print(f"  {folder}: {len(funcs)} functions")
        

if __name__ == "__main__":
    main()