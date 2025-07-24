# TRANSFORM

## Description
The `TRANSFORM` function applies a lambda expression to each element of an array, creating a new array with the transformed values. It's equivalent to the "map" operation in functional programming.

## Syntax
```sql
TRANSFORM(array, lambda_expression)
```

## Parameters
- `array`: The input array to transform
- `lambda_expression`: A lambda function that takes an array element as input and returns the transformed value

## Return Value
Returns a new array where each element is the result of applying the lambda expression to the corresponding element in the input array. The size of the returned array is the same as the input array.

## Examples

### Example 1: Square Numbers
```sql
SELECT TRANSFORM(ARRAY[1, 2, 3, 4, 5], x -> x * x) AS squared_numbers;
```
**Result:** `[1, 4, 9, 16, 25]`

### Example 2: Convert to Uppercase
```sql
SELECT TRANSFORM(ARRAY['hello', 'world', 'sql'], word -> UPPER(word)) AS uppercase_words;
```
**Result:** `['HELLO', 'WORLD', 'SQL']`

### Example 3: Complex Calculations
```sql
SELECT TRANSFORM(ARRAY[10, 20, 30, 40], x -> (x * 1.5) + 10) AS calculated_values;
```
**Result:** `[25.0, 40.0, 55.0, 70.0]`

### Example 4: String Concatenation
```sql
SELECT TRANSFORM(ARRAY['file1', 'file2', 'file3'], 
                 name -> CONCAT(name, '.txt')) AS filenames;
```
**Result:** `['file1.txt', 'file2.txt', 'file3.txt']`

### Example 5: Conditional Transformation
```sql
SELECT TRANSFORM(ARRAY[-2, -1, 0, 1, 2, 3], 
                 x -> CASE WHEN x < 0 THEN 0 
                          WHEN x > 2 THEN 2 
                          ELSE x END) AS clamped_values;
```
**Result:** `[0, 0, 0, 1, 2, 2]`

## Usage Notes
- The lambda expression is applied to each element independently
- The transformation preserves the array order
- NULL elements in the input array can be transformed like any other value
- If the input array is NULL, the function returns NULL
- The output array has the same cardinality as the input array
- Different data types can be returned by the lambda expression

## Common Use Cases
1. **Data Formatting**: Format values for display (e.g., adding currency symbols)
2. **Mathematical Operations**: Apply calculations to numeric arrays
3. **String Manipulation**: Process text data in bulk
4. **Data Type Conversion**: Convert array elements to different types
5. **Business Logic Application**: Apply complex transformations based on business rules

## Performance Considerations
- Each element transformation is independent, making it suitable for parallel processing
- The complexity of the lambda expression affects overall performance
- For simple transformations, consider using built-in array functions when available

## Related Functions
- `FILTER_ARRAY`: Filter array elements based on conditions
- `REDUCE`: Aggregate array elements into a single value
- `ZIP_WITH`: Transform elements from multiple arrays together
- `ARRAY_APPLY`: Similar transformation capabilities