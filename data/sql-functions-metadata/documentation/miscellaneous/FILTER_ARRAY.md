# FILTER_ARRAY

## Description
The `FILTER_ARRAY` function filters elements from an array based on a lambda expression predicate. It returns a new array containing only the elements for which the lambda expression evaluates to true.

## Syntax
```sql
FILTER_ARRAY(array, lambda_expression)
```

## Parameters
- `array`: The input array to filter
- `lambda_expression`: A lambda function that takes an array element as input and returns a boolean value

## Return Value
Returns an array containing only the elements that satisfy the lambda predicate. The returned array maintains the original order of elements.

## Examples

### Example 1: Filter Numbers Greater Than 5
```sql
SELECT FILTER_ARRAY(ARRAY[1, 5, 8, 3, 10, 2], x -> x > 5) AS filtered_numbers;
```
**Result:** `[8, 10]`

### Example 2: Filter Even Numbers
```sql
SELECT FILTER_ARRAY(ARRAY[1, 2, 3, 4, 5, 6, 7, 8], x -> x % 2 = 0) AS even_numbers;
```
**Result:** `[2, 4, 6, 8]`

### Example 3: Filter Strings by Length
```sql
SELECT FILTER_ARRAY(ARRAY['cat', 'elephant', 'dog', 'bird', 'rhinoceros'], 
                    word -> LENGTH(word) > 4) AS long_words;
```
**Result:** `['elephant', 'rhinoceros']`

### Example 4: Filter Complex Conditions
```sql
SELECT FILTER_ARRAY(ARRAY[10, 25, 30, 45, 50, 75, 100], 
                    n -> n >= 30 AND n <= 75) AS range_filtered;
```
**Result:** `[30, 45, 50, 75]`

### Example 5: Filter with NULL Handling
```sql
SELECT FILTER_ARRAY(ARRAY[1, NULL, 3, NULL, 5, 6], 
                    x -> x IS NOT NULL AND x > 2) AS non_null_greater_than_2;
```
**Result:** `[3, 5, 6]`

## Usage Notes
- The lambda expression is evaluated for each element in the array
- Elements for which the lambda returns `true` are included in the result
- Elements for which the lambda returns `false` or `NULL` are excluded
- If the input array is `NULL`, the function returns `NULL`
- If the array is empty, an empty array is returned
- The original array is not modified; a new array is created

## Common Use Cases
1. **Data Cleaning**: Remove null or invalid values from arrays
2. **Range Filtering**: Select values within specific ranges
3. **Pattern Matching**: Filter strings based on patterns or conditions
4. **Business Logic**: Apply complex business rules to filter data
5. **Statistical Analysis**: Filter outliers or specific data points

## Performance Considerations
- The lambda expression is executed once for each array element
- For large arrays, consider the complexity of the lambda expression
- Indexes on the underlying data are not used within the lambda evaluation

## Related Functions
- `TRANSFORM`: Apply transformations to array elements
- `ANY_MATCH`: Check if any element matches a condition
- `ALL_MATCH`: Check if all elements match a condition
- `ARRAY_CONTAINS`: Check if array contains a specific value