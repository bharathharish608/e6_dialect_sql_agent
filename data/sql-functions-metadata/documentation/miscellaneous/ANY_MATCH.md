# ANY_MATCH

## Description
The `ANY_MATCH` function checks whether any element in an array satisfies a given lambda expression predicate. It returns true if at least one element matches the condition, false otherwise.

## Syntax
```sql
ANY_MATCH(array, lambda_expression)
```

## Parameters
- `array`: The input array to check
- `lambda_expression`: A lambda function that takes an array element as input and returns a boolean value

## Return Value
Returns a boolean value:
- `true` if at least one element satisfies the lambda predicate
- `false` if no elements satisfy the predicate or if the array is empty
- `NULL` if the input array is NULL

## Examples

### Example 1: Check for Values Greater Than 100
```sql
SELECT ANY_MATCH(ARRAY[10, 50, 150, 75], x -> x > 100) AS has_large_value;
```
**Result:** `true`

### Example 2: Check for Even Numbers
```sql
SELECT ANY_MATCH(ARRAY[1, 3, 5, 7, 9], x -> x % 2 = 0) AS has_even_number;
```
**Result:** `false`

### Example 3: Check String Pattern
```sql
SELECT ANY_MATCH(ARRAY['apple', 'banana', 'apricot', 'orange'], 
                 fruit -> fruit LIKE 'ap%') AS has_ap_prefix;
```
**Result:** `true`

### Example 4: Complex Condition Check
```sql
SELECT ANY_MATCH(ARRAY[5, 10, 15, 20, 25], 
                 n -> n > 10 AND n < 20) AS has_value_in_range;
```
**Result:** `true`

### Example 5: Check for NULL Values
```sql
SELECT ANY_MATCH(ARRAY[1, 2, NULL, 4, 5], 
                 x -> x IS NULL) AS contains_null;
```
**Result:** `true`

## Usage Notes
- The function uses short-circuit evaluation - it stops checking once a matching element is found
- Returns false for empty arrays
- NULL elements can be checked using IS NULL in the lambda expression
- The lambda expression should return a boolean value
- Non-boolean return values from the lambda are treated as false

## Common Use Cases
1. **Validation**: Check if data contains any invalid values
2. **Search Operations**: Determine if an array contains elements meeting specific criteria
3. **Quality Checks**: Verify if any data points exceed thresholds
4. **Existence Checks**: Check for the presence of specific patterns or values
5. **Alert Conditions**: Trigger alerts when any element meets certain conditions

## Performance Considerations
- Best case: O(1) when the first element matches
- Worst case: O(n) when no elements match or the match is at the end
- For better performance, place more likely conditions first in complex lambda expressions
- Short-circuit evaluation makes it efficient for large arrays when matches are found early

## Related Functions
- `ALL_MATCH`: Check if all elements match a condition
- `NONE_MATCH`: Check if no elements match a condition
- `FILTER_ARRAY`: Get all elements that match a condition
- `ARRAY_CONTAINS`: Check for exact value presence