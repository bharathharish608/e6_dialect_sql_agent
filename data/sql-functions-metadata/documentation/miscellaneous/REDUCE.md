# REDUCE

## Description
The `REDUCE` function applies a lambda expression cumulatively to the elements of an array, reducing it to a single value. It's similar to fold or reduce operations in functional programming, allowing complex aggregations beyond simple sum or average.

## Syntax
```sql
REDUCE(array, initial_value, lambda_expression)
```

## Parameters
- `array`: The input array to reduce
- `initial_value`: The starting value for the reduction
- `lambda_expression`: A lambda function that takes two parameters (accumulator, current_element) and returns the new accumulator value

## Return Value
Returns a single value of the type determined by the lambda expression and initial value. The final result after processing all array elements.

## Examples

### Example 1: Sum of Array Elements
```sql
SELECT REDUCE(ARRAY[1, 2, 3, 4, 5], 0, (acc, x) -> acc + x) AS total_sum;
```
**Result:** `15`

### Example 2: Product of Array Elements
```sql
SELECT REDUCE(ARRAY[2, 3, 4, 5], 1, (acc, x) -> acc * x) AS product;
```
**Result:** `120`

### Example 3: String Concatenation
```sql
SELECT REDUCE(ARRAY['Hello', 'World', 'from', 'SQL'], 
              '', 
              (acc, word) -> CONCAT(acc, ' ', word)) AS sentence;
```
**Result:** `' Hello World from SQL'`

### Example 4: Find Maximum Value
```sql
SELECT REDUCE(ARRAY[15, 8, 23, 42, 4, 16], 
              -2147483648, 
              (max_val, x) -> CASE WHEN x > max_val THEN x ELSE max_val END) AS maximum;
```
**Result:** `42`

### Example 5: Count Specific Elements
```sql
SELECT REDUCE(ARRAY['apple', 'banana', 'apple', 'cherry', 'apple'], 
              0, 
              (count, fruit) -> CASE WHEN fruit = 'apple' THEN count + 1 ELSE count END) AS apple_count;
```
**Result:** `3`

## Usage Notes
- The accumulator is initialized with the `initial_value`
- The lambda expression is called for each element in order
- The first parameter of the lambda is always the accumulator
- The second parameter is the current array element
- For empty arrays, the initial value is returned
- If the input array is NULL, the function returns NULL
- The accumulator can be of a different type than the array elements

## Common Use Cases
1. **Custom Aggregations**: Implement complex aggregation logic not available in standard functions
2. **Running Calculations**: Compute running totals, products, or other cumulative values
3. **String Building**: Construct formatted strings from array elements
4. **Statistical Computations**: Calculate variance, standard deviation, or custom metrics
5. **Data Transformation**: Convert arrays to different data structures

## Performance Considerations
- Processing is sequential - elements are processed in array order
- Cannot be parallelized due to dependency on previous results
- Lambda complexity directly impacts performance
- For simple operations like sum or max, consider using built-in aggregate functions

## Advanced Examples

### Custom Object Building
```sql
-- Build a JSON object from key-value pairs
SELECT REDUCE(ARRAY['a:1', 'b:2', 'c:3'], 
              '{}', 
              (json, pair) -> JSON_SET(json, 
                                      CONCAT('$.', SPLIT(pair, ':')[1]), 
                                      SPLIT(pair, ':')[2])) AS json_object;
```

### Running Average Calculation
```sql
-- Calculate cumulative average (would need array index access)
SELECT REDUCE(ARRAY[10, 20, 30, 40], 
              ARRAY[0, 0], 
              (state, val) -> ARRAY[state[1] + val, state[2] + 1]) AS sum_and_count;
```

## Related Functions
- `ARRAY_AGG`: Aggregate values into an array
- `TRANSFORM`: Transform each element independently
- `FILTER_ARRAY`: Filter elements based on conditions
- `FOLD`: Alternative name for REDUCE in some systems
- `AGGREGATE`: Similar functionality with different syntax