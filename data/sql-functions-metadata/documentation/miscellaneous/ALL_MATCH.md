# ALL_MATCH

## Description
The `ALL_MATCH` function checks whether all elements in an array satisfy a given lambda expression predicate. It returns true only if every element matches the condition.

## Syntax
```sql
ALL_MATCH(array, lambda_expression)
```

## Parameters
- `array`: The input array to check
- `lambda_expression`: A lambda function that takes an array element as input and returns a boolean value

## Return Value
Returns a boolean value:
- `true` if all elements satisfy the lambda predicate (including empty arrays)
- `false` if at least one element does not satisfy the predicate
- `NULL` if the input array is NULL

## Examples

### Example 1: Check All Positive Numbers
```sql
SELECT ALL_MATCH(ARRAY[1, 5, 10, 15], x -> x > 0) AS all_positive;
```
**Result:** `true`

### Example 2: Verify String Length Constraint
```sql
SELECT ALL_MATCH(ARRAY['cat', 'dog', 'bird', 'fish'], 
                 word -> LENGTH(word) <= 4) AS all_short_words;
```
**Result:** `true`

### Example 3: Check Range Compliance
```sql
SELECT ALL_MATCH(ARRAY[10, 20, 30, 40, 50], 
                 n -> n >= 10 AND n <= 50) AS all_in_range;
```
**Result:** `true`

### Example 4: Validate Data Format
```sql
SELECT ALL_MATCH(ARRAY['2023-01-01', '2023-02-15', '2023-12-31'], 
                 date_str -> date_str REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$') AS all_valid_dates;
```
**Result:** `true`

### Example 5: Check Non-NULL Values
```sql
SELECT ALL_MATCH(ARRAY[1, 2, NULL, 4, 5], 
                 x -> x IS NOT NULL) AS all_non_null;
```
**Result:** `false`

## Usage Notes
- The function uses short-circuit evaluation - it stops checking once a non-matching element is found
- Returns true for empty arrays (vacuous truth)
- NULL elements can cause the predicate to return false unless explicitly handled
- The lambda expression should return a boolean value
- Non-boolean return values from the lambda are treated as false

## Common Use Cases
1. **Data Validation**: Ensure all values meet specific criteria
2. **Quality Assurance**: Verify data integrity across entire datasets
3. **Compliance Checks**: Confirm all elements adhere to business rules
4. **Input Validation**: Validate user inputs or API parameters
5. **Constraint Verification**: Check mathematical or logical constraints

## Performance Considerations
- Best case: O(1) when the first element doesn't match
- Worst case: O(n) when all elements match or the non-match is at the end
- Short-circuit evaluation makes it efficient for large arrays when non-matches are found early
- Consider the complexity of the lambda expression for large arrays

## Related Functions
- `ANY_MATCH`: Check if any element matches a condition
- `NONE_MATCH`: Check if no elements match a condition
- `FILTER_ARRAY`: Get elements that match a condition
- `EVERY`: Alternative function name for ALL_MATCH in some systems