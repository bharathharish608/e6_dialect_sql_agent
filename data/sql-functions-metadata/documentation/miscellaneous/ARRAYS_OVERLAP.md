# ARRAYS_OVERLAP

Checks if two arrays have any elements in common. This function is useful for finding shared values between two arrays, such as checking if a user has any of the required permissions or if product categories overlap.

## Syntax

```sql
ARRAYS_OVERLAP( <array1>, <array2> )
```

## Arguments

### array1
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The first array to compare. Can be any array type (INTEGER[], VARCHAR[], etc.)

### array2
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The second array to compare. Must be the same type as array1.

## Returns

- **Type**: BOOLEAN
- **Description**: Returns TRUE if the arrays have at least one element in common, FALSE if they have no common elements.
- **NULL Handling**: Returns NULL if either array is NULL. Empty arrays return FALSE when compared with non-empty arrays.

## Usage Notes

- Both arrays must be of the same element type
- The comparison is case-sensitive for string arrays
- Order of elements doesn't matter for the overlap check
- Duplicate elements within an array don't affect the result
- Efficient for permission checks, tag matching, and category filtering

## Examples

### Example 1: Employee Skills Matching

Sample data in `employee_skills` table:
```
+-------------+------------------+---------------------------------------+------------------------+
| employee_id | employee_name    | skills                                | required_for_project   |
+-------------+------------------+---------------------------------------+------------------------+
| 1           | John Doe         | ['Java', 'Python', 'SQL']            | ['Python', 'Docker']   |
| 2           | Jane Smith       | ['JavaScript', 'React', 'Node.js']   | ['Python', 'Docker']   |
| 3           | Bob Johnson      | ['Python', 'Docker', 'Kubernetes']   | ['Python', 'Docker']   |
| 4           | Alice Brown      | ['Java', 'Spring', 'Microservices'] | ['Python', 'Docker']   |
| 5           | Charlie Wilson   | ['SQL', 'Python', 'Tableau']         | ['Python', 'Docker']   |
+-------------+------------------+---------------------------------------+------------------------+
```

Query:
```sql
SELECT 
    employee_id,
    employee_name,
    skills,
    required_for_project,
    ARRAYS_OVERLAP(skills, required_for_project) AS has_required_skills
FROM employee_skills
ORDER BY employee_id;
```

Result:
```
+-------------+------------------+---------------------------------------+------------------------+--------------------+
| employee_id | employee_name    | skills                                | required_for_project   | has_required_skills|
+-------------+------------------+---------------------------------------+------------------------+--------------------+
| 1           | John Doe         | ['Java', 'Python', 'SQL']            | ['Python', 'Docker']   | TRUE               |
| 2           | Jane Smith       | ['JavaScript', 'React', 'Node.js']   | ['Python', 'Docker']   | FALSE              |
| 3           | Bob Johnson      | ['Python', 'Docker', 'Kubernetes']   | ['Python', 'Docker']   | TRUE               |
| 4           | Alice Brown      | ['Java', 'Spring', 'Microservices'] | ['Python', 'Docker']   | FALSE              |
| 5           | Charlie Wilson   | ['SQL', 'Python', 'Tableau']         | ['Python', 'Docker']   | TRUE               |
+-------------+------------------+---------------------------------------+------------------------+--------------------+
```

This example identifies employees who have at least one of the required skills for a project.

### Example 2: Product Category Overlap for Recommendations

Sample data in `customer_purchases` and `product_catalog` tables:
```
-- customer_purchases table
+-------------+------------------+----------------------------------------+
| customer_id | customer_name    | purchased_categories                   |
+-------------+------------------+----------------------------------------+
| 101         | Sarah Miller     | ['Electronics', 'Books', 'Sports']    |
| 102         | Tom Anderson     | ['Clothing', 'Shoes', 'Accessories']  |
| 103         | Lisa Chen        | ['Books', 'Music', 'Movies']          |
| 104         | Mike Davis       | ['Electronics', 'Gaming', 'Computers']|
| 105         | Emma Wilson      | ['Home', 'Garden', 'Kitchen']         |
+-------------+------------------+----------------------------------------+

-- product_catalog table
+------------+-------------------------+------------------------------------+--------+
| product_id | product_name           | categories                         | price  |
+------------+-------------------------+------------------------------------+--------+
| 1001       | Wireless Headphones    | ['Electronics', 'Music', 'Sports'] | 89.99  |
| 1002       | Running Shoes          | ['Shoes', 'Sports', 'Fitness']     | 129.99 |
| 1003       | Programming Book       | ['Books', 'Computers', 'Education']| 49.99  |
| 1004       | Smart Watch            | ['Electronics', 'Sports', 'Health']| 299.99 |
| 1005       | Coffee Maker           | ['Kitchen', 'Home', 'Appliances'] | 79.99  |
+------------+-------------------------+------------------------------------+--------+
```

Query:
```sql
SELECT 
    c.customer_id,
    c.customer_name,
    p.product_id,
    p.product_name,
    p.price,
    ARRAYS_OVERLAP(c.purchased_categories, p.categories) AS might_be_interested
FROM customer_purchases c
CROSS JOIN product_catalog p
WHERE ARRAYS_OVERLAP(c.purchased_categories, p.categories) = TRUE
ORDER BY c.customer_id, p.product_id;
```

Result:
```
+-------------+------------------+------------+-------------------------+--------+--------------------+
| customer_id | customer_name    | product_id | product_name           | price  | might_be_interested|
+-------------+------------------+------------+-------------------------+--------+--------------------+
| 101         | Sarah Miller     | 1001       | Wireless Headphones    | 89.99  | TRUE               |
| 101         | Sarah Miller     | 1002       | Running Shoes          | 129.99 | TRUE               |
| 101         | Sarah Miller     | 1003       | Programming Book       | 49.99  | TRUE               |
| 101         | Sarah Miller     | 1004       | Smart Watch            | 299.99 | TRUE               |
| 102         | Tom Anderson     | 1002       | Running Shoes          | 129.99 | TRUE               |
| 103         | Lisa Chen        | 1001       | Wireless Headphones    | 89.99  | TRUE               |
| 103         | Lisa Chen        | 1003       | Programming Book       | 49.99  | TRUE               |
| 104         | Mike Davis       | 1001       | Wireless Headphones    | 89.99  | TRUE               |
| 104         | Mike Davis       | 1003       | Programming Book       | 49.99  | TRUE               |
| 104         | Mike Davis       | 1004       | Smart Watch            | 299.99 | TRUE               |
| 105         | Emma Wilson      | 1005       | Coffee Maker           | 79.99  | TRUE               |
+-------------+------------------+------------+-------------------------+--------+--------------------+
```

This example finds products that might interest customers based on overlapping categories.

### Example 3: Access Control and Permission Checking

Sample data in `user_permissions` and `resource_requirements` tables:
```
-- user_permissions table
+---------+------------------+--------------------------------------------------+
| user_id | username         | permissions                                      |
+---------+------------------+--------------------------------------------------+
| 1       | admin_user       | ['read', 'write', 'delete', 'admin']           |
| 2       | editor_jane      | ['read', 'write', 'publish']                   |
| 3       | viewer_bob       | ['read']                                        |
| 4       | moderator_alice  | ['read', 'write', 'moderate']                  |
| 5       | guest_user       | ['read', 'comment']                             |
+---------+------------------+--------------------------------------------------+

-- resource_requirements table
+-------------+-------------------------+-----------------------------------+
| resource_id | resource_name          | required_permissions              |
+-------------+-------------------------+-----------------------------------+
| 101         | Financial Reports      | ['read', 'admin']                 |
| 102         | Blog Posts             | ['read', 'write']                 |
| 103         | User Management        | ['admin']                         |
| 104         | Comment Moderation     | ['moderate', 'write']             |
| 105         | Public Documents       | ['read']                          |
+-------------+-------------------------+-----------------------------------+
```

Query:
```sql
SELECT 
    u.user_id,
    u.username,
    r.resource_id,
    r.resource_name,
    u.permissions,
    r.required_permissions,
    ARRAYS_OVERLAP(u.permissions, r.required_permissions) AS has_access
FROM user_permissions u
CROSS JOIN resource_requirements r
WHERE ARRAYS_OVERLAP(u.permissions, r.required_permissions) = TRUE
ORDER BY u.user_id, r.resource_id;
```

Result:
```
+---------+------------------+-------------+-------------------------+--------------------------------------------------+-----------------------------------+------------+
| user_id | username         | resource_id | resource_name          | permissions                                      | required_permissions              | has_access |
+---------+------------------+-------------+-------------------------+--------------------------------------------------+-----------------------------------+------------+
| 1       | admin_user       | 101         | Financial Reports      | ['read', 'write', 'delete', 'admin']           | ['read', 'admin']                 | TRUE       |
| 1       | admin_user       | 102         | Blog Posts             | ['read', 'write', 'delete', 'admin']           | ['read', 'write']                 | TRUE       |
| 1       | admin_user       | 103         | User Management        | ['read', 'write', 'delete', 'admin']           | ['admin']                         | TRUE       |
| 1       | admin_user       | 104         | Comment Moderation     | ['read', 'write', 'delete', 'admin']           | ['moderate', 'write']             | TRUE       |
| 1       | admin_user       | 105         | Public Documents       | ['read', 'write', 'delete', 'admin']           | ['read']                          | TRUE       |
| 2       | editor_jane      | 102         | Blog Posts             | ['read', 'write', 'publish']                   | ['read', 'write']                 | TRUE       |
| 2       | editor_jane      | 105         | Public Documents       | ['read', 'write', 'publish']                   | ['read']                          | TRUE       |
| 3       | viewer_bob       | 105         | Public Documents       | ['read']                                        | ['read']                          | TRUE       |
| 4       | moderator_alice  | 102         | Blog Posts             | ['read', 'write', 'moderate']                  | ['read', 'write']                 | TRUE       |
| 4       | moderator_alice  | 104         | Comment Moderation     | ['read', 'write', 'moderate']                  | ['moderate', 'write']             | TRUE       |
| 4       | moderator_alice  | 105         | Public Documents       | ['read', 'write', 'moderate']                  | ['read']                          | TRUE       |
| 5       | guest_user       | 105         | Public Documents       | ['read', 'comment']                             | ['read']                          | TRUE       |
+---------+------------------+-------------+-------------------------+--------------------------------------------------+-----------------------------------+------------+
```

This example demonstrates using ARRAYS_OVERLAP for access control, showing which users have access to which resources.

### Example 4: Event Tag Matching for Notifications

Sample data in `user_interests` and `events` tables:
```
-- user_interests table
+---------+------------------+------------------------------------------------+
| user_id | user_email       | interest_tags                                  |
+---------+------------------+------------------------------------------------+
| 1       | tech@email.com   | ['technology', 'ai', 'programming', 'startup'] |
| 2       | art@email.com    | ['art', 'design', 'photography', 'music']     |
| 3       | health@email.com | ['fitness', 'nutrition', 'wellness', 'sports'] |
| 4       | biz@email.com    | ['business', 'finance', 'startup', 'marketing']|
| 5       | edu@email.com    | ['education', 'science', 'research', 'ai']     |
+---------+------------------+------------------------------------------------+

-- events table
+----------+--------------------------------+------------------------------------------+------------+
| event_id | event_name                    | event_tags                               | event_date |
+----------+--------------------------------+------------------------------------------+------------+
| 201      | AI Conference 2024            | ['technology', 'ai', 'research']         | 2024-03-15 |
| 202      | Startup Pitch Night           | ['startup', 'business', 'networking']    | 2024-03-20 |
| 203      | Digital Art Exhibition        | ['art', 'design', 'technology']          | 2024-03-25 |
| 204      | Fitness & Wellness Expo       | ['fitness', 'wellness', 'health']        | 2024-04-01 |
| 205      | Photography Workshop          | ['photography', 'art', 'education']      | 2024-04-10 |
+----------+--------------------------------+------------------------------------------+------------+
```

Query:
```sql
SELECT 
    u.user_id,
    u.user_email,
    e.event_id,
    e.event_name,
    e.event_date,
    ARRAYS_OVERLAP(u.interest_tags, e.event_tags) AS should_notify
FROM user_interests u
CROSS JOIN events e
WHERE ARRAYS_OVERLAP(u.interest_tags, e.event_tags) = TRUE
ORDER BY e.event_date, u.user_id;
```

Result:
```
+---------+------------------+----------+--------------------------------+------------+---------------+
| user_id | user_email       | event_id | event_name                    | event_date | should_notify |
+---------+------------------+----------+--------------------------------+------------+---------------+
| 1       | tech@email.com   | 201      | AI Conference 2024            | 2024-03-15 | TRUE          |
| 5       | edu@email.com    | 201      | AI Conference 2024            | 2024-03-15 | TRUE          |
| 1       | tech@email.com   | 202      | Startup Pitch Night           | 2024-03-20 | TRUE          |
| 4       | biz@email.com    | 202      | Startup Pitch Night           | 2024-03-20 | TRUE          |
| 1       | tech@email.com   | 203      | Digital Art Exhibition        | 2024-03-25 | TRUE          |
| 2       | art@email.com    | 203      | Digital Art Exhibition        | 2024-03-25 | TRUE          |
| 3       | health@email.com | 204      | Fitness & Wellness Expo       | 2024-04-01 | TRUE          |
| 2       | art@email.com    | 205      | Photography Workshop          | 2024-04-10 | TRUE          |
| 5       | edu@email.com    | 205      | Photography Workshop          | 2024-04-10 | TRUE          |
+---------+------------------+----------+--------------------------------+------------+---------------+
```

This example shows how to match users with events based on overlapping interest tags for targeted notifications.

### Example 5: Ingredient Allergy Checking

Sample data in `recipes` and `customer_allergies` tables:
```
-- recipes table
+-----------+-------------------------+------------------------------------------------------+
| recipe_id | recipe_name            | ingredients                                          |
+-----------+-------------------------+------------------------------------------------------+
| 1         | Chocolate Chip Cookies | ['flour', 'butter', 'sugar', 'eggs', 'chocolate']   |
| 2         | Caesar Salad           | ['lettuce', 'parmesan', 'croutons', 'anchovies']   |
| 3         | Vegetable Stir Fry     | ['broccoli', 'carrots', 'soy sauce', 'garlic']     |
| 4         | Seafood Pasta          | ['pasta', 'shrimp', 'cream', 'garlic', 'parmesan'] |
| 5         | Fruit Smoothie         | ['banana', 'strawberry', 'yogurt', 'honey']         |
+-----------+-------------------------+------------------------------------------------------+

-- customer_allergies table
+-------------+------------------+-----------------------------------+
| customer_id | customer_name    | allergies                         |
+-------------+------------------+-----------------------------------+
| 101         | John Smith       | ['gluten', 'eggs']               |
| 102         | Mary Johnson     | ['dairy', 'nuts']                 |
| 103         | David Lee        | ['shellfish', 'soy']              |
| 104         | Sarah Brown      | []                                |
| 105         | Tom Wilson       | ['dairy']                         |
+-------------+------------------+-----------------------------------+
```

Query:
```sql
WITH recipe_allergens AS (
    SELECT 
        recipe_id,
        recipe_name,
        ingredients,
        CASE 
            WHEN ARRAYS_OVERLAP(ingredients, ['flour', 'bread', 'pasta', 'croutons']) THEN ['gluten']
            ELSE []
        END ||
        CASE 
            WHEN ARRAYS_OVERLAP(ingredients, ['milk', 'butter', 'cream', 'cheese', 'yogurt', 'parmesan']) THEN ['dairy']
            ELSE []
        END ||
        CASE 
            WHEN ARRAYS_OVERLAP(ingredients, ['eggs']) THEN ['eggs']
            ELSE []
        END ||
        CASE 
            WHEN ARRAYS_OVERLAP(ingredients, ['shrimp', 'lobster', 'crab']) THEN ['shellfish']
            ELSE []
        END ||
        CASE 
            WHEN ARRAYS_OVERLAP(ingredients, ['soy sauce', 'tofu']) THEN ['soy']
            ELSE []
        END AS contains_allergens
    FROM recipes
)
SELECT 
    c.customer_id,
    c.customer_name,
    r.recipe_id,
    r.recipe_name,
    c.allergies,
    r.contains_allergens,
    CASE 
        WHEN c.allergies = [] THEN FALSE
        ELSE ARRAYS_OVERLAP(c.allergies, r.contains_allergens)
    END AS is_allergic
FROM customer_allergies c
CROSS JOIN recipe_allergens r
WHERE ARRAYS_OVERLAP(c.allergies, r.contains_allergens) = TRUE
   OR c.allergies = []
ORDER BY c.customer_id, r.recipe_id;
```

Result:
```
+-------------+------------------+-----------+-------------------------+-----------------------------------+-------------------------------+-------------+
| customer_id | customer_name    | recipe_id | recipe_name            | allergies                         | contains_allergens            | is_allergic |
+-------------+------------------+-----------+-------------------------+-----------------------------------+-------------------------------+-------------+
| 101         | John Smith       | 1         | Chocolate Chip Cookies | ['gluten', 'eggs']               | ['gluten', 'dairy', 'eggs']  | TRUE        |
| 101         | John Smith       | 2         | Caesar Salad           | ['gluten', 'eggs']               | ['gluten', 'dairy']          | TRUE        |
| 101         | John Smith       | 4         | Seafood Pasta          | ['gluten', 'eggs']               | ['gluten', 'dairy']          | TRUE        |
| 102         | Mary Johnson     | 1         | Chocolate Chip Cookies | ['dairy', 'nuts']                | ['gluten', 'dairy', 'eggs']  | TRUE        |
| 102         | Mary Johnson     | 2         | Caesar Salad           | ['dairy', 'nuts']                | ['gluten', 'dairy']          | TRUE        |
| 102         | Mary Johnson     | 4         | Seafood Pasta          | ['dairy', 'nuts']                | ['gluten', 'dairy']          | TRUE        |
| 102         | Mary Johnson     | 5         | Fruit Smoothie         | ['dairy', 'nuts']                | ['dairy']                    | TRUE        |
| 103         | David Lee        | 3         | Vegetable Stir Fry     | ['shellfish', 'soy']             | ['soy']                      | TRUE        |
| 103         | David Lee        | 4         | Seafood Pasta          | ['shellfish', 'soy']             | ['shellfish', 'dairy']       | TRUE        |
| 105         | Tom Wilson       | 1         | Chocolate Chip Cookies | ['dairy']                         | ['gluten', 'dairy', 'eggs']  | TRUE        |
| 105         | Tom Wilson       | 2         | Caesar Salad           | ['dairy']                         | ['gluten', 'dairy']          | TRUE        |
| 105         | Tom Wilson       | 4         | Seafood Pasta          | ['dairy']                         | ['gluten', 'dairy']          | TRUE        |
| 105         | Tom Wilson       | 5         | Fruit Smoothie         | ['dairy']                         | ['dairy']                    | TRUE        |
+-------------+------------------+-----------+-------------------------+-----------------------------------+-------------------------------+-------------+
```

This example uses ARRAYS_OVERLAP to identify recipes that contain ingredients matching customer allergies, helping restaurants provide safe meal options.