# ARRAY_INTERSECT

Returns the intersection of two arrays, containing only the elements that appear in both arrays. This function is useful for finding common elements between two arrays, such as shared skills, common tags, or matching attributes.

## Syntax

```sql
ARRAY_INTERSECT( <array1>, <array2> )
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

- **Type**: ARRAY (same element type as input arrays)
- **Description**: Returns a new array containing only elements present in both input arrays. Duplicates are removed from the result.
- **NULL Handling**: Returns NULL if either array is NULL. Returns an empty array if there are no common elements.

## Usage Notes

- Both arrays must have the same element type
- The result array contains unique elements only (no duplicates)
- The order of elements in the result is not guaranteed
- Case-sensitive comparison for string arrays
- Useful for finding commonalities, filtering data, and set operations

## Examples

### Example 1: Finding Common Skills Between Team Members

Sample data in `team_skills` table:
```
+-------------+------------------+--------------------------------------------------+
| employee_id | employee_name    | skills                                           |
+-------------+------------------+--------------------------------------------------+
| 1           | Alice Johnson    | ['Python', 'SQL', 'Docker', 'AWS', 'Git']      |
| 2           | Bob Smith        | ['Java', 'SQL', 'Docker', 'Jenkins', 'Git']    |
| 3           | Carol Williams   | ['Python', 'JavaScript', 'React', 'Git', 'CSS'] |
| 4           | David Brown      | ['Java', 'Spring', 'Docker', 'Kubernetes']      |
| 5           | Eve Davis        | ['Python', 'SQL', 'AWS', 'Terraform', 'Git']   |
+-------------+------------------+--------------------------------------------------+
```

Query:
```sql
SELECT 
    t1.employee_name AS employee1,
    t2.employee_name AS employee2,
    t1.skills AS skills1,
    t2.skills AS skills2,
    ARRAY_INTERSECT(t1.skills, t2.skills) AS common_skills,
    CARDINALITY(ARRAY_INTERSECT(t1.skills, t2.skills)) AS num_common_skills
FROM team_skills t1
JOIN team_skills t2 ON t1.employee_id < t2.employee_id
WHERE CARDINALITY(ARRAY_INTERSECT(t1.skills, t2.skills)) >= 2
ORDER BY num_common_skills DESC, t1.employee_id, t2.employee_id;
```

Result:
```
+---------------+---------------+--------------------------------------------------+--------------------------------------------------+--------------------------------+-------------------+
| employee1     | employee2     | skills1                                          | skills2                                          | common_skills                  | num_common_skills |
+---------------+---------------+--------------------------------------------------+--------------------------------------------------+--------------------------------+-------------------+
| Alice Johnson | Eve Davis     | ['Python', 'SQL', 'Docker', 'AWS', 'Git']      | ['Python', 'SQL', 'AWS', 'Terraform', 'Git']   | ['Python', 'SQL', 'AWS', 'Git']| 4                 |
| Alice Johnson | Bob Smith     | ['Python', 'SQL', 'Docker', 'AWS', 'Git']      | ['Java', 'SQL', 'Docker', 'Jenkins', 'Git']    | ['SQL', 'Docker', 'Git']       | 3                 |
| Bob Smith     | David Brown   | ['Java', 'SQL', 'Docker', 'Jenkins', 'Git']    | ['Java', 'Spring', 'Docker', 'Kubernetes']      | ['Java', 'Docker']             | 2                 |
| Alice Johnson | Carol Williams| ['Python', 'SQL', 'Docker', 'AWS', 'Git']      | ['Python', 'JavaScript', 'React', 'Git', 'CSS'] | ['Python', 'Git']              | 2                 |
| Carol Williams| Eve Davis     | ['Python', 'JavaScript', 'React', 'Git', 'CSS'] | ['Python', 'SQL', 'AWS', 'Terraform', 'Git']   | ['Python', 'Git']              | 2                 |
+---------------+---------------+--------------------------------------------------+--------------------------------------------------+--------------------------------+-------------------+
```

This example finds team members with at least 2 common skills, useful for pairing or collaboration opportunities.

### Example 2: Product Feature Comparison

Sample data in `product_features` table:
```
+------------+-------------------------+------------------------------------------------------------+--------+
| product_id | product_name           | features                                                   | price  |
+------------+-------------------------+------------------------------------------------------------+--------+
| 101        | Premium Laptop         | ['16GB RAM', 'SSD', 'Touchscreen', 'Backlit Keyboard']   | 1299   |
| 102        | Business Laptop        | ['8GB RAM', 'SSD', 'Fingerprint', 'Backlit Keyboard']    | 899    |
| 103        | Gaming Laptop          | ['32GB RAM', 'SSD', 'RGB Keyboard', 'Dedicated GPU']     | 1799   |
| 104        | Student Laptop         | ['8GB RAM', 'HDD', 'Webcam', 'Lightweight']              | 599    |
| 105        | Professional Workstation| ['32GB RAM', 'SSD', 'Dedicated GPU', 'Fingerprint']      | 2299   |
+------------+-------------------------+------------------------------------------------------------+--------+
```

Query:
```sql
WITH product_pairs AS (
    SELECT 
        p1.product_id AS product1_id,
        p1.product_name AS product1_name,
        p1.price AS price1,
        p2.product_id AS product2_id,
        p2.product_name AS product2_name,
        p2.price AS price2,
        ARRAY_INTERSECT(p1.features, p2.features) AS shared_features,
        CARDINALITY(ARRAY_INTERSECT(p1.features, p2.features)) AS num_shared
    FROM product_features p1
    JOIN product_features p2 ON p1.product_id < p2.product_id
)
SELECT 
    product1_name,
    product2_name,
    shared_features,
    num_shared,
    ABS(price1 - price2) AS price_difference
FROM product_pairs
WHERE num_shared >= 2
ORDER BY num_shared DESC, price_difference;
```

Result:
```
+-------------------------+-------------------------+-------------------------------------+------------+------------------+
| product1_name           | product2_name           | shared_features                     | num_shared | price_difference |
+-------------------------+-------------------------+-------------------------------------+------------+------------------+
| Gaming Laptop          | Professional Workstation| ['32GB RAM', 'SSD', 'Dedicated GPU']| 3          | 500              |
| Premium Laptop         | Business Laptop        | ['SSD', 'Backlit Keyboard']        | 2          | 400              |
| Business Laptop        | Professional Workstation| ['SSD', 'Fingerprint']             | 2          | 1400             |
| Premium Laptop         | Gaming Laptop          | ['SSD']                            | 1          | 500              |
| Premium Laptop         | Professional Workstation| ['SSD']                            | 1          | 1000             |
+-------------------------+-------------------------+-------------------------------------+------------+------------------+
```

This example compares products to find those with similar features, useful for recommendation systems.

### Example 3: Customer Interest Analysis

Sample data in `customer_interests` and `marketing_campaigns` tables:
```
-- customer_interests table
+-------------+------------------+--------------------------------------------------------+
| customer_id | customer_email   | interests                                              |
+-------------+------------------+--------------------------------------------------------+
| 201         | john@email.com   | ['travel', 'photography', 'food', 'technology']       |
| 202         | sarah@email.com  | ['fitness', 'nutrition', 'yoga', 'wellness']          |
| 203         | mike@email.com   | ['gaming', 'technology', 'movies', 'music']           |
| 204         | lisa@email.com   | ['fashion', 'beauty', 'travel', 'photography']        |
| 205         | david@email.com  | ['sports', 'fitness', 'outdoor', 'travel']            |
+-------------+------------------+--------------------------------------------------------+

-- marketing_campaigns table
+-------------+--------------------------------+--------------------------------------------------+
| campaign_id | campaign_name                  | target_interests                                 |
+-------------+--------------------------------+--------------------------------------------------+
| 301         | Summer Travel Deals           | ['travel', 'outdoor', 'photography', 'adventure']|
| 302         | Tech Innovation Week          | ['technology', 'gaming', 'gadgets', 'innovation']|
| 303         | Wellness & Lifestyle          | ['fitness', 'wellness', 'nutrition', 'health']   |
| 304         | Fashion Forward               | ['fashion', 'beauty', 'style', 'luxury']         |
| 305         | Entertainment Hub             | ['movies', 'music', 'gaming', 'streaming']       |
+-------------+--------------------------------+--------------------------------------------------+
```

Query:
```sql
SELECT 
    ci.customer_id,
    ci.customer_email,
    mc.campaign_id,
    mc.campaign_name,
    ARRAY_INTERSECT(ci.interests, mc.target_interests) AS matching_interests,
    CARDINALITY(ARRAY_INTERSECT(ci.interests, mc.target_interests)) AS match_score,
    ROUND(CAST(CARDINALITY(ARRAY_INTERSECT(ci.interests, mc.target_interests)) AS FLOAT) / 
          CARDINALITY(mc.target_interests) * 100, 1) AS relevance_percentage
FROM customer_interests ci
CROSS JOIN marketing_campaigns mc
WHERE CARDINALITY(ARRAY_INTERSECT(ci.interests, mc.target_interests)) > 0
ORDER BY ci.customer_id, match_score DESC;
```

Result:
```
+-------------+------------------+-------------+--------------------------------+----------------------------+-------------+----------------------+
| customer_id | customer_email   | campaign_id | campaign_name                  | matching_interests         | match_score | relevance_percentage |
+-------------+------------------+-------------+--------------------------------+----------------------------+-------------+----------------------+
| 201         | john@email.com   | 301         | Summer Travel Deals           | ['travel', 'photography']  | 2           | 50.0                 |
| 201         | john@email.com   | 302         | Tech Innovation Week          | ['technology']             | 1           | 25.0                 |
| 202         | sarah@email.com  | 303         | Wellness & Lifestyle          | ['fitness', 'wellness', 'nutrition'] | 3    | 75.0                 |
| 203         | mike@email.com   | 302         | Tech Innovation Week          | ['technology', 'gaming']   | 2           | 50.0                 |
| 203         | mike@email.com   | 305         | Entertainment Hub             | ['movies', 'music', 'gaming'] | 3        | 75.0                 |
| 204         | lisa@email.com   | 301         | Summer Travel Deals           | ['travel', 'photography']  | 2           | 50.0                 |
| 204         | lisa@email.com   | 304         | Fashion Forward               | ['fashion', 'beauty']      | 2           | 50.0                 |
| 205         | david@email.com  | 301         | Summer Travel Deals           | ['travel', 'outdoor']      | 2           | 50.0                 |
| 205         | david@email.com  | 303         | Wellness & Lifestyle          | ['fitness']                | 1           | 25.0                 |
+-------------+------------------+-------------+--------------------------------+----------------------------+-------------+----------------------+
```

This example matches customers to marketing campaigns based on shared interests, calculating relevance scores.

### Example 4: Menu Dietary Restriction Analysis

Sample data in `menu_items` and `dietary_profiles` tables:
```
-- menu_items table
+----------+-------------------------+----------------------------------------------------------------------+--------+
| item_id  | item_name              | ingredients                                                          | price  |
+----------+-------------------------+----------------------------------------------------------------------+--------+
| 1        | Margherita Pizza       | ['tomato', 'mozzarella', 'basil', 'wheat', 'olive oil']           | 12.99  |
| 2        | Grilled Chicken Salad  | ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil']         | 14.99  |
| 3        | Vegan Buddha Bowl      | ['quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini']          | 13.99  |
| 4        | Seafood Pasta          | ['pasta', 'shrimp', 'scallops', 'cream', 'garlic', 'wheat']      | 18.99  |
| 5        | Mushroom Risotto       | ['rice', 'mushrooms', 'parmesan', 'butter', 'wine']               | 16.99  |
+----------+-------------------------+----------------------------------------------------------------------+--------+

-- dietary_profiles table
+------------+------------------+-------------------------------------------------------------------+
| profile_id | diet_name        | allowed_ingredients                                               |
+------------+------------------+-------------------------------------------------------------------+
| 1          | Vegetarian       | ['tomato', 'mozzarella', 'basil', 'wheat', 'olive oil', 'lettuce', 'cucumber', 'quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini', 'rice', 'mushrooms', 'parmesan', 'butter', 'wine'] |
| 2          | Vegan           | ['tomato', 'basil', 'wheat', 'olive oil', 'lettuce', 'cucumber', 'quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini', 'rice', 'mushrooms', 'wine'] |
| 3          | Gluten-Free     | ['tomato', 'mozzarella', 'basil', 'olive oil', 'chicken', 'lettuce', 'cucumber', 'quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini', 'shrimp', 'scallops', 'cream', 'garlic', 'rice', 'mushrooms', 'parmesan', 'butter', 'wine'] |
| 4          | Dairy-Free      | ['tomato', 'basil', 'wheat', 'olive oil', 'chicken', 'lettuce', 'cucumber', 'quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini', 'pasta', 'shrimp', 'scallops', 'garlic', 'rice', 'mushrooms', 'wine'] |
+------------+------------------+-------------------------------------------------------------------+
```

Query:
```sql
SELECT 
    m.item_id,
    m.item_name,
    d.diet_name,
    m.ingredients,
    ARRAY_INTERSECT(m.ingredients, d.allowed_ingredients) AS allowed_parts,
    CARDINALITY(m.ingredients) AS total_ingredients,
    CARDINALITY(ARRAY_INTERSECT(m.ingredients, d.allowed_ingredients)) AS allowed_count,
    CASE 
        WHEN CARDINALITY(m.ingredients) = CARDINALITY(ARRAY_INTERSECT(m.ingredients, d.allowed_ingredients))
        THEN 'Fully Compatible'
        ELSE 'Not Compatible'
    END AS compatibility
FROM menu_items m
CROSS JOIN dietary_profiles d
WHERE CARDINALITY(m.ingredients) = CARDINALITY(ARRAY_INTERSECT(m.ingredients, d.allowed_ingredients))
ORDER BY d.diet_name, m.item_id;
```

Result:
```
+----------+-------------------------+---------------+----------------------------------------------------------------------+----------------------------------------------------------------------+-------------------+---------------+------------------+
| item_id  | item_name              | diet_name     | ingredients                                                          | allowed_parts                                                        | total_ingredients | allowed_count | compatibility    |
+----------+-------------------------+---------------+----------------------------------------------------------------------+----------------------------------------------------------------------+-------------------+---------------+------------------+
| 1        | Margherita Pizza       | Vegetarian    | ['tomato', 'mozzarella', 'basil', 'wheat', 'olive oil']           | ['tomato', 'mozzarella', 'basil', 'wheat', 'olive oil']           | 5                 | 5             | Fully Compatible |
| 3        | Vegan Buddha Bowl      | Vegetarian    | ['quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini']          | ['quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini']          | 5                 | 5             | Fully Compatible |
| 5        | Mushroom Risotto       | Vegetarian    | ['rice', 'mushrooms', 'parmesan', 'butter', 'wine']               | ['rice', 'mushrooms', 'parmesan', 'butter', 'wine']               | 5                 | 5             | Fully Compatible |
| 3        | Vegan Buddha Bowl      | Vegan         | ['quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini']          | ['quinoa', 'chickpeas', 'avocado', 'spinach', 'tahini']          | 5                 | 5             | Fully Compatible |
| 2        | Grilled Chicken Salad  | Gluten-Free   | ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil']         | ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil']         | 5                 | 5             | Fully Compatible |
+----------+-------------------------+---------------+----------------------------------------------------------------------+----------------------------------------------------------------------+-------------------+---------------+------------------+
```

This example identifies menu items that are fully compatible with different dietary restrictions.

### Example 5: Project Technology Stack Comparison

Sample data in `project_tech_stacks` table:
```
+------------+-------------------------+--------------------------------------------------------------------------------+
| project_id | project_name           | technologies                                                                   |
+------------+-------------------------+--------------------------------------------------------------------------------+
| 501        | E-commerce Platform    | ['React', 'Node.js', 'MongoDB', 'Redis', 'Docker', 'AWS', 'GraphQL']         |
| 502        | Mobile Banking App     | ['React Native', 'Node.js', 'PostgreSQL', 'Redis', 'Docker', 'AWS']          |
| 503        | Data Analytics Portal  | ['Angular', 'Python', 'PostgreSQL', 'Elasticsearch', 'Docker', 'Kubernetes']  |
| 504        | Social Media Dashboard | ['Vue.js', 'Node.js', 'MongoDB', 'Redis', 'Docker', 'GCP', 'GraphQL']        |
| 505        | IoT Management System  | ['React', 'Python', 'InfluxDB', 'Redis', 'Docker', 'AWS', 'MQTT']            |
+------------+-------------------------+--------------------------------------------------------------------------------+
```

Query:
```sql
WITH tech_overlap AS (
    SELECT 
        p1.project_id AS proj1_id,
        p1.project_name AS proj1_name,
        p2.project_id AS proj2_id,
        p2.project_name AS proj2_name,
        p1.technologies AS tech1,
        p2.technologies AS tech2,
        ARRAY_INTERSECT(p1.technologies, p2.technologies) AS shared_tech,
        CARDINALITY(ARRAY_INTERSECT(p1.technologies, p2.technologies)) AS shared_count,
        CARDINALITY(p1.technologies) AS tech1_count,
        CARDINALITY(p2.technologies) AS tech2_count
    FROM project_tech_stacks p1
    JOIN project_tech_stacks p2 ON p1.project_id < p2.project_id
)
SELECT 
    proj1_name,
    proj2_name,
    shared_tech,
    shared_count,
    ROUND(CAST(shared_count AS FLOAT) / LEAST(tech1_count, tech2_count) * 100, 1) AS similarity_percentage
FROM tech_overlap
WHERE shared_count >= 3
ORDER BY shared_count DESC, similarity_percentage DESC;
```

Result:
```
+-------------------------+-------------------------+--------------------------------------------+--------------+-----------------------+
| proj1_name             | proj2_name             | shared_tech                                | shared_count | similarity_percentage |
+-------------------------+-------------------------+--------------------------------------------+--------------+-----------------------+
| E-commerce Platform    | Mobile Banking App     | ['Node.js', 'Redis', 'Docker', 'AWS']     | 4            | 66.7                  |
| E-commerce Platform    | Social Media Dashboard | ['Node.js', 'MongoDB', 'Redis', 'Docker', 'GraphQL'] | 5   | 71.4                  |
| Mobile Banking App     | Social Media Dashboard | ['Node.js', 'Redis', 'Docker']            | 3            | 50.0                  |
| E-commerce Platform    | IoT Management System  | ['React', 'Redis', 'Docker', 'AWS']       | 4            | 57.1                  |
+-------------------------+-------------------------+--------------------------------------------+--------------+-----------------------+
```

This example finds projects with significant technology overlap, useful for team assignments and knowledge sharing.