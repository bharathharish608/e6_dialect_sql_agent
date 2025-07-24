# ARRAY_JOIN

Concatenates array elements into a single string using a specified delimiter.

## Syntax

```sql
ARRAY_JOIN( <array>, <delimiter> [ , <null_replacement> ] )
```

## Arguments

### array
- **Type**: ARRAY
- **Required**: Yes
- **Description**: The array of elements to join. Elements will be converted to strings before joining

### delimiter
- **Type**: STRING
- **Required**: Yes
- **Description**: The string to use as a separator between array elements

### null_replacement
- **Type**: STRING
- **Required**: No
- **Default**: Empty string
- **Description**: The string to use in place of NULL values within the array. If not specified, NULL values are treated as empty strings

## Returns

- **Type**: VARCHAR(2000) NULLABLE
- **Description**: Returns a string containing all array elements joined by the specified delimiter
- **NULL Handling**: Returns NULL if the input array is NULL. NULL elements within the array are replaced with the null_replacement string or empty string if not specified

## Usage Notes

- Elements are converted to strings before joining
- The delimiter can be any string, including empty string
- NULL values within the array can be replaced with a custom string
- The result is limited to VARCHAR(2000)
- Useful for creating CSV lists, formatting output, or building dynamic strings

## Examples

### Example 1: Basic String Array Joining

Sample data in `employee_skills` table:
```
+--------+---------------+------------------------------------------------+-------------------------------------+
| emp_id | employee_name | programming_languages                          | certifications                      |
+--------+---------------+------------------------------------------------+-------------------------------------+
| 1      | Alice Johnson | ['Python','Java','JavaScript','SQL']          | ['AWS Solutions Architect','PMP']  |
| 2      | Bob Smith     | ['C++','Python','Go']                         | ['Azure Developer','Scrum Master'] |
| 3      | Carol Davis   | ['JavaScript','TypeScript','React','Node.js'] | ['Google Cloud','CISSP']           |
| 4      | David Wilson  | ['Java','Kotlin','Swift']                     | ['Oracle Java','Android']          |
| 5      | Eve Chen      | ['Python','R','SQL','Scala']                  | ['Data Science','Machine Learning'] |
+--------+---------------+------------------------------------------------+-------------------------------------+
```

Query:
```sql
SELECT 
    employee_name,
    programming_languages,
    ARRAY_JOIN(programming_languages, ', ') AS languages_list,
    certifications,
    ARRAY_JOIN(certifications, ' | ') AS certs_list
FROM employee_skills
ORDER BY emp_id;
```

Result:
```
+---------------+------------------------------------------------+----------------------------------------+-------------------------------------+----------------------------------------+
| employee_name | programming_languages                          | languages_list                         | certifications                      | certs_list                             |
+---------------+------------------------------------------------+----------------------------------------+-------------------------------------+----------------------------------------+
| Alice Johnson | ['Python','Java','JavaScript','SQL']          | Python, Java, JavaScript, SQL          | ['AWS Solutions Architect','PMP']  | AWS Solutions Architect | PMP          |
| Bob Smith     | ['C++','Python','Go']                         | C++, Python, Go                        | ['Azure Developer','Scrum Master'] | Azure Developer | Scrum Master        |
| Carol Davis   | ['JavaScript','TypeScript','React','Node.js'] | JavaScript, TypeScript, React, Node.js | ['Google Cloud','CISSP']           | Google Cloud | CISSP                   |
| David Wilson  | ['Java','Kotlin','Swift']                     | Java, Kotlin, Swift                    | ['Oracle Java','Android']          | Oracle Java | Android                  |
| Eve Chen      | ['Python','R','SQL','Scala']                  | Python, R, SQL, Scala                  | ['Data Science','Machine Learning'] | Data Science | Machine Learning       |
+---------------+------------------------------------------------+----------------------------------------+-------------------------------------+----------------------------------------+
```

This example joins programming languages with commas and certifications with pipe separators.

### Example 2: Handling NULL Values

Sample data in `product_features` table:
```
+------------+-----------------+----------------------------------------+----------------------------------------+
| product_id | product_name    | features                               | optional_features                      |
+------------+-----------------+----------------------------------------+----------------------------------------+
| 1          | Smart TV        | ['4K Display','HDR',NULL,'Smart Apps'] | ['Voice Control',NULL,'Gaming Mode']  |
| 2          | Laptop Pro      | ['Intel i7','16GB RAM','512GB SSD']   | [NULL,'Touchscreen','Backlit Keys']   |
| 3          | Smartphone      | ['5G',NULL,'128GB Storage','OLED']    | ['Wireless Charging',NULL,NULL]       |
| 4          | Tablet          | ['10 inch','WiFi',NULL,'Stylus']      | []                                     |
| 5          | Smart Watch     | ['Heart Rate','GPS',NULL,NULL]        | NULL                                   |
+------------+-----------------+----------------------------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    product_name,
    features,
    ARRAY_JOIN(features, ', ') AS features_default,
    ARRAY_JOIN(features, ', ', 'N/A') AS features_with_na,
    ARRAY_JOIN(features, ', ', '[Not Specified]') AS features_detailed,
    optional_features,
    ARRAY_JOIN(optional_features, ' + ', 'None') AS optional_list
FROM product_features
ORDER BY product_id;
```

Result:
```
+-----------------+----------------------------------------+----------------------------+--------------------------------+----------------------------------------------+----------------------------------------+--------------------------------+
| product_name    | features                               | features_default           | features_with_na               | features_detailed                            | optional_features                      | optional_list                  |
+-----------------+----------------------------------------+----------------------------+--------------------------------+----------------------------------------------+----------------------------------------+--------------------------------+
| Smart TV        | ['4K Display','HDR',NULL,'Smart Apps'] | 4K Display, HDR, , Smart Apps | 4K Display, HDR, N/A, Smart Apps | 4K Display, HDR, [Not Specified], Smart Apps | ['Voice Control',NULL,'Gaming Mode']  | Voice Control + None + Gaming Mode |
| Laptop Pro      | ['Intel i7','16GB RAM','512GB SSD']   | Intel i7, 16GB RAM, 512GB SSD | Intel i7, 16GB RAM, 512GB SSD | Intel i7, 16GB RAM, 512GB SSD               | [NULL,'Touchscreen','Backlit Keys']   | None + Touchscreen + Backlit Keys |
| Smartphone      | ['5G',NULL,'128GB Storage','OLED']    | 5G, , 128GB Storage, OLED     | 5G, N/A, 128GB Storage, OLED   | 5G, [Not Specified], 128GB Storage, OLED     | ['Wireless Charging',NULL,NULL]       | Wireless Charging + None + None   |
| Tablet          | ['10 inch','WiFi',NULL,'Stylus']      | 10 inch, WiFi, , Stylus       | 10 inch, WiFi, N/A, Stylus     | 10 inch, WiFi, [Not Specified], Stylus       | []                                     |                                |
| Smart Watch     | ['Heart Rate','GPS',NULL,NULL]        | Heart Rate, GPS, ,            | Heart Rate, GPS, N/A, N/A      | Heart Rate, GPS, [Not Specified], [Not Specified] | NULL                                   | NULL                           |
+-----------------+----------------------------------------+----------------------------+--------------------------------+----------------------------------------------+----------------------------------------+--------------------------------+
```

This example demonstrates different ways to handle NULL values when joining arrays.

### Example 3: Creating Formatted Output

Sample data in `order_details` table:
```
+----------+---------------+----------------------------------------+---------------------------+
| order_id | customer_name | item_names                             | item_quantities           |
+----------+---------------+----------------------------------------+---------------------------+
| 1        | John Smith    | ['Laptop','Mouse','Keyboard']         | [1,2,1]                  |
| 2        | Jane Doe      | ['Monitor','HDMI Cable']               | [2,3]                    |
| 3        | Bob Johnson   | ['Printer','Paper','Ink Cartridge']    | [1,5,2]                  |
| 4        | Alice Brown   | ['Desk Lamp','USB Hub','Webcam']      | [1,1,1]                  |
| 5        | Tom Wilson    | ['External SSD','USB-C Cable']         | [2,4]                    |
+----------+---------------+----------------------------------------+---------------------------+
```

Query:
```sql
SELECT 
    order_id,
    customer_name,
    item_names,
    item_quantities,
    ARRAY_JOIN(item_names, ' / ') AS items_summary,
    CONCAT(
        'Order contains: ',
        ARRAY_JOIN(item_names, ', '),
        ' (Total items: ',
        ARRAY_SIZE(item_names),
        ')'
    ) AS order_description
FROM order_details
ORDER BY order_id;
```

Result:
```
+----------+---------------+----------------------------------------+---------------------------+--------------------------------+--------------------------------------------------------------------------+
| order_id | customer_name | item_names                             | item_quantities           | items_summary                  | order_description                                                        |
+----------+---------------+----------------------------------------+---------------------------+--------------------------------+--------------------------------------------------------------------------+
| 1        | John Smith    | ['Laptop','Mouse','Keyboard']         | [1,2,1]                  | Laptop / Mouse / Keyboard      | Order contains: Laptop, Mouse, Keyboard (Total items: 3)                |
| 2        | Jane Doe      | ['Monitor','HDMI Cable']               | [2,3]                    | Monitor / HDMI Cable           | Order contains: Monitor, HDMI Cable (Total items: 2)                    |
| 3        | Bob Johnson   | ['Printer','Paper','Ink Cartridge']    | [1,5,2]                  | Printer / Paper / Ink Cartridge| Order contains: Printer, Paper, Ink Cartridge (Total items: 3)          |
| 4        | Alice Brown   | ['Desk Lamp','USB Hub','Webcam']      | [1,1,1]                  | Desk Lamp / USB Hub / Webcam   | Order contains: Desk Lamp, USB Hub, Webcam (Total items: 3)             |
| 5        | Tom Wilson    | ['External SSD','USB-C Cable']         | [2,4]                    | External SSD / USB-C Cable     | Order contains: External SSD, USB-C Cable (Total items: 2)              |
+----------+---------------+----------------------------------------+---------------------------+--------------------------------+--------------------------------------------------------------------------+
```

This example shows how to create formatted output strings from arrays.

### Example 4: Building Dynamic Queries and URLs

Sample data in `api_endpoints` table:
```
+--------+----------------+----------------------------------------+----------------------------------------+
| api_id | endpoint_name  | path_segments                          | query_params                           |
+--------+----------------+----------------------------------------+----------------------------------------+
| 1      | User Profile   | ['api','v2','users','profile']        | ['id=123','format=json']              |
| 2      | Product Search | ['api','v1','products','search']      | ['q=laptop','category=electronics','limit=10'] |
| 3      | Order Status   | ['api','v3','orders','status']        | ['order_id=456','include=items']      |
| 4      | Analytics      | ['api','v2','analytics','dashboard']  | ['start=2024-01-01','end=2024-03-31','type=summary'] |
| 5      | Auth Token     | ['oauth','token']                      | ['grant_type=client','scope=read']    |
+--------+----------------+----------------------------------------+----------------------------------------+
```

Query:
```sql
SELECT 
    endpoint_name,
    path_segments,
    query_params,
    CONCAT('/', ARRAY_JOIN(path_segments, '/')) AS path,
    ARRAY_JOIN(query_params, '&') AS query_string,
    CONCAT(
        'https://api.example.com/',
        ARRAY_JOIN(path_segments, '/'),
        '?',
        ARRAY_JOIN(query_params, '&')
    ) AS full_url
FROM api_endpoints
ORDER BY api_id;
```

Result:
```
+----------------+----------------------------------------+----------------------------------------+--------------------------------+----------------------------------------+--------------------------------------------------------------------------------+
| endpoint_name  | path_segments                          | query_params                           | path                           | query_string                           | full_url                                                                       |
+----------------+----------------------------------------+----------------------------------------+--------------------------------+----------------------------------------+--------------------------------------------------------------------------------+
| User Profile   | ['api','v2','users','profile']        | ['id=123','format=json']              | /api/v2/users/profile          | id=123&format=json                     | https://api.example.com/api/v2/users/profile?id=123&format=json               |
| Product Search | ['api','v1','products','search']      | ['q=laptop','category=electronics','limit=10'] | /api/v1/products/search        | q=laptop&category=electronics&limit=10 | https://api.example.com/api/v1/products/search?q=laptop&category=electronics&limit=10 |
| Order Status   | ['api','v3','orders','status']        | ['order_id=456','include=items']      | /api/v3/orders/status          | order_id=456&include=items             | https://api.example.com/api/v3/orders/status?order_id=456&include=items       |
| Analytics      | ['api','v2','analytics','dashboard']  | ['start=2024-01-01','end=2024-03-31','type=summary'] | /api/v2/analytics/dashboard    | start=2024-01-01&end=2024-03-31&type=summary | https://api.example.com/api/v2/analytics/dashboard?start=2024-01-01&end=2024-03-31&type=summary |
| Auth Token     | ['oauth','token']                      | ['grant_type=client','scope=read']    | /oauth/token                   | grant_type=client&scope=read           | https://api.example.com/oauth/token?grant_type=client&scope=read              |
+----------------+----------------------------------------+----------------------------------------+--------------------------------+----------------------------------------+--------------------------------------------------------------------------------+
```

This example demonstrates building URLs and query strings from array components.

### Example 5: Complex Data Formatting

Sample data in `event_logs` table:
```
+----------+--------------------+----------------------------------------+----------------------------------------+-----------------------------+
| event_id | event_type         | participants                           | actions                                | timestamps                  |
+----------+--------------------+----------------------------------------+----------------------------------------+-----------------------------+
| 1        | Team Meeting       | ['Alice','Bob','Carol','David']       | ['Joined','Presented','Discussed','Left'] | ['09:00','09:15','10:00','10:30'] |
| 2        | Code Review        | ['Eve','Frank',NULL,'Grace']          | ['Started',NULL,'Commented','Approved'] | ['14:00','14:10','14:30','15:00'] |
| 3        | Deploy             | ['System','Admin','QA']                | ['Initiated','Validated','Completed']  | ['18:00','18:15','18:30']  |
| 4        | Customer Call      | ['Support','Customer',NULL]           | ['Connected','Troubleshooting',NULL]   | ['11:00','11:05',NULL]      |
| 5        | Training Session   | ['Instructor','Student1','Student2']  | ['Started','Exercise','Completed']     | ['13:00','13:30','14:30']   |
+----------+--------------------+----------------------------------------+----------------------------------------+-----------------------------+
```

Query:
```sql
SELECT 
    event_id,
    event_type,
    ARRAY_JOIN(participants, ' → ', '[Unknown]') AS participant_flow,
    ARRAY_JOIN(actions, ' | ', '[No Action]') AS action_sequence,
    CONCAT(
        'Event: ', event_type, 
        ' | Participants: ', ARRAY_JOIN(participants, ', ', 'Unknown'),
        ' | Duration: ', ARRAY_MIN(timestamps), ' - ', ARRAY_MAX(timestamps)
    ) AS event_summary,
    ARRAY_JOIN(
        ARRAY_SORT(participants), 
        '; '
    ) AS sorted_participants
FROM event_logs
ORDER BY event_id;
```

Result:
```
+----------+--------------------+------------------------------------+----------------------------------------+-------------------------------------------------------------------------------------------+--------------------------------+
| event_id | event_type         | participant_flow                   | action_sequence                        | event_summary                                                                             | sorted_participants            |
+----------+--------------------+------------------------------------+----------------------------------------+-------------------------------------------------------------------------------------------+--------------------------------+
| 1        | Team Meeting       | Alice → Bob → Carol → David        | Joined | Presented | Discussed | Left | Event: Team Meeting | Participants: Alice, Bob, Carol, David | Duration: 09:00 - 10:30       | Alice; Bob; Carol; David       |
| 2        | Code Review        | Eve → Frank → [Unknown] → Grace    | Started | [No Action] | Commented | Approved | Event: Code Review | Participants: Eve, Frank, Unknown, Grace | Duration: 14:00 - 15:00    | Eve; Frank; Grace              |
| 3        | Deploy             | System → Admin → QA                | Initiated | Validated | Completed      | Event: Deploy | Participants: System, Admin, QA | Duration: 18:00 - 18:30                  | Admin; QA; System              |
| 4        | Customer Call      | Support → Customer → [Unknown]     | Connected | Troubleshooting | [No Action] | Event: Customer Call | Participants: Support, Customer, Unknown | Duration: 11:00 - 11:05  | Customer; Support              |
| 5        | Training Session   | Instructor → Student1 → Student2   | Started | Exercise | Completed        | Event: Training Session | Participants: Instructor, Student1, Student2 | Duration: 13:00 - 14:30 | Instructor; Student1; Student2 |
+----------+--------------------+------------------------------------+----------------------------------------+-------------------------------------------------------------------------------------------+--------------------------------+
```

This example shows complex formatting combining ARRAY_JOIN with other functions to create detailed event summaries.