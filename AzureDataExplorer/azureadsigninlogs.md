# Azure AD signin logs analysis queries for Azure Data Explorer

## Use ADX queries to analyze Azure AD signin logs

## Use Case

- Build a system with security to analyze logs
- Here location based user assignment is what used
- Certain user will have certain location access
- Create a AuthTable to store username and locations
- Abilityt to change the user, location assignment

## Code

- First Create AuthTable

```
.create table Authtable ( Group:string, Timestamp:datetime, UserId:string, location:string, Message:string )
```

- Now insert some rows

```
.ingest inline into table Authtable <|
midwest,04-08-2022T16:00:00,xx1@domain1.com,US,"Sample ingest"
midwest,04-08-2022T16:00:00,xx1@domain1.com,US,"Sample ingest"

.ingest inline into table Authtable <|
midwest,04-08-2022T16:00:00,xx2@domain1.com,GB,"Sample ingest"

.ingest inline into table Authtable <|
midwest,04-08-2022T16:00:00,xx3@domain1.com,GB,"Sample ingest"
midwest,04-08-2022T16:00:00,xx3@domain1.com,US,"Sample ingest"
midwest,04-08-2022T16:00:00,xx3@domain1.com,AU,"Sample ingest"
midwest,04-08-2022T16:00:00,xx3@domain1.com,,"Sample ingest"
```

- Display the data

```
Authtable
| limit 100
```

- Now load the data for Azure AD singin logs into a table called azureadinternal
- Either using manual blob data into ADX or use event hub

- to get the current use

```
print d=current_principal_details().UserPrincipalName
```

- get logged in users

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == current_principal_details().UserPrincipalName
| limit 400
```

- Find count based on location
- Categorize by identity, operationname, callerIpAddress and Category
- Should filter results only the xx1 user

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx1@domain1.com"
| summarize count() by location, identity, operationName, callerIpAddress, category
```

- Find count based on location
- Categorize by identity, operationname, callerIpAddress and Category
- Should filter results only the xx2 user

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx2@domain1.com"
| summarize count() by location, identity, operationName, callerIpAddress, category
```

- Find count based on location
- Categorize by identity, operationname, callerIpAddress and Category
- Should filter results only the xx3 user

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| summarize count() by location, identity, operationName, callerIpAddress, category
```

- render as bar chart

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx1@domain1.com"
| summarize count() by location, identity, operationName, callerIpAddress, category
| render barchart  
```

- Time bound queries
- Pull only selected time range

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx1@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| summarize count() by location, identity, operationName, callerIpAddress
| project location, identity, operationName, callerIpAddress
```

- Display data to analyze schema

```
azureadinternal
| where properties.status.errorCode != 0
| limit 10000
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx1@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| project location, identity, operationName, callerIpAddress, resultDescription,
properties.clientAppUsed,properties.status.errorCode, properties.status.failureReason,
properties.location.city, properties.location.state, properties.userPrincipalName
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city, errorCode, failureReason, Result description
- Nested objects query
- render as barchart

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx1@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| project location, identity, operationName, callerIpAddress, resultDescription,
properties.clientAppUsed,properties.status.errorCode, properties.status.failureReason,
properties.location.city, properties.location.state
| summarize  count() by location, identity, operationName, callerIpAddress, resultDescription
| render barchart 
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- different user xx3

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| project location, identity, operationName, callerIpAddress, resultDescription,
properties.clientAppUsed,properties.status.errorCode, properties.status.failureReason,
properties.location.city, properties.location.state
| summarize  count() by location, identity, operationName, callerIpAddress, resultDescription
| render barchart 
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
different user

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| project location, identity, operationName, callerIpAddress, resultDescription,
properties.clientAppUsed,properties.status.errorCode, properties.status.failureReason,
properties.location.city, properties.location.state
| summarize  count() by location, identity, operationName, callerIpAddress, resultDescription
| render barchart 
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Aggregate on day basis

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| summarize event_count = count() by bin(['time'], 1d), location, identity, operationName, callerIpAddress, resultDescription
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| project location, identity, operationName, callerIpAddress, resultDescription,
properties.clientAppUsed,properties.status.errorCode, properties.status.failureReason,
properties.location.city, properties.location.state
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Build patterns

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| reduce by resultDescription with threshold=0.35
| project Count, Pattern
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- result description

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by resultDescription
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- result description, clientIpAddress

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
callerIpAddress, resultDescription
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- Caller Ip Address

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
callerIpAddress
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- result description

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
resultDescription
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- error code, clientsappsued, failure reason,city and state

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
resultDescription, tostring(properties.clientAppUsed),
tostring(properties.status.errorCode), tostring(properties.status.failureReason),
tostring(properties.location.city), tostring(properties.location.state)
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- Clientsapp used, error code

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
tostring(properties.clientAppUsed),tostring(properties.status.errorCode)
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- errorCode, Failure Reason

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
tostring(properties.status.errorCode), tostring(properties.status.failureReason)
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- state, city

```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
tostring(properties.status.state), tostring(properties.status.city)
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```

- Time bound queries
- Pull only selected time range
- Pull more data points like clientapps, state, city
- Nested objects query
- Filter only error
- Create time series with various categories to analyze Root Casuse
- Failure Reason


```
azureadinternal | join kind=fullouter (Authtable) on location, $left.location == $right.location
| where UserId == "xx3@domain1.com"
| where ['time'] between (datetime('2022-01-01 00:00:00') .. datetime('2022-05-31 00:00:00'))
| where properties.status.errorCode != 0
| make-series n=count() default=0 on ['time'] in range(datetime('2022-01-01 00:00:00'), datetime('2022-05-31 00:00:00'), 1d) by 
tostring(properties.status.failureReason)
| extend series_stats(n)
| top 3 by series_stats_n_max desc
| render timechart
```