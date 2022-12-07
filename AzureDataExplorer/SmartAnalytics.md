# Smart Space Analytics

## ADX Queries

## Code

- Office location table

```
.create table Officelocations ( callerIpAddress:string, city:string , state:string, OfficeProvider:string )
```

- we loaded from csv file with above column names

- Employee details

```
.create table employees ( EmployeeName:string, ID:string , city:string, state:string, latitude:string, longitude:string )

.set-or-append employees <| (iamlogssignin
    | mv-expand  r = data.records
    | project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
    rlocal = tostring(r.location), city = tostring(r.properties.location.city), state = tostring(r.properties.location.state),
    latitude = tostring(r.properties.location.geoCoordinates.latitude),
    longitude = tostring(r.properties.location.geoCoordinates.longitude),
    logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName),
    userid = tostring(r.properties.userid)
    | distinct identity, userid, city, state, latitude, longitude)
```

- Maps

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
//| distinct callerIpAddress, identity
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where logintime  between (_startTime.._endTime)
| where OfficeProvider has_any (_OfficeProvider)
| where identity  has_any (_employee)
| project geocolatitude, geocolongitude, operationName, OfficeProvider, identity, callerIpAddress
| summarize peoplecount = dcount(callerIpAddress), identitycount=dcount(identity) by OfficeProvider, geocolongitude = tostring(geocolongitude), geocolatitude = tostring(geocolatitude)
//| render scatterchart with (kind = map)
//| render scatterchart with (kind = map, xcolumn = geocolongitude, ycolumns = geocolatitude, series = OfficeProvider)
```


- Pie chart

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| where OfficeProvider   has_any (_OfficeProvider)
| summarize location=max(OfficeProvider), peoplecount = dcount(identity) by OfficeProvider
```

- Anomaly Detection

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize location=max(OfficeProvider), peoplecount = dcount(identity) by bin(logintime, 1d)
```

- Autocluster

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
//| summarize location=max(OfficeProvider), peoplecount = count() by bin(logintime, 1d)
| evaluate autocluster()
```

- Forecasting

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize location=max(OfficeProvider), peoplecount = dcount(identity) by bin(logintime, 1d)
| project location, peoplecount, logintime
| make-series y=max(peoplecount) on logintime from _startTime to _endTime+1*7*1d step 1d // create a time series of 5 weeks (last week is empty)
| extend y_forcasted = series_decompose_forecast(todynamic(tostring(y)), 1*7, 168, 'linefit', 0.4)  // forecast a week forward
//| render timechart 
```

- Trends by office

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize location=max(OfficeProvider), peoplecount = dcount(identity) by bin(logintime, 1d)
```

- Employee count by office

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize location=max(OfficeProvider), peoplecount = dcount(identity) by bin(logintime, 1d)
```

- Trends by office

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize location=dcount(OfficeProvider) by bin(logintime, 1d)
```

- Trends by Poeple

```
iamlogssignin
| mv-expand  r = data.records
| project callerIpAddress = tostring(r.callerIpAddress), identity = tostring(r.identity), 
rlocal = tostring(r.location), loc1city = tostring(r.properties.location.city), loc1state = tostring(r.properties.location.state),
geocolatitude = tostring(r.properties.location.geoCoordinates.latitude),
geocolongitude = tostring(r.properties.location.geoCoordinates.longitude),
logintime = todatetime(tostring(r.['time'])), operationName = tostring(r.operationName)
| join kind=inner Officelocations on $left.callerIpAddress == $right.callerIpAddress
| where OfficeProvider has_any (_OfficeProvider)
| where logintime  between (_startTime.._endTime)
| where identity  has_any (_employee)
| summarize peoplecount = dcount(identity) by bin(logintime, 1d)
```

- Above are those queries