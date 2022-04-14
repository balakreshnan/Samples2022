# Azure AD signin logs analysis queries for Azure Data Explorer

## Use ADX queries to analyze Azure AD signin logs

## Use Case

- Build a system with security to analyze logs
- Here location based user assignment is what used
- Certain user will have certain location access
- Create a AuthTable to store username and locations
- Abilityt to change the user, location assignment

## Code

- Azure AD signin logs

```
iamlogssignin
| limit 10

auditlogs1
| limit 100

.drop extents <| .show table iamlogssignin extents 

.show table iamlogssignin policy  ingestionbatching 


.alter column iamlogssignin.data type=dynamic

.create table iamlogssignin ingestion json mapping 'iamlogssigninMapping' '[{"column":"data","path":"$","datatype":"dynamic"}]'

iamlogssignin
| mv-expand r = data.records
| project  rectime = todatetime(r.['time']), resourceId = tostring(r.resourceId), 
operationName = tostring(r.operationName),  operationVersion = tostring(r.operationVersion), category= tostring(r.category), 
tenantId = tostring(r.tenantId), resultType = toint(r.resultType), resultSignature = tostring(r.resultSignature), durationMs = todouble(r.durationMs), 
callerIpAddress = tostring(r.callerIpAddress), correlationId = tostring(r.correlationId), 
identity = tostring(r.identity), Level = toint(r.Level), location = tostring(r.location),
properties = todynamic(r.properties)

.create-or-alter function iamlogssigninparsing(){
    iamlogssignin
| mv-expand r = data.records
| project  rectime = todatetime(r.['time']), resourceId = tostring(r.resourceId), 
operationName = tostring(r.operationName),  operationVersion = tostring(r.operationVersion), category= tostring(r.category), 
tenantId = tostring(r.tenantId), resultType = toint(r.resultType), resultSignature = tostring(r.resultSignature), durationMs = todouble(r.durationMs), 
callerIpAddress = tostring(r.callerIpAddress), correlationId = tostring(r.correlationId), 
identity = tostring(r.identity), Level = toint(r.Level), location = tostring(r.location),
properties = todynamic(r.properties)
}

.set aadsignins <|
    iamlogssigninparsing()
    | limit 0


//  Create an update policy to transfer landing to landingTransformed
.alter table aadsignins policy update
@'[{"IsEnabled": true, "Source": "iamlogssignin", "Query": "iamlogssigninparsing", "IsTransactional": true, "PropagateIngestionProperties": false}]'

aadsignins
| limit 10

//  Alter retention policy as this is also only for end-user queries
.alter table iamlogssignin policy retention "{'SoftDeletePeriod': '00:00:00', 'Recoverability':'Enabled'}"
```

- Audit logs

```
auditlogs1
| limit 10

external_table("auditlogs")
| limit 10

.alter column auditlogs1.data type=dynamic
.create table auditlogs1 ingestion json mapping 'auditlogs1Mapping' '[{"column":"data","path":"$","datatype":"dynamic"}]'
//update the connection setting if it's already created other wise do this and use this mapping

.drop extents <| .show table auditlogs1 extents 

.show table auditlogs1 policy  ingestionbatching 


auditlogs1
| mv-expand r = data.records
| project  rectime = todatetime(r.['time']), resourceId = tostring(r.resourceId), 
operationName = tostring(r.operationName),  operationVersion = tostring(r.operationVersion), category= tostring(r.category), 
tenantId = tostring(r.tenantId), resultSignature = tostring(r.resultSignature), durationMs = todouble(r.durationMs), 
callerIpAddress = tostring(r.callerIpAddress), correlationId = tostring(r.correlationId), 
identity = tostring(r.identity), Level = toint(r.Level), location = tostring(r.location),
properties = todynamic(r.properties)

.create-or-alter function auditlogs1parsing(){
auditlogs1
| mv-expand r = data.records
| project  rectime = todatetime(r.['time']), resourceId = tostring(r.resourceId), 
operationName = tostring(r.operationName),  operationVersion = tostring(r.operationVersion), category= tostring(r.category), 
tenantId = tostring(r.tenantId), resultSignature = tostring(r.resultSignature), durationMs = todouble(r.durationMs), 
callerIpAddress = tostring(r.callerIpAddress), correlationId = tostring(r.correlationId), 
identity = tostring(r.identity), Level = toint(r.Level), location = tostring(r.location),
properties = todynamic(r.properties)
}

.set aadauditlogs <|
    auditlogs1parsing()
    | limit 0


//  Create an update policy to transfer landing to landingTransformed
.alter table aadauditlogs policy update
@'[{"IsEnabled": true, "Source": "auditlogs1", "Query": "auditlogs1parsing", "IsTransactional": true, "PropagateIngestionProperties": false}]'

aadauditlogs
| limit 10

```