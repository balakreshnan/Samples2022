# Synapse ML batch processing Health Care Text analytics

## Azure Healthcare text analytics api using SynapseML

## Prerequisites

- Document - https://microsoft.github.io/SynapseML/docs/next/features/cognitive_services/CognitiveServices%20-%20Overview/#text-analytics-for-health-sample
- Create Azure cognitive services account
- Create Azure storage account
- Create Azure synapse analytics workspace
- Create a spark compute cluster
- Use pyspark

## Steps

- get the synapseml installed

```
%%configure -f
{
"name": "synapseml",
"conf": {
"spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.9.5-13-d1b51517-SNAPSHOT",
"spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
"spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.12,org.scalactic:scalactic_2.12,org.scalatest:scalatest_2.12",
"spark.yarn.user.classpath.first": "true"
}
}
```

- Import the cognitive services

```
from synapse.ml.cognitive import *
```

- Now bring the keyvault secrets for cognitive service and sql password

```
key = TokenLibrary.getSecret("accvault1", "bbcogtext", "accvault1")
password = TokenLibrary.getSecret("accvault1", "bbaccdbauth", "accvault1")
```

- configure the JDBC

```
jdbcHostname = "sqlserver.database.windows.net"
jdbcDatabase = "dbname"
jdbcPort = 1433
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)
connectionProperties = {
"user" : "sqladmin",
"password" : password,
"driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}
```

- read the azure sql data

```
Spdf = spark.read.jdbc(url=jdbcUrl, table="dbo.precribdata", properties=connectionProperties).limit(100)
display(Spdf)
```

- now connect to health care api

```
healthcare = (HealthcareSDK()
    .setSubscriptionKey(key)
    .setLocation("centralus")
    .setLanguage("en")
    .setOutputCol("response"))

display(healthcare.transform(df))
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/healthcare1.jpg "Entire Flow")