# SecOps Analytics platform Architecture using Azure Data Explorer  (Part of Azure Synapse Analtyics)

## Build your own Security Analytics platform

## Use Case

- Given Cyber security is a big problem for all organizations, we need to build a platform to analyze the security events and provide insights to the security team to take action.
- No organization is bound to one tool to run their business
- Security is a big problem for all systems and tools used on a organization
- Goal is to bring all the security events from different tools and analyze them to provide insights to the security team
- Ability to do cyber threat analysis and provide insights to the security team
- Provide cyber threat intelligence to CISO and security team
- There is also lots of challenges in building a security analytics platform
- Most focus on Technology space
- Some do in Infrastructure space
- Some do in Application space
- How do we build something that can cover all the above
- How do we handle pyshical, virtual and cyber security
- Given we have everything run in computers, how do we build a platform that can analyze all the security events

## Goal

- Build a platform that can analyze all the security events from different tools and sources
- Ability to be agile to add new sources and tools
- Ability to agile for find new threats and insights
- Ability to do root cause analysis
- Not a complete solution, rather a platform that can be used to build a complete solution
- Ability to store lots of data for long time
- Ability to do real time ad-hoc analysis
- No pre baked dashboard's but ability to build your own dashboard
- Ability to find new use cases and threats

## Architecture

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SecOps/images/SecOpsDraw.jpg "Architecture")

## Architecture Explained

- Above architecture goes from left to right
- On the left side is all our sources
- Sources can be any systems or applications
- All the data collected flows through Event hub whic acts like ingestion agent
- Event hub can scale as needed.
- From Event hub Azure data explorer has a connector to pull Event hub data and store in temp tables
- Then we can transform into the final tables with proper schema
- Now there is lots of possible on schema changes
- We can bring the data as dynamics schema and then transform into final schema to fit to source to preserve the business logic
- Azure Data explorer has option to store hot and cold data with just configuration
- Sizing of Azure data explorer is also scaled as needed
- Bringing cold data to query is also pretty simple with Azure Data explorer
- Design for scale and agility
- Cost optimized design
- No need to do ETL/ELT work
- We can extend data processing if needed using Azure Synapse Spark with Azure data explorer connector
- Below you will see the architecture with Azure data explorer extended to do large scale machine learning
- Also ability to process big data using spark clusters

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SecOps/images/SecOpsDraw1.jpg "Architecture")

- Most of the ad-hoc analytsis can be executed using Azure data explorer
- Kusto is very power ful language
- Also has ability to do machine learning like Forecasting, anomaly detection, clustering
- If you want to build your own model depending on the data size and complexity, you can use Azure Synapse Spark with Azure data explorer connector
- Now why do we need multiple options, not all use cases in machine learning or deep learning needs lots of data, in that case Azure machine learning is faster and improves productivity
- if the data set size increased then we have to use Azure Synapse Spark to get the parallelism and scale
- The above architecture can also be extension of existing cyber security tools available in the market

## Use cases

- Threat detection
- Real time alerts
- Automated Malware analysis
- Threat Monitoring
- Patch analysis and prirotization
- Actor profiling
- Incident Response
- Operation intelligence reporting
- Strategic intelligence reporting
- Campaign Tracking
- Insider threat
- Threat Research
- Deception operations
- User behavior analysis and profiling between different systems
- Network Threat detection
- Detect hacking attempts
- Detect fraud activities