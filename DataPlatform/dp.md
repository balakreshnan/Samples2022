# Data Regulated platform - How to build Enterprise ready Data Platform with Data mesh and Centralized

## Build next generation data platform for New Data regulated enterprise

## Use Cases

- Build Enterprise ready data platform for data regulated enterprises
- Data mesh for business domain or country specific regulation
- Centralized data estate/swamp/lake C-Level reporting
- Data Governance, privacy and security enabled
- Lineage for right to be forgotten and other user centric privacy
- Data Driven insights
- Aritifical intelligence, machine/deep learning driven insights
- Reinforcement learning drive decision making
- Enable business users to become knowledge/Intelligent driven insights assisted users
- Enterprises should be able to automate and apply application life cycle management
- Security enabled for business domains or country / Region specific
- Every business domains or country or data regulation based will have it's own security model
- There will be global security model
- No need to move data
- Centralized data lake can connect to individual Mesh storage as linked service to read data
- Every data mesh will have it's own storage and execution to accomodate mesh or country specific regulation

## Architecture

- Let's build a solution for the above modern data regulated data platform
- Choice is to show how it can be achieved by using Azure synapse analytics - unified platform

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/DataPlatform/images/aifabmesh-Page-2.jpg "Architecture")

## Solution

- Architecture above is split into 2 sections
- Top part which is the high level architecture for data regulated platform
- Bottom part is the common services that each mesh/centralized platform will follow
- Bottom part is also the data governance that cut's across the entire platform
- Common data model can be part of both Mesh and Centralized data platform
- Data security and infra security both are applied
- Data security is how the rows are filtered based on user context. So only authorized users can access necessary part of the system
- Optional block chain can be used for lineage
- Ability to converge any type of data is also covered
- Above architecture can accomodate GDPR, CCPA and other data regulations
- By using Purview we can manage the data governance and lineage
- By using Azure Machine learning, data can extend to build own insights using machine/deep learning
- Mesh can be split based on country for global companies
- Synapse Spark, serverless SQL provides way to connect to country specific storage and read when needed, instead moving data
- This also applies to different business domains as well.
- Multi cloud data store is not added for now will be future based
- Multi engine or data processing is durable