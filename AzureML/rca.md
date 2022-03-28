# Root Cause analysis using machine learning

## Assign proper categories for issues or problems

## Use Case

- In any automateion or machine automation space, it's important to understand the issue or problem
- To understand the issue with complex Hiearchical machinery, understanding and assigning proper categories is important
- Once we find where the issue and problem falls, then it's easy for us to find the solution
- Most of the time is spent on understanding the issue and assigning proper categories
- How can machine learning solve this problem

## Process

- First get a data set with past history of issues and problems
- Your past history should also have the categories classified
- The question to ask is is that correct categories are assigned to the issue or problem
- Lets find a SME to reassign the categories to the right one
- For example may be for 25 different classes, about 70,000 rows of data was collected
- Once the data is classified properly, split the data set for training and testing
- We used multiclass classification to train the model
- For example take 60,000 for training and 10,000 for testing
- Use Azure ML automl to train the model
- Use test data to valdiate the model with test model feature
- Next validate with new data set that the model hasn't seen which are not categorized
- When you take the new data set, no need to add categories, but model assigns category
- We can even create layers or level of categories like cat1, cat2, cat3, cat4.
- Each level or layer should be ML model in sequence to assign like cat1 -> cat2 -> cat3 -> cat4
- Expose the model as REST API or create batch api to consume the model

## done