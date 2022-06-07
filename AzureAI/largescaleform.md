# Process Large scale forms in optimized way

## Using Azure AI to process forms

## Use Case

- Have millions of forms to process
- Like 15 to 20 million pages
- These are forms
- Forms might have 3-15 pages
- Data to pull might be 2 or 3 pages
- Split the pages to process in Form recognizer to reduce AI cost
- Use python ai library to filter the pages needed for AI services

## Architecture

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/largeformprocessing.jpg "Architecture")

## 2 Parts processing

### Azure Python Function

- Python function to process PDF to only pick pages needed to process in AI
- Instead of 15 million pages can be reduced to 2 or 3 million pages
- Using existing open source packages like pytesseract to pull only pages needed
- Scale pdf processing using azure functions

### Azure C# function to process Form Recognizer

- Functions to take the reduced pages and send to Form Recognizer
- Process form recognizer output save to SQL for further reporting
- Azure analytics is used for further data processing
- Scale functions as needed to process forms
- Reduced form sends 2 to 3 million request rather than 15 million pages to AI services

## Done