# Azure Open AI with Azure Databricks Completions API

## Using  Datbricks process open ai Completions API

## pre-requisites

- Azure account
- Azure storage account
- Azure DataBricks
- create compute
- install library from https://microsoft.github.io/SynapseML/docs/getting_started/installation/#databricks
- library location - https://mmlspark.blob.core.windows.net/dbcs/SynapseMLExamplesv0.10.1.dbc

## Code

```
import os
from pyspark.sql import SparkSession
from synapse.ml.core.platform import running_on_synapse, find_secret

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()
if running_on_synapse():
    from notebookutils.visualization import display

# Fill in the following lines with your service information
service_name = "openaiservicenamecreated"
deployment_name = "synapseml-openai"
key = "xxxxxxxxxxxxxxxxxxxxxx"
```

- create a sample dataframe

```
df = spark.createDataFrame(
    [
        ("Hello my name is",),
        ("The best code is code thats",),
        ("SynapseML is ",),
    ]
).toDF("prompt")
```

- Open AI completion

```
from synapse.ml.cognitive import OpenAICompletion

completion = (
    OpenAICompletion()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    .setUrl("https://{}.openai.azure.com/".format(service_name))
    .setMaxTokens(200)
    .setPromptCol("prompt")
    .setErrorCol("error")
    .setOutputCol("completions")
)
```

- Run the dataframe

```
from pyspark.sql.functions import col

completed_df = completion.transform(df).cache()
display(
    completed_df.select(
        col("prompt"),
        col("error"),
        col("completions.choices.text").getItem(0).alias("text"),
    )
)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech10.jpg "Architecture")

- Now batch processing

```
batch_df = spark.createDataFrame(
    [
        (["The time has come", "Pleased to", "Today stocks", "Here's to"],),
        (["The only thing", "Ask not what", "Every litter", "I am"],),
    ]
).toDF("batchPrompt")
```

```
batch_completion = (
    OpenAICompletion()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    .setUrl("https://{}.openai.azure.com/".format(service_name))
    .setMaxTokens(200)
    .setBatchPromptCol("batchPrompt")
    .setErrorCol("error")
    .setOutputCol("completions")
)
```

```
completed_batch_df = batch_completion.transform(batch_df).cache()
display(completed_batch_df)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech11.jpg "Architecture")

- Install python openai

```
!pip install openai
```

- create a open ai api rest

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://bbopenai.openai.azure.com/"
openai.api_version = "2022-06-01-preview"
openai.api_key = "xxxxxxxx"

response = openai.Completion.create(
  engine="synapseml-openai",
  prompt="you have reached the essence of hargon this is donna i'll be assisting you with your inquiries today please be informed that this call is being recorded and monitored for quality assurance purposes how may i help you well i bought i i got this from essence of oregon oil umm for shipping handling and handling costs of five ninety nine sample of it and if i want to cancel the order i had to do it within fifteen days and so that is what i want wanted to do i didn't want to i didn't want to get you know like a monthly for what is it eighty three dollars a month OK i can't afford that OK i'm more than happy for me i'm happy to assist you for me to be able to pull up your subscription here could you kindly provide me your first and your last name from caroline C A R O L Y N lake L A K E O L Y and then yes lake yes OK let's just go ahead and pull up here subscription here or your account OK and could you gonna verify your email address please it's lake three nine two one at hotmail dot com how about your shipping address three ten warren avenue number two gillette wyoming eight two seven one six and is your shipping address same as your billing address correct how about your phone number three oh seven six eight oh two oh six eight OK thank you very much for that information miss lake if you don't mind me asking may i know the reason why you want to cancel the subscription well i just i thought i was just getting a a sample order of it you know i was curious have you seen how it worked and everything yes i mean have you already used not all of it but not all of it but i have that i have been using it yes OK i do understand that miss lake this is what i can do for you for you to be able you know to maximize or enjoy the benefits of the organ outgoing to extend your pre trial for another fifteen days with no charge so at least you do have fifteen days to enjoy you know the amazing product and then give us a call back before the end of that fifteen day additional extension period to give us the feedback because what i heard from you is that you haven't used the product that much so you're not you know here did not yet get what the benefits of it so that's the reason why i'm extending your uh period are your pre trial for you to be able to enjoy and discover the benefits of essence of argan OK OK umm what date would that be OK let me check here so you're already extended your pre tier it will end on august twenty fourth so you need to give us a call back before august twenty fourth to give us a feedback if you i mean what happened to the product if it didn't something it gives you the benefits that you need but let's say you love the product like the product you don't need to give us a call back then OK it will be automatically uh you will be receiving another;dr",
  temperature=0.7,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)
```

- display the response

```
print(response.choices[0].text)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech12.jpg "Architecture")