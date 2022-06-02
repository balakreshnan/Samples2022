# # Hugging Face consuming using sdk

## how to consume Hugging Face's python aml sdk

## Details

- This tutorial is to show how easy to call hugging face models and consume within AML workspace
- Also see how easy to download pretrained models from hugging face
- Once downloaded we can consume in python code
- show case how to create pipeline to consume the models

## Steps

- First update the transformer package with this version

```
%pip install transformers==4.17.0
```

- Other wise the generator code was unable to find the config json files
- Now imports

```
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, pipeline
```

- Lets download the hugging face model

```
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./model/tokenizer")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.save_pretrained("./model/weights")
config = GPT2Config.from_pretrained("gpt2")
config.save_pretrained("./model/config")
```

- Create the pipeline

```
generator = pipeline(
    "text-generation",
    model="./model/weights",
    tokenizer="./model/tokenizer",
    config="./model/config",
)
```

- package some text to send
- Prompt is the text to send
- max length is the max length of the generated text
- num_return_sequences is the number of Sentences generated and sent back

```
prompt = "what is the best place in world"
max_length = 50
num_return_sequences = 3
```

- invoke the model inference

```
generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
```

- output

```
[{'generated_text': 'what is the best place in world to learn and be educated about the best fields of life?" he asks. "I believe that what is considered what is considered to be more dangerous is not what the best would do." The question of whether or not'},
 {'generated_text': 'what is the best place in world to buy a book so you can make your own? How is it different if the author is not interested in doing a single book or reading a couple of books in advance because they are not sure what a novel is'},
 {'generated_text': 'what is the best place in world for young scientists (myself included) to study the environment and to think about our planet and the future in the interest of other people?\n\nIn a world with a lot of different weather events and much more'}]
```

- Done