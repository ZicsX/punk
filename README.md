
# Hindi-Punk: Punctuation Prediction Model

Hindi-Punk is a fine-tuned model based on BERT MuRIL (Multilingual Representations for Indian Languages), specifically designed for adding punctuation to Hindi text. Leveraging the powerful capabilities of Google's MuRIL, which excels in understanding and representing multiple Indian languages, Hindi-Punk offers precise punctuation prediction for Hindi, making it a highly effective tool for natural language processing applications involving Hindi text.


## Getting Started

To use the Hindi-Punk model, you'll need to have Python installed on your system along with PyTorch and the Hugging Face Transformers library. If you don't have them installed, you can install them using pip:

```bash
pip install torch transformers
```

## Using the Model

### Step 1: Import Required Libraries

Start by importing the necessary libraries:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from transformers import BertModel
```

### Step 2: Download and Load the Model

The model is hosted on Hugging Face, and you can download it directly using the following code:

```python
# Define the repository name and filename
repo_name = "zicsx/Hindi-Punk"
filename = "Hindi-Punk-model.pth"

# Download the file
model_path = hf_hub_download(repo_id=repo_name, filename=filename)
```

Load the model using PyTorch:

```python
# Define the model classes
class CustomTokenClassifier(nn.Module):
    # ...

class PunctuationModel(nn.Module):
    # ...

# Initialize and load the model
model = PunctuationModel(
    bert_model_name='google/muril-base-cased',
    punct_num_classes=5,
    hidden_size=768
)
model.load_state_dict(torch.load(model_path))
```

### Step 3: Tokenization

Use the tokenizer associated with the model:

```python
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="zicsx/Hindi-Punk", use_fast=True,
)
```

### Step 4: Define Inference Functions

Create functions to perform inference and process the model's output:

```python
def predict_punctuation_capitalization(model, text, tokenizer):
    # ...

def combine_predictions_with_text(text, tokenizer, punct_predictions, punct_index_to_label):
    # ...
```

### Step 5: Run the Model

You can now run the model on your input text:

```python
text = "Your Hindi text here"
punct_predictions = predict_punctuation_capitalization(model, text, tokenizer)
combined_text = combine_predictions_with_text(text, tokenizer, punct_predictions, punct_index_to_label)
print("Combined Text:", combined_text)
```

## Example

Here's an example of how to use the model:

```python
example_text = "सलामअलैकुम कहाँ जा रहे हैं जी आओ बैठो छोड़ देता हूँ हेलो एक्सक्यूज मी आपका क्या नाम है तुम लोगों को बाद में देख लेता हूँ"
punct_predictions = predict_punctuation_capitalization(model, example_text, tokenizer)
combined_text = combine_predictions_with_text(example_text, tokenizer, punct_predictions, punct_index_to_label)
print("Combined Text:", combined_text)
```

## License

This model is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---
