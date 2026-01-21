#!/usr/bin/env python
# coding: utf-8

# - **Important: This Notebook requires GPU access on Google Colab. You can access this notebook using this link: https://colab.research.google.com/drive/1QsH6BV3n51-YbmqWXPXyk2a6UDSoz9YO?usp=sharing** 

# # TASK 1. UNDERSTAND THE PROBELM STATMENT & KEY LEARNING OBJECTIVES

# # TASK 2. EXPLORE HUGGING FACE!

# - Hugging Face: https://huggingface.co/models

# **PRACTICE OPPORTUNITY:**
# 
# - **Visit [huggingface.co/models](https://huggingface.co/models) and filter for Text Generation on the left sidebar, find the "Tasks" filter and select "Text Generation".**
# - **Sort by Downloads: Near the top right, you can sort the models. Try sorting by "Most Downloads". What are some of the most popular text generation models you see?**
# - **Search for models with "phi", "gemma", "qwen", or "llama" in their names, often with numbers like "2b", "1.8b", "1.5", or "4k-instruct". These indicate smaller model sizes more likely to run on our Colab GPU. Click on one (e.g., `microsoft/Phi-3-mini-4k-instruct`). Notice the "Files and versions" tab and the model card (README) explaining the model.**
# - **Test Phi-3-mini-4k-instruct Model with the following prompt "Explain Newton's second law of motion as if I am 5 year old"**
# 

# In[1]:


import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant teaching children aged 5."}, 
    {"role": "user", "content": "Explain Newton's second law of motion"}, 
 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])


# # TASK 3. INSTALL KEY LIBRARIES, OBTAIN HF ACCESS TOKENS, & GPU CHECK

# We need to install the necessary Python libraries. We'll also need to potentially log in to Hugging Face if we want to use certain models (like some versions of Llama or Gemma) that require user agreement.
# 
# **Installing Libraries:**
# 
# *   `transformers`: The core Hugging Face library for models and tokenizers.
# *   `accelerate`: Helps run models efficiently across different hardware (like GPUs) and use less memory.
# *   `bitsandbytes`: Enables model quantization (like loading in 4-bit or 8-bit), drastically reducing memory usage. Essential for running decent models on free Colab GPUs!
# *   `torch`: The underlying deep learning framework (PyTorch).
# *   `pypdf`: A library to easily extract text from PDF files.
# 

# In[2]:


# Let's install key libraries
print("Installing necessary libraries...")
get_ipython().system('pip install -q transformers accelerate bitsandbytes torch pypdf gradio')
print("Libraries installed successfully!")


# In[3]:


# Let's import these libraries
import torch  # PyTorch, the backend for transformers
import pypdf  # For reading PDFs
import gradio as gr  # For building the UI
from IPython.display import display, Markdown  # For nicer printing in notebooks
print("Core libraries imported.")


# **Hugging Face Hub Login:**
# 
# Some models on the Hugging Face Hub are "gated," meaning you need to agree to their terms and conditions before downloading. Logging in allows the `transformers` library to download these models if needed.
# 
# *   **Get a Hugging Face Token:**
#     1.  Go to [huggingface.co](https://huggingface.co/).
#     2.  Sign up or log in.
#     3.  Click your profile picture (top right) -> Settings -> Access Tokens.
#     4.  Create a new token (a 'read' role is usually sufficient).
#     5.  Copy the generated token. **Treat this like a password!**
# *   **Log in within Colab:** We'll use a helper function from the `huggingface_hub` library.

# In[4]:


import os
from huggingface_hub import login, notebook_login
print("Attempting Hugging Face login...")

# Use notebook_login() for an interactive prompt in Colab/Jupyter
# This is generally preferred for notebooks.

notebook_login()
print("Login successful (or token already present)!")


# In[ ]:


# Check if GPU is available (essential for running these models)
# Why GPU is Important: LLMs involve billions of calculations (matrix multiplications).
# GPUs are designed for massive parallel processing, making these calculations thousands of times faster than a standard CPU.
# Running these models on a CPU would take an impractically long time (hours for a single answer instead of seconds/minutes).
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    # Set default device to GPU
    torch.set_default_device("cuda")
    print("PyTorch default device set to CUDA (GPU).")
else:
    print("WARNING: No GPU detected. Running these models on CPU will be extremely slow!")
    print("Make sure 'GPU' is selected in Runtime > Change runtime type.")


# In[5]:


# Helper function for markdown display
def print_markdown(text):
    """Displays text as Markdown in Colab/Jupyter."""
    display(Markdown(text))


# **PRACTICE OPPORTUNITY:**
# - **Go to "Runtime" -> "Change runtime type" and explore available hardware accelerator types in your Colab.**
# - **Compare CPUs, T4 GPU, A100 GPU from a performance standpoint. What speed gains can developers get by shifting from a CPU to A100 GPU?**

# In[ ]:





# # TASK 4. HUGGING FACE TRANSFORMERS LIBRARY: PIPELINES

# - **`transformers`:** A Python library that provides a standardized way to download, load, and use models from the Hub with just a few lines of code. Key classes:
#     *   `pipeline()`: A high-level, easy-to-use abstraction for common tasks (like text generation, summarization). Great for quick tests and beginners.
#     *   `AutoTokenizer`: Automatically downloads the correct "tokenizer" for a model. A tokenizer converts human-readable text into numerical IDs the model understands.
#     *   `AutoModelFor...`: Automatically downloads the correct model architecture and pre-trained weights (e.g., `AutoModelForCausalLM` for text generation models like GPT, Llama, Gemma).
# - **Other Libraries:** HF also develops libraries like `accelerate` (for efficient loading/distributed training), `datasets` (for handling datasets), and `evaluate` (for model evaluation metrics).
# 

# - Attention Is All You Need: https://arxiv.org/abs/1706.03762

# In[3]:


# The pipelines are a great and easy way to use models for inference.
# These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks
# Those tasks include Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.
from transformers import pipeline

# Load a sentiment classifier model on financial news data
# Check the model here: https://huggingface.co/ProsusAI/finbert
pipe = pipeline(model = "ProsusAI/finbert")
pipe("Apple lost 10 Million dollars today due to US tarrifs")


# **PRACTICE OPPORTUNITY:**
# - **Use Hugging Face's pipeline() function and the ProsusAI/finbert model to analyze sentiment in financial news using the following examples:**
#   - **news_samples = ["Tesla stock surged after record-breaking quarterly earnings.", "Microsoft is scheduled to release its earnings report next week."]**
# - **Compare the results with Hugging Face's finbert Inference API and perform a sanity check**
# 

# In[4]:


from transformers import pipeline

# Load a sentiment classifier model on financial news data
# Check the model here: https://huggingface.co/ProsusAI/finbert
pipe = pipeline(model = "ProsusAI/finbert")
pipe(["Tesla stock surged after record breaking quarterly earnings", "Microsoft is scheduled to release its earnings report next week"])


# # TASK 5. HUGGING FACE TRANSFORMERS LIBRARY: AUTOTOKENIZERS

# In[5]:


# Let's explore AutoTokenizer
# A tokenizer converts text into numerical IDs that the model understands
# Check a demo for OpenAI's Tokenizers here: https://platform.openai.com/tokenizer
from transformers import AutoTokenizer

# Load tokenizer for GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode text to token IDs
tokens = tokenizer("Hello everyone and welcome to LLM and AI Agents Bootcamp")
print(tokens['input_ids'])


# In[6]:


new_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokens_new = new_tokenizer("Generative AI is transforming the future of work")
print(tokens_new['input_ids'])


# **PRACTICE OPPORTUNITY:**
# - **Use the AutoTokenizer.from_pretrained() method to load a different tokenizer (e.g., "bert-base-uncased" or "facebook/opt-125m")**
# - **Try encoding a new sentence like: "Generative AI is transforming the future of work.**
# - **Print both: The raw tokens (tokenizer.tokenize(text)) and the input IDs (tokenizer(text)["input_ids"])**
# - **Compare the tokenization between two models. What differences do you observe when handling spaces between words?**

# In[7]:


new_tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_new = new_tokenizer("Generative AI is transforming the future of work")
print(tokens_new['input_ids'])


# # TASK 6. HUGGING FACE TRANSFORMERS LIBRARY: AutoModelForCausalLM

# AutoModelForCausalLM is a Hugging Face class that automatically loads a pretrained model for causal (left-to-right) language modeling, such as GPT, LLaMA, or Gemma.
# 
# Let's get hands-on and load a model! We'll start with a relatively small but capable model that should fit comfortably in Colab's free tier GPU memory, thanks to quantization.
# 
# **Key Steps:**
# 
# 1.  **Choose a Model ID:** We need the unique identifier from the Hugging Face Hub (e.g., `"google/gemma-2b-it"` or `"microsoft/Phi-3-mini-4k-instruct"`).
# 2.  **Load the Tokenizer:** Use `AutoTokenizer.from_pretrained(model_id)` to get the specific tokenizer for that model.
# 3.  **Load the Model:** Use `AutoModelForCausalLM.from_pretrained(...)` with crucial arguments:
#     *   `model_id`: The identifier.
#     *   `torch_dtype=torch.float16` (or `bfloat16`): Loads the model using 16-bit floating point numbers instead of 32-bit, saving memory.
#     *   `load_in_4bit=True` or `load_in_8bit=True`: This is **quantization** via `bitsandbytes`. It further reduces memory by representing model weights with fewer bits (4 or 8 instead of 16/32). Essential for free Colab! 4-bit saves more memory but might have a tiny impact on quality compared to 8-bit.
#     *   `device_map="auto"`: Tells `accelerate` to automatically figure out how to spread the model across available devices (primarily the GPU in our case).
# 4.  **Combine Tokenizer and Model (Optional but common):** Using the `pipeline` function is often simpler for basic text generation. It handles tokenization, model inference, and decoding back to text for you.
# 

# In[8]:


get_ipython().system('pip install -U bitsandbytes')


# In[9]:


# Let's import AutoModelForCasualLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Let's choose a small, powerful model suitable for Colab.
# Alternatives you could try (might need login/agreement):
# model_id = "unsloth/gemma-3-4b-it-GGUF"
# model_id = "Qwen/Qwen2.5-3B-Instruct"
model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "unsloth/Llama-3.2-3B-Instruct"


# In[12]:


# Let's load the Tokenizer
# The tokenizer prepares text input for the model
# trust_remote_code=True is sometimes needed for newer models with custom code.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
print("Tokenizer loaded successfully.")


# In[13]:


# Let's Load the Model with Quantization

print(f"Loading model: {model_id}")
print("This might take a few minutes, especially the first time...")

# Create BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                         bnb_4bit_compute_dtype = torch.float16,  # or torch.bfloat16 if available
                                         bnb_4bit_quant_type = "nf4",  # normal float 4 quantization
                                         bnb_4bit_use_double_quant = True  # use nested quantization for more efficient memory usage
                                         )

# Load the model with the quantization config
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config = quantization_config,
                                             device_map = "auto",
                                             trust_remote_code = True)


# In[10]:


# Let's define a prompt
prompt = "Explain how Electric Vehicles work in a funny way!"


# In[11]:


prompt = "What is the capital of France?"


# In[12]:


# Method 1: Let's test the model and Tokenizer using the .generate() method!

# Let's encode the input first
inputs = tokenizer(prompt, return_tensors = "pt")

# Then we will generate the output
outputs = model.generate(**inputs, max_new_tokens = 1000)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print_markdown(response)


# In[18]:


# Method 2: alternatively, you can create a pipeline that includes your model and tokenizer
# The pipeline wraps tokenization, generation, and decoding

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = "auto", # Match model dtype
                device_map = "auto" # Ensure pipeline uses the same device mapping
                )


outputs = pipe(prompt,
               max_new_tokens = 1000, # max_new_tokens limits the length of the generated response.
               temperature = 1, # temperature controls randomness (lower = more focused).
               )

# Print the generated text
print_markdown(outputs[0]['generated_text'])