#!/usr/bin/env python
# coding: utf-8



# # TASK 2.LET'S READ A SAMPLE IMAGE

# In[2]:


# Let's install and import OpenAI Package
get_ipython().system('pip install --upgrade openai')
from openai import OpenAI  

# Let's import os Python module, which stands for "Operating System"
# The os module lets you interact with your computer’s operating system
import os

# This will be used to load the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key = openai_api_key)
print("OpenAI client successfully configured.")

# Let's view the first few characters in the key
print(openai_api_key[:15])


# In[3]:


# Define a helper function named "print_markdown" to display markdown
from IPython.display import display, Markdown  

def print_markdown(text):
    """Displays text as Markdown in Jupyter."""
    display(Markdown(text))


# In[4]:


# Let's try loading and displaying a sample image
# Before sending images to OpenAI API, we need to learn how to load and view them in our notebook
# We'll use the Pillow library (imported as PIL) for this task

# Import Pillow for image handling
from PIL import Image  


# In[6]:


# IMPORTANT: Replace this with the path to your downloaded image file
# Make sure the image file is in the same directory as the notebook
image_filename = "C:\\Users\\mayur\\OneDrive\\Desktop\\LLM and Agentic AI Bootcamp Materials\\Module 2 - Vision GPT\\images\\pizza_slice.png"  # <--- CHANGE THIS to your image file name

# Use Pillow's Image.open() to load the image from the file
img = Image.open(image_filename)
print(f"Image '{image_filename}' loaded successfully.")
print(f"Format: {img.format}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")

# Use IPython.display to show the image directly in the notebook output
display(img)

# Keep the loaded image object in a variable for later use
image_to_analyze = img


# **PRACTICE OPPORTUNITY:**
# - **Download another food image (e.g., a banana, a slice of pizza) and save it to your project folder. Update the `image_filename` variable in the code cell above to the new filename and run the cell again. Does it load and display correctly?**
# - **Look at the printed output for the `Format`, `Size`, and `Mode` of your images.**

# In[ ]:





# In[ ]:





# # TASK 3. UNDERSTAND PROMPT ENGINEERING FUNDAMENTALS





# In[9]:


#when mayur ran this in Claude Sonnet

# Classification: OPTIMISTIC
# Top 3 Influential Phrases:

# "ninth consecutive quarter of revenue expansion" - This demonstrates sustained, consistent growth over more than two years, signaling strong underlying business fundamentals and competitive positioning rather than a one-time performance spike.
# "Gross margin expanded to 42.1%, reflecting improved production efficiency and favourable pricing strategies" - Margin expansion combined with efficiency gains shows the company has pricing power and operational excellence—two critical indicators of a healthy, well-managed business with competitive advantages.
# "reinforcing our strong liquidity position" - Strong cash flow generation ($50M from operations) coupled with this statement indicates financial resilience and flexibility to invest in growth, weather downturns, or return capital to shareholders.

# Actionable Recommendations:

# Consider accumulating shares on any near-term weakness - The combination of consistent revenue growth, margin expansion, and strong cash generation suggests a company executing well. Current holders should maintain positions while new investors could add on pullbacks.
# Monitor management's capital allocation strategy - With strong liquidity and cash flow, watch for announcements regarding dividends, share buybacks, or strategic investments that could enhance shareholder value and signal management's confidence.
# Track competitive positioning in air suspension systems - The sustained revenue growth indicates strong market demand. Investigate whether this is market share gain or industry growth to assess long-term runway and potential valuation expansion.


# In[ ]:





# # TASK 4. LET'S PERFORM IMAGE RECONGITION USING OPENAI'S VISION API




# The io module in Python provides tools for working with streams of data
# like reading from or writing to files in memory
import io  

# Used for encoding images for OpenAI's API
import base64  


# In[11]:


# This function converts an image into a special text format (called base64)
# This is used if we want to send an image to OpenAI’s API

# This function works with two types of inputs: 
# (1) A file path: a string that tells the function where the image is stored on your computer.
# (2) An image object: a photo already loaded in memory using the PIL library (Python Imaging Library).

def encode_image_to_base64(image_path_or_pil):
    if isinstance(image_path_or_pil, str):  # If it's a file path
        # Check if the file exists
        if not os.path.exists(image_path_or_pil):
            raise FileNotFoundError(f"Image file not found at: {image_path_or_pil}")
        with open(image_path_or_pil, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    elif isinstance(image_path_or_pil, Image.Image):  # If it's a PIL Image object
        buffer = io.BytesIO()
        image_format = image_path_or_pil.format or "JPEG"  # Default to JPEG if format unknown
        image_path_or_pil.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("Input must be a file path (str) or a PIL Image object.")


# In[12]:


# Let's define a function that queries OpenAI's vision model with an image
def query_openai_vision(client, image, prompt, model = "gpt-4o", max_tokens = 100):
    """
    Function to query OpenAI's vision model with an image
    
    Args:
        client: The OpenAI client
        image: PIL Image object to analyze
        prompt: Text prompt to send with the image
        model: OpenAI model to use (default: gpt-4o)
        max_tokens: Maximum tokens in response (default: 100)
        
    Returns:
        The model's response text or an error message
    """

    # Encode the image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        # Construct the message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens,
        )

        # Extract the response
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error calling API: {e}"


# In[13]:


# Let's define our text prompt that will be sent with the image
food_recognition_prompt = """
Context: I'm analyzing a food image for a calorie-tracking application.
Instruction: Please identify the food item in this image.
Input: [The image I'm about to share]
Output: Provide the name of the food, a brief description of what you see, and if possible, mention its typical ingredients or nutritional profile.
"""
print(f"{food_recognition_prompt}")


# In[14]:


# Let's call the function and send it an image!
print("🤖 Querying OpenAI Vision...")
openai_description = query_openai_vision(
    openai_client, 
    image_to_analyze, 
    food_recognition_prompt
)
print_markdown(openai_description)





# # TASK 5. LET'S OBTAIN THE NUMBER OF CALORIES USING VISION API

# In[17]:


# Let's define a structured prompt to ensure consistent model output
structured_nutrition_prompt = """
# Nutritional Analysis Task

## Context
You are a nutrition expert analyzing food images to provide accurate nutritional information.

## Instructions
Analyze the food item in the image and provide estimated nutritional information based on your knowledge.

## Input
- An image of a food item

## Output
Provide the following estimated nutritional information for a typical serving size or per 100g:
- food_name (string)
- serving_description (string, e.g., '1 slice', '100g', '1 cup')
- calories (float)
- fat_grams (float)
- protein_grams (float)
- confidence_level (string: 'High', 'Medium', or 'Low')

**IMPORTANT:** Respond ONLY with a single JSON object containing these fields. Do not include any other text, explanations, or apologies. The JSON keys must match exactly: "food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level". If you cannot estimate a value, use `null`.

Example valid JSON response:
{
  "food_name": "Banana",
  "serving_description": "1 medium banana (approx 118g)",
  "calories": 105.0,
  "fat_grams": 0.4,
  "protein_grams": 1.3,
  "confidence_level": "High"
}
"""


# In[18]:


# Let's call OpenAI API with the image and the new structured prompt
openai_nutrition_result = query_openai_vision(client = openai_client,
                                              image = image_to_analyze,
                                              prompt = structured_nutrition_prompt,)

print_markdown(openai_nutrition_result)



# In[20]:


# Let's define a structured prompt to ensure consistent model output
structured_nutrition_prompt_modified = """
# Nutritional Analysis Task

## Context
You are a nutrition expert analyzing food images to provide accurate nutritional information.

## Instructions
Analyze the food item in the image and provide estimated nutritional information based on your knowledge.

## Input
- An image of a food item

## Output
Provide the following estimated nutritional information for a typical serving size or per 100g:
- food_name (string)
- serving_description (string, e.g., '1 slice', '100g', '1 cup')
- calories (float)
- fat_grams (float)
- protein_grams (float)
- confidence_level (string: 'High', 'Medium', or 'Low')
- sugar_grams

**IMPORTANT:** Respond ONLY with a single JSON object containing these fields. Do not include any other text, explanations, or apologies. The JSON keys must match exactly: "food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level". If you cannot estimate a value, use `null`.

Example valid JSON response:
{
  "food_name": "Banana",
  "serving_description": "1 medium banana (approx 118g)",
  "calories": 105.0,
  "fat_grams": 0.4,
  "protein_grams": 1.3,
  "confidence_level": "High",
  "sugar_grams": 10
}
"""


# In[21]:


# Let's call OpenAI API with the image and the new structured prompt
openai_nutrition_result = query_openai_vision(client = openai_client,
                                              image = image_to_analyze,
                                              prompt = structured_nutrition_prompt_modified,)

print_markdown(openai_nutrition_result)


# In[ ]:





# In[ ]:






# In[ ]:


# Import Pillow for image handling
from PIL import Image  

# IMPORTANT: Replace this with the path to your downloaded image file
# Make sure the image file is in the same directory as the notebook
image_filename = "images/pizza_slice.png"  # <--- CHANGE THIS to your image file name

# Use Pillow's Image.open() to load the image from the file
img = Image.open(image_filename)
print(f"Image '{image_filename}' loaded successfully.")
print(f"Format: {img.format}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")

# Use IPython.display to show the image directly in the notebook output
display(img)

# Keep the loaded image object in a variable for later use
image_to_analyze = img



# ```text
# Context: 
# 
# You are a senior financial analyst with expertise in private equity.
# 
# Instruction: 
# 
# Carefully review the provided earnings call transcript of Solid Power. Based on the language, sentiment, and key financial and operational signals shared by the CEO, classify the CEO’s tone as one of the following: Optimistic, Cautious, or Concerning. Your analysis should identify specific language cues, strategic outlooks, and underlying business sentiment.
# 
# Input:
#   
# "Operator: Good morning, and welcome to Solid Power's Fourth Quarter 2024 Earnings Conference Call. At this time, all participants are in a listen-only mode. After management’s prepared remarks, we will open the call for questions. I would now like to turn the call over to our CEO, Mark Reynolds. Please go ahead.
# CEO Mark Reynolds: Thank you, and good morning, everyone. I’m pleased to share our results for Q4 2024 and our outlook for the year ahead. Despite ongoing macroeconomic uncertainties, Solid Power posted strong revenue
# growth of 8.2% year-over-year, reaching $420 million for the quarter. This marks our ninth consecutive quarter of revenue expansion, driven by continued demand for high-performance air suspension systems and strategic investments in supply chain resilience.
# Key Highlights:
# - Gross margin expanded to 42.1%, reflecting improved production efficiency and favourable pricing strategies.
# - EBITDA came in at $78.5 million, a 6.5% increase from last year.
# - Net income for the quarter was $24.8 million, or $1.35 per share, up from $1.20 per share in Q3 2024.
# - Cash flow from operations totalled $50 million, reinforcing our strong liquidity
# position.
# 
# Output Indicator:
# 
# Tone Classification: (Optimistic / Cautious / Concerning)
# Key Supporting Evidence: (Direct quotes from the transcript that support the classification)
# Actionable Recommendation: (Brief recommendation for investors or stakeholders based on the CEO’s tone and disclosed information)
# 

# ```text
# 
# Expected output
# 
# Tone Classification: Optimistic
# 
# Key Supporting Evidence:
# 
# “I’m pleased to share our results for Q4 2024 and our outlook for the year ahead.”
# – The CEO opens with a confident and upbeat tone, setting the stage for positive news.
# 
# “Solid Power posted strong revenue growth of 8.2% year-over-year, reaching $420 million for the quarter.”
# – Describes revenue growth as "strong" and highlights consistent performance with nine consecutive quarters of expansion.
# 
# “Gross margin expanded to 42.1%, reflecting improved production efficiency and favourable pricing strategies.”
# – Signals operational strength and successful strategic pricing, both hallmarks of strong business fundamentals.
# 
# “Cash flow from operations totalled $50 million, reinforcing our strong liquidity position.”
# – Reinforces financial stability and cash generation capacity, which are key indicators of corporate health.
# 
# Actionable Recommendation:
# 
# Given the CEO’s optimistic tone, consistent revenue and margin expansion, and solid cash flow, investors may consider increasing exposure to Solid Power, particularly if seeking long-term value in industrial or automotive components. However, macroeconomic headwinds were briefly mentioned, so monitoring broader market conditions and cost inflation trends remains prudent.

# **PRACTICE OPPORTUNITY SOLUTION:**
# - **Modify the `food_recognition_prompt` variable in the code above. Ask a different question, like `"What is the main color of the food in this image?"` or `"Is this food likely sweet or savory?"`. Run the cell again and perform a sanity check on OpenAI's API response.**

# In[22]:


# Let's define our text prompt that will be sent with the image
food_recognition_prompt = """
Context: I'm analyzing a food image for a calorie-tracking application.
Instruction: Determine if this food is sweet or savory and list the colors of the food
Input: [The image I'm about to share]
Output: A brief description of colors and if it's sweet or savory
"""
print(f"{food_recognition_prompt}")


# **PRACTICE OPPORTUNITY SOLUTION:** 
# - **Modify the `structured_nutrition_prompt` to include more fields (e.g. sugar_grams or fiber_grams)**
# - **Try using an image of pizza slice (simple) or a complex dish (like a mixed salad) or a packaged food item. How well does OpenAI's API estimate nutritional value? Do they lower their confidence level?**

# In[29]:


# Let's define a structured prompt to ensure consistent model output
structured_nutrition_prompt = """
# Nutritional Analysis Task

## Context
You are a nutrition expert analyzing food images to provide accurate nutritional information.

## Instructions
Analyze the food item in the image and provide estimated nutritional information based on your knowledge.

## Input
- An image of a food item

## Output
Provide the following estimated nutritional information for a typical serving size or per 100g:
- food_name (string)
- serving_description (string, e.g., '1 slice', '100g', '1 cup')
- calories (float)
- fat_grams (float)
- protein_grams (float)
- sugar_grams (float)
- fiber_grams (float)
- confidence_level (string: 'High', 'Medium', or 'Low')

**IMPORTANT:** Respond ONLY with a single JSON object containing these fields. Do not include any other text, explanations, or apologies. The JSON keys must match exactly: "food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level". If you cannot estimate a value, use `null`.

Example valid JSON response:
{
  "food_name": "Banana",
  "serving_description": "1 medium banana (approx 118g)",
  "calories": 105.0,
  "fat_grams": 0.4,
  "protein_grams": 1.3,
  "confidence_level": "High"
}
"""


# In[31]:


# Let's try the pizza slice!
image_filename = "images/pizza_slice.png"  # <--- CHANGE THIS to your image file name

# Use Pillow's Image.open() to load the image from the file
img = Image.open(image_filename)

# Keep the loaded image object in a variable for later use
image_to_analyze = img

# Let's call OpenAI API with the image and the new structured prompt
openai_nutrition_result = query_openai_vision(client = openai_client,
                                              image = image_to_analyze,
                                              prompt = structured_nutrition_prompt,)

print_markdown(openai_nutrition_result)


# In[33]:


# Let's try the Greek salad!
image_filename = "images/greek_salad.png"  # <--- CHANGE THIS to your image file name

# Use Pillow's Image.open() to load the image from the file
img = Image.open(image_filename)

# Keep the loaded image object in a variable for later use
image_to_analyze = img

# Let's call OpenAI API with the image and the new structured prompt
openai_nutrition_result = query_openai_vision(client = openai_client,
                                              image = image_to_analyze,
                                              prompt = structured_nutrition_prompt,)

print_markdown(openai_nutrition_result)



