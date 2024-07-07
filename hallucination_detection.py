import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Function to preprocess text
def preprocess_text(text):
    # Convert non-string data to string and handle missing values
    if pd.isnull(text):
        text = '' 
    else:
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'[™+\-\/(){}[\]\|@,;]', '', text)
    text_tokens = text.split()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text_tokens if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

nltk.download('stopwords')
nltk.download('wordnet')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the trained model weights
model_path = 'hallucination_detection_model_bert_base.pth' 
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Load the state dict into the model
model.load_state_dict(model_state_dict, strict=False)

# Ensure the model is in evaluation mode
model.eval()

# Define function to detect hallucination
def detect_hallucination(input_texts, model, tokenizer):
    predictions = []
    for text in input_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.item())
    return predictions

# Load and preprocess the dataset
df = pd.read_csv('Hallucination-Dataset-400-Samples.csv')
df['Context'] = df['Context'].apply(preprocess_text)
df['Question'] = df['Question'].apply(preprocess_text)
df['Answer'] = df['Answer'].apply(preprocess_text)
df['Combined_Text'] = df['Context'] + " [SEP] " + df['Question'] + " [SEP] " + df['Answer']

# Perform hallucination detection
input_texts = df['Combined_Text'].tolist()
predictions = detect_hallucination(input_texts, model, tokenizer)
df['Prediction'] = predictions
df.drop(columns=['Combined_Text'], inplace=True)

# Save the predictions back to the CSV file
df.to_csv('Hallucination-Dataset-400-Samples.csv', index=False)

print("Predictions have been added to the CSV file successfully.")
