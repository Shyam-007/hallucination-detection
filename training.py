import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('wordnet')


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

# Load the dataset (adjust the path as necessary)
df = pd.read_csv('Hallucination-Dataset-400-Samples.csv')


# Preprocess the dataset
df['Context'] = df['Context'].apply(preprocess_text)
df['Question'] = df['Question'].apply(preprocess_text)
df['Answer'] = df['Answer'].apply(preprocess_text)
df['Combined_Text'] = df['Context'] + "[SEP]" + df['Question'] + "[SEP] " + df['Answer']

# Split the data
X = df['Combined_Text']
y = df['Hallucination']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and prepare the data for BERT
def encode_data(tokenizer, texts, max_length=512):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

# Encoding data
train_inputs, train_masks = encode_data(tokenizer, X_train.tolist())
test_inputs, test_masks = encode_data(tokenizer, X_test.tolist())

# Convert to torch tensors
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)
train_inputs = torch.tensor(train_inputs)
train_masks = torch.tensor(train_masks)
test_inputs = torch.tensor(test_inputs)
test_masks = torch.tensor(test_masks)


# Create the DataLoader
batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop with validation accuracy
for epoch_i in range(0, epochs):
    print(f'Epoch {epoch_i + 1}/{epochs}')
    print('Training')

    total_loss = 0
    model.train()
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        model.zero_grad()
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss: {avg_train_loss}')
    
    # Validation phase
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

print("\nTraining complete.")


# Evaluation on the test set
print("Starting evaluation on the test set")

model.eval()
predictions , true_labels = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

print('Accuracy:', accuracy_score(flat_true_labels, flat_predictions))
print('F1 Score:', f1_score(flat_true_labels, flat_predictions))
print('AUC-ROC:', roc_auc_score(flat_true_labels, flat_predictions))
'''
# Save the model
model.save_pretrained('./hallucination_model')
tokenizer.save_pretrained('./hallucination_model')'''
# Save the model using torch.save for the entire model
torch.save(model.state_dict(), 'hallucination_detection_model_bert_base.pth')
print("Model saved successfully.")

