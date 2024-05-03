import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import pandas as pd

def t_tweets(file, encoding):
    df = pd.read_csv(file, encoding = encoding)
    
    tweets_by_target = {}
    stances_by_target = {}
    all_tweets = []
    all_stances = []
    
    unique_targets = df["Target"].unique()
    for target in unique_targets:
        target_df = df[df["Target"] == target]
        
        tweets_by_target[target] = target_df["Tweet"].tolist()
        stances_by_target[target] = target_df["Stance"].tolist()
        
        all_tweets.extend(target_df["Tweet"].tolist())
        all_stances.extend(target_df["Stance"].tolist())
        
    return tweets_by_target, stances_by_target, all_tweets, all_stances


# Define the neural network model
class StanceClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(StanceClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Number of classes (number of different stances)
num_classes = 3  # Assuming you have 3 classes (positive, negative, neutral)

# Initialize the classifier model
classifier_model = StanceClassifier(bert_model, num_classes)

# Define optimizer and loss function
optimizer = optim.Adam(classifier_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Convert data to BERT input format and create DataLoader
# Assuming you have train_features, train_labels, test_features, test_labels
# train_features and test_features should be BERT input format (input_ids, attention_mask)
# train_labels and test_labels should be tensors containing the labels
tweets_train, stances_train, all_tweets_train, all_stances_train = t_tweets('IBM_train.csv', "ANSI")

tweets_test, stances_test, all_tweets_test, all_stances_test = t_tweets('IBM_test.csv', "ANSI")


# Define a label mapping
label_map = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}

# Encode labels
train_labels = torch.tensor([label_map[label] for label in all_stances_train])
test_labels = torch.tensor([label_map[label] for label in all_stances_test])

tokenized_train = tokenizer(all_tweets_train, padding=True, truncation=True, return_tensors='pt')
tokenized_test = tokenizer(all_tweets_test, padding=True, truncation=True, return_tensors='pt')

train_dataset = TensorDataset(tokenized_train['input_ids'], tokenized_train['attention_mask'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define test dataset and dataloader
test_dataset = TensorDataset(tokenized_test['input_ids'], tokenized_test['attention_mask'], test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, true_labels, predictions



# Training and evaluation loop
for epoch in range(num_epochs):
    # Training loop
    classifier_model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = classifier_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation loop on test data
    test_loss, true_labels, predictions = evaluate_model(classifier_model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}")

    # Calculate accuracy
    accuracy = (torch.tensor(true_labels) == torch.tensor(predictions)).float().mean().item()
    print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")
