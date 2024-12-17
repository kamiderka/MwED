from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset

# Parametry
MODEL_NAME = 'bert-base-uncased'
SEQ_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Przygotowujemy dane
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def preprocess_data(batch):
    encoded = tokenizer(
        batch['text'],
        max_length=SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    batch['input_ids'] = encoded['input_ids'].squeeze(0)
    batch['attention_mask'] = encoded['attention_mask'].squeeze(0)
    return batch


# Ładujemy dane
dataset = load_dataset(
    'csv', data_files={'train': 'dane/twitter_sentiment_analysis_train.csv', 'test': 'dane/twitter_sentiment_analysis_test.csv'})
dataset = dataset.map(preprocess_data, batched=True)
dataset.set_format(type='torch', columns=[
                   'input_ids', 'attention_mask', 'feeling'])

train_dataset = dataset['train']
test_dataset = dataset['test']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Tworzymy model


class TwitterSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['feeling'])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['feeling'])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['feeling']).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


# Trenujemy model
model = TwitterSentimentClassifier()

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices=1,
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, test_loader)

model.model.save_pretrained("twitter_sentiment_model")
tokenizer.save_pretrained("twitter_sentiment_model")


model_path = "twitter_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()


def classify_text(text):
    inputs = tokenizer(
        text,
        max_length=SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"]).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[predicted_class]


text = "i hate you"
print(f"Text: {text}")
print(f"Predicted Sentiment: {classify_text(text)}")
