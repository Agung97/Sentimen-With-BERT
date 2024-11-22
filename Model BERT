#import library yang akan digunakan untuk model BERT
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Tokenisasi teks menggunakan tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Pisahkan data menjadi set pelatihan, validasi, dan uji
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(df_shuffled['processed_text'],
                                                                    df_shuffled['label_num'],
                                                                    test_size=0.2,
                                                                    random_state=42)
# Tokenisasi teks dan konversi menjadi tensors
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Buat dataset PyTorch
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels.reset_index(drop=True)))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels.reset_index(drop=True)))

# Buat DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Inisialisasi model BERT untuk klasifikasi
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df_shuffled['label_num']))

# Tentukan optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Tentukan device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inisialisasi list untuk menyimpan nilai loss dan akurasi
train_losses = []
val_accuracies = []

# Pelatihan model
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Kode pelatihan yang ada
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Simpan nilai loss pada setiap epoch

    # Evaluasi model pada set validasi
    model.eval()
    val_preds = []
    val_true = []
    for batch in val_loader:
        # Kode evaluasi yang ada
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to('cpu')

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        preds = torch.argmax(logits, dim=1).tolist()
        val_preds.extend(preds)
        val_true.extend(labels.tolist())

    val_accuracy = accuracy_score(val_true, val_preds)
    val_accuracies.append(val_accuracy)  # Simpan nilai akurasi pada setiap epoch

    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Gambar grafik perubahan loss dan akurasi selama pelatihan
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Validation Accuracy')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
# Evaluasi model pada set uji
test_texts = df_shuffled['processed_text']
test_labels = df_shuffled['label_num']
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
test_preds = []
test_true = []
for batch in test_loader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to('cpu')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    preds = torch.argmax(logits, dim=1).tolist()
    test_preds.extend(preds)
    test_true.extend(labels.tolist())

# Define label dictionary
label_dict = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}

test_accuracy = accuracy_score(test_true, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Tampilkan classification report
print(classification_report(test_true, test_preds, target_names=label_dict.keys()))
