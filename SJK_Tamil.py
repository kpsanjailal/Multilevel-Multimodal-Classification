import pandas as pd
import numpy as np
import os
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModel
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Concatenate, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm
from ocr_tamil.ocr import OCR

#  Hugging Face User Access Token access token 
HF_TOKEN = "Please Replace with Hugging Face User Access Token" 


try:
    model_name = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_bert =AutoModel.from_pretrained(model_name)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")


# Load VGG16 for image encoding pre-trained on ImageNet, without top layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

Extract text using the specialised ocr_tamil tool
def extract_text_using_ocr_tamil(image_path):
    ocr = OCR(detect=True)
    text_list = ocr.predict(image_path)
    for item in text_list:
        return " ".join(item).strip()
    

# Function to encode text using IndicBERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Use mean pooling of the last hidden state
    text_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    return text_embedding.flatten()

# Function to encode image using VGG16
def encode_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = vgg_model.predict(img_array)
    return features.flatten()

# Load training data
train_df = pd.read_csv('train.csv')  # Assuming the file is named 'train.csv'
train_image_folder = 'Train_Images_Tamil'

# Prepare training data
X_text_train = []
X_img_train = []
y1_train = []
y2_train = []

#with progress bar
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
    img_path = os.path.join(train_image_folder, row['Image_name'])
    if os.path.exists(img_path):
        text = extract_text_using_ocr_tamil(img_path)
        text_enc = encode_text(text)
        img_enc = encode_image(img_path)
        X_text_train.append(text_enc)
        X_img_train.append(img_enc)
        y1_train.append(row['Level1'])
        y2_train.append(row['Level2'])

X_text_train = np.array(X_text_train)
X_img_train = np.array(X_img_train)

# Encode labels for Level1 and Level2
le1 = LabelEncoder()
y1_enc = le1.fit_transform(y1_train)
y1_cat = to_categorical(y1_enc)

le2 = LabelEncoder()
y2_enc = le2.fit_transform(y2_train)
y2_cat = to_categorical(y2_enc)

# Build the multimodal model
# IndicBERT embedding size: 768
text_input_dim = X_text_train.shape[1]
# Image (VGG16):25088
img_input_dim = X_img_train.shape[1]

text_input = Input(shape=(text_input_dim,), name='text_input')
img_input = Input(shape=(img_input_dim,), name='img_input')

# Concatenating text and image features
concat = Concatenate()([text_input, img_input])

# Dense layers for classification
dense1 = Dense(256, activation='relu')(concat)
dense2 = Dense(128, activation='relu')(dense1)

# Outputs for Level1 and Level2
out1 = Dense(len(le1.classes_), activation='softmax', name='level1')(dense2)
out2 = Dense(len(le2.classes_), activation='softmax', name='level2')(dense2)

#dense1 = Dense(64, activation='relu')(concat)
#dense2 = Dense(32, activation='relu')(dense1)
#dense3 = Dense(16, activation='relu')(dense2)
#out1 = Dense(len(le1.classes_), activation='softmax', name='level1')(dense3)
#out2 = Dense(len(le2.classes_), activation='softmax', name='level2')(dense3)

model = Model(inputs=[text_input, img_input], outputs=[out1, out2])
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss={'level1': 'categorical_crossentropy', 'level2': 'categorical_crossentropy'},
              metrics={'level1': 'accuracy', 'level2': 'accuracy'})

# Train for 20 epochs with batch size 16
history = model.fit([X_text_train, X_img_train], {'level1': y1_cat, 'level2': y2_cat}, epochs=20, batch_size=16)

# Load test data
test_df = pd.read_csv('test.csv')
test_image_folder = 'Test_images_Tamil'
if 'extracted_text' not in test_df.columns:
    test_df['extracted_text'] = None
# Prepare test data
X_text_test = []
X_img_test = []
# process test data with progress bar
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing training data"):
    img_path = os.path.join(test_image_folder, row['Image_name'])
    if os.path.exists(img_path):
        text = extract_text_using_ocr_tamil(img_path)
        test_df.at[idx, 'extracted_text'] = text
        text_enc = encode_text(text)
        img_enc = encode_image(img_path)
        X_text_test.append(text_enc)
        X_img_test.append(img_enc)

X_text_test = np.array(X_text_test)
X_img_test = np.array(X_img_test)

# Predict on test data
predictions = model.predict([X_text_test, X_img_test])
pred1_indices = np.argmax(predictions[0], axis=1)
pred2_indices = np.argmax(predictions[1], axis=1)

pred1 = le1.inverse_transform(pred1_indices)
pred2 = le2.inverse_transform(pred2_indices)

# Update test_df with predictions
test_df['Level1'] = pred1
test_df['Level2'] = pred2

# Print model performance after training
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

print("\nFINAL EPOCH PERFORMANCE:")
print("-" * 40)

# Get the last epoch metrics
last_epoch = len(history.history['level1_accuracy']) - 1

print(f"Level1 Training Accuracy: {history.history['level1_accuracy'][-1]:.4f}")
print(f"Level1 Training Loss: {history.history['level1_loss'][-1]:.4f}")

print(f"\nLevel2 Training Accuracy: {history.history['level2_accuracy'][-1]:.4f}")
print(f"Level2 Training Loss: {history.history['level2_loss'][-1]:.4f}")

print("\n" + "="*60)
print("EPOCH-WISE PERFORMANCE")
print("="*60)

# Print epoch-wise performance
print("\nEpoch\tLevel1 Acc\tLevel1 Loss\tLevel2 Acc\tLevel2 Loss")
print("-" * 60)

for epoch in range(len(history.history['level1_accuracy'])):
    print(f"{epoch+1}\t"
          f"{history.history['level1_accuracy'][epoch]:.4f}\t\t"
          f"{history.history['level1_loss'][epoch]:.4f}\t\t"
          f"{history.history['level2_accuracy'][epoch]:.4f}\t\t"
          f"{history.history['level2_loss'][epoch]:.4f}")
        
# Save predictions to CSV
output_file = 'SJK_Tamil.csv'
print(f"\nSaving predictions to {output_file}...")
test_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print("Predictions saved")