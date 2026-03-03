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
import cv2
import matplotlib.pyplot as plt
#  Hugging Face User Access Token access token 
HF_TOKEN = "Please Replace with Hugging Face User Access Token" 

# First Tesseract need to be installed with Malayalam Support & set the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    model_name = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_bert =AutoModel.from_pretrained(model_name)
    print("IndicBERT loaded successfully!")   
except Exception as e:
    print(f"Error: {e}")


# Load VGG16 for image encoding pre-trained on ImageNet, without top layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Enhanced text extraction Function to extract text from Malayalam meme images - IMPROVED VERSION
def extract_malayalam_text_enhanced(image_path):
    try:
        from PIL import Image, ImageEnhance
        import pytesseract
        
        img = Image.open(image_path)
        
        # Create multiple preprocessing variants
        variants = []
        
        # Original image
        variants.append(('original', img))
        
        # Grayscale variants
        img_gray = img.convert('L')
        variants.append(('gray', img_gray))
        
        # Contrast enhanced
        enhancer = ImageEnhance.Contrast(img_gray)
        img_high_contrast = enhancer.enhance(2.0)
        variants.append(('high_contrast', img_high_contrast))
        
        # Brightness enhanced
        enhancer = ImageEnhance.Brightness(img_gray)
        img_bright = enhancer.enhance(1.5)
        variants.append(('bright', img_bright))
        
        # Adaptive thresholding for better results
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        thresh_img = Image.fromarray(thresh)
        variants.append(('adaptive_thresh', thresh_img))
        
        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        denoised_img = Image.fromarray(denoised)
        variants.append(('denoised', denoised_img))
        
        # Configs for better character recognition
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened_img = Image.fromarray(sharpened)
        variants.append(('sharpened', sharpened_img))
        
        # Try different page segmentation modes for multi-line text
        psm_configs = [
            ('--psm 3', 'Fully automatic page segmentation (default)'),
            ('--psm 6', 'Assume a single uniform block of text'),
            ('--psm 11', 'Sparse text - find as much text as possible'),
            ('--psm 12', 'Sparse text with OSD'),
            ('--psm 4', 'Assume a single column of text of variable sizes'),
            ('--psm 5', 'Assume a single uniform block of vertically aligned text'),
            ('--psm 7', 'Treat the image as a single text line'),
            ('--psm 13', 'Raw line - treat the image as a single text line'),
            ('--psm 1', 'Automatic page segmentation with OSD')
        ]
        
        # Set Language configurations
        lang_configs = ['mal', 'mal+eng', 'eng']
        
        # Set 'Malayalam' script code in Tesseract
        script_configs = ['script/Malayalam', 'mal', 'mal+eng']
        
        best_text = ""
        best_score = 0
        best_config = ""
        
        for variant_name, variant_img in variants:
            # Try language configs
            for lang in lang_configs:
                for psm, psm_desc in psm_configs:
                    try:
                        # Combine configs with Malayalam-specific optimizations
                        config = f'{psm} -c preserve_interword_spaces=1 -c tessedit_char_whitelist='
                                   
                        # Extract text
                        ocr_text = pytesseract.image_to_string(
                            variant_img, 
                            lang=lang, 
                            config=config
                        )
                        
                        if ocr_text.strip():
                            # Clean text
                            ocr_text = ocr_text.strip()
                            
                            # Score based on text quality for Malayalam
                            text_length = len(ocr_text)
                            line_count = ocr_text.count('\n') + 1
                            
                            # Malayalam Unicode range: U+0D00 to U+0D7F
                            malayalam_chars = sum(1 for c in ocr_text if '\u0D00' <= c <= '\u0D7F')
                            
                            # Check for Malayalam-specific patterns
                            malayalam_patterns = 0
                            malayalam_patterns += ocr_text.count('?')
                            malayalam_patterns += ocr_text.count('?')
                            malayalam_patterns += ocr_text.count('?')
                            
                            # Scoring scheme to select best text            
                            unique_chars = len(set(ocr_text.replace('\n', '').replace(' ', '')))
                            
                            score = (
                                text_length * 0.3 +
                                line_count * 40 + 
                                malayalam_chars * 5 + 
                                malayalam_patterns * 10 +
                                unique_chars * 2
                            )
                            
                            # Additional points if text ends with a punctuation mark
                            if ocr_text[-1] in ['.', '!', '?', '?', '?', '?']:
                                score += 20
                            
                            # Additional points for more Malayalam than non-Malayalam
                            if malayalam_chars > (text_length / 2):
                                score += 30
                            
                            if score > best_score:
                                best_score = score
                                best_text = ocr_text
                                best_config = f"{variant_name}, lang={lang}, {psm_desc}"
                                
                    except Exception as e:
                        continue
            
            # Script-based detection if not getting good results using language-based
            for script in script_configs:
                for psm, psm_desc in psm_configs:
                    try:
                        config = f'{psm} -c preserve_interword_spaces=1'
                        ocr_text = pytesseract.image_to_string(
                            variant_img, 
                            lang=script, 
                            config=config
                        )
                        
                        if ocr_text.strip() and len(ocr_text.strip()) > best_score * 0.8:
                            # Similar scoring logic
                            malayalam_chars = sum(1 for c in ocr_text if '\u0D00' <= c <= '\u0D7F')
                            if malayalam_chars > len(ocr_text) * 0.3:  # At least 30% Malayalam
                                best_text = ocr_text.strip()
                                best_config = f"{variant_name}, lang={script}, {psm_desc}"
                                break
                    except:
                        continue
        
        # If we got text, clean it up
        if best_text:
            # Remove additional whitespace if any
            lines = [line.strip() for line in best_text.split('\n') if line.strip()]
            best_text = '\n'.join(lines)
            
            # Remove common OCR artifacts in Malayalam text
            import re
            best_text = re.sub(r'[|\\/~^`@#$%^&*()_+=\[\]{};:"<>]', '', best_text)
            
            # Debug output for first few images
            if len(best_text) > 0:
                malayalam_count = sum(1 for c in best_text if '\u0D00' <= c <= '\u0D7F')
                print(f"  Extracted {len(lines)} lines with {malayalam_count} Malayalam chars using config: {best_config}")
        
        return best_text
    
    except ImportError as e:
        print(f"Required library not found: {e}")
        return ""
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""
    
#Function encode text using IndicBERT
def encode_text(text):
    try:
        # Ensure text is in Unicode format
        if isinstance(text, bytes):
            text = text.decode('utf-8')
         # Clean and normalize text
        text = clean_text(text)
        
        # Tokenize with IndicBERT
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        
        # Move to device if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_bert.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model_bert(**inputs)
            
        # Mean pooling with attention mask
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Create mask for mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by sum of mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        text_embedding = (sum_embeddings / sum_mask).cpu().numpy()
        
        return text_embedding.flatten()
        
    except Exception as e:
        print(f"Error encoding text: {e}")
        # Return zero vector as fallback
        return np.zeros(model_bert.config.hidden_size)

#Function to clean and normalize Malayalam text for encoding
def clean_text(text):

    if not text:
        return ""
    
    import re
    
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)  # Uncomment to keep only Malayalam
    text = ' '.join(text.split())
    
    return text

# Function to encode image using VGG16
def encode_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    # Normalize pixel value by dividing with 255.0
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = vgg_model.predict(img_array)
    return features.flatten()

# Load training data
train_df = pd.read_csv('train_malayalam.csv')
train_image_folder = 'Train_images_Malayalam'

# Prepare training data
X_text_train = []
X_img_train = []
y1_train = []
y2_train = []

#with progress bar
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
#for idx, row in train_df.iterrows():
    img_path = os.path.join(train_image_folder, f"{row['meme_id']:.0f}.jpg")
    if os.path.exists(img_path):
        text = extract_malayalam_text_enhanced(img_path)
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
test_df = pd.read_csv('test_malayalam.csv')
#Set Test Image Folder
test_image_folder = 'Test_images_Malayalam'
if 'extracted_text' not in test_df.columns:
    test_df['extracted_text'] = None
# Prepare test data
X_text_test = []
X_img_test = []
# process test data with progress bar
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing training data"):
    img_path = os.path.join(test_image_folder, f"{row['meme_id']:.0f}.jpg")
    if os.path.exists(img_path):
        text = extract_malayalam_text_enhanced(img_path)
        test_df.at[idx, 'extracted_text'] = text;
        text_enc = encode_text(text)
        img_enc = encode_image(img_path)
        X_text_test.append(text_enc)
        X_img_test.append(img_enc)

X_text_test = np.array(X_text_test)
X_img_test = np.array(X_img_test)

# Generate Predictions for test data
predictions = model.predict([X_text_test, X_img_test])
pred1_indices = np.argmax(predictions[0], axis=1)
pred2_indices = np.argmax(predictions[1], axis=1)

pred1 = le1.inverse_transform(pred1_indices)
pred2 = le2.inverse_transform(pred2_indices)

# Update predictions
test_df['Level1'] = pred1
test_df['Level2'] = pred2

# Print performance details
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
        
# Save predictions in csv format
test_df.to_csv('SJK_Malayalam.csv', index=False, encoding='utf-8-sig')
print("Predictions saved to SJK_Malayalam.csv")

# =======PLOTS =======
plt.figure(figsize=(12, 5))

# (1) Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['level1_accuracy'], label='Level-1 Train Acc')
plt.plot(history.history['level2_accuracy'], label='Level-2 Train Acc')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# (2) Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['level1_loss'], label='Level-1 Train Loss')
plt.plot(history.history['level2_loss'], label='Level-2 Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()