DravidianLangTech-2026: Shared Tasks 2026 - Multilevel Political Meme Classification in Tamil and Malayalam (Tamil & Malayalam)

Dataset can be download from : https://sites.google.com/view/dravidianlangtech-2026/ or https://www.codabench.org/competitions/11325/

This project implements a multimodal deep learning model to classify Tamil and Malayalam political memes into two hierarchical levels of categories. The model combines text extracted from meme images using OCR (Tesseract) with visual features from VGG16, and uses IndicBERT for text encoding.
Overview
The system processes Malayalam meme images to:
1. Extract Tamil & Malayalam text using enhanced OCR
2. Generate text embeddings using IndicBERT (AI4Bharat's multilingual BERT model)
3. Extract visual features using VGG16 pre-trained on ImageNet
4. Combine both modalities for hierarchical classification (Level 1 and Level 2 categories)
Dependencies
pandas
numpy
Pillow
ocr_tamil
pytesseract
transformers
keras
tensorflow
scikit-learn
torch
tqdm
opencv-python
matplotlib
seaborn
Installation
pip install pandas numpy pillow pytesseract ocr_tamil transformers keras tensorflow scikit-learn torch tqdm opencv-python matplotlib seaborn

Tesseract OCR with Malayalam Support
1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
   
3. Install Malayalam language pack: tesseract-ocr-mal
   
5. Update the Tesseract path in the code:
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Hugging Face Token
The code uses IndicBERT from Hugging Face. Replace the token with your own:
HF_TOKEN = "your_huggingface_token_here"
Dataset Structure

project/<br>
│<br>
├── train.csv           # Training data with meme_ids and labels (Tamil)<br>
├── test.csv            # Test data with meme_ids (Tamil)<br>
├── Train_images_Tamil/       # Training images folder (Tamil)<br>
│   ├── 1.jpg<br>
│   ├── 2.jpg<br>
│   └── ...<br>
├── Test_images_Tamil/        # Test images folder (Tamil)<br>
│   ├── 101.jpg<br>
│   ├── 102.jpg<br>
│   └── ...<br>
├── train_malayalam.csv           # Training data with meme_ids and labels (Malayalam)<br>
├── test_malayalam.csv            # Test data with meme_ids (Malayalam)<br>
├── Train_images_Malayalam/       # Training images folder (Malayalam)<br>
│   ├── 1.jpg<br>
│   ├── 2.jpg<br>
│   └── ...<br>
├── Test_images_Malalayam/        # Test images folder (Malayalam)<br>
│   ├── 101.jpg<br>
│   ├── 102.jpg<br>
│   └── ...<br>

Adjust OCR Parameters
Modify psm_configs and lang_configs in extract_malayalam_text_enhanced():
psm_configs = [
    ('--psm 3', 'Default'),
    ('--psm 6', 'Single block'),
    # Add more modes
]

Notes:
- The OCR function uses multiple preprocessing techniques to handle various image qualities
- Text encoding uses mean pooling of IndicBERT token embeddings
- The model is trained for 20 epochs by default (adjustable)
- All CSV outputs use UTF-8 encoding with BOM for Excel compatibility

Acknowledgments
- DravidianLangTech-2026
- AI4Bharat for IndicBERT
- Tesseract OCR team
- ocr_tamil team
- VGG16 model from Keras applications

Contact
For questions or suggestions, please open an issue on GitHub.

Note: Ensure you have the required permissions and computational resources before running the full pipeline. The process can be memory-intensive when processing large datasets.
