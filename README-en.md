--------------------------------------------------------------------------
EVALUATION OF DEEP MODELS FOR PEDESTRIAN ATTRIBUTE RECOGNITION
--------------------------------------------------------------------------

PROJECT OVERVIEW:
This project implements and evaluates Deep Learning models for pedestrian 
attribute recognition in images. Different architectures of convolutional neural 
networks (CNN) and Vision Transformers (ViT) were tested to classify 5 attributes: 
upper clothing color, lower clothing color, gender, bag presence, and hat presence.

Dataset used: PAR2025
Dataset repository: https://github.com/MatheusKozak/Clothing-Detection-Challenge

--------------------------------------------------------------------------
FILE NAMING STRUCTURE
--------------------------------------------------------------------------

NOTEBOOKS (.ipynb):
Notebooks follow the numbering and description pattern:
[NUMBER]-[ARCHITECTURE]-[MODEL]-[TYPE].ipynb

Where:
- NUMBER: development/experimentation order (0, 1, 1.1, 1.2, 2, 2.1, etc.)
- ARCHITECTURE: CNN or ViT (Vision Transformer)
- MODEL: ResNet18, EfficientNetB0, MobileNetV2
- TYPE: "todos" or "individual"
  * "individual": 5 separate models, one for each attribute
  * "todos": single multi-task model that classifies all 5 attributes simultaneously

Example: "1-CNN-ResNet18-individual.ipynb"
Internal structure: Notebooks are organized in markdown sections like:
- Standardizing image sizes
- Deep learning - CNN - ResNet18
- Upper clothing
- Lower clothing
- Gender
- Bag
- Hat

--------------------------------------------------------------------------
CLASSIFIED ATTRIBUTES
--------------------------------------------------------------------------

1. TOP COLOR (Upper clothing color) - 11 classes:
   black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

2. BOTTOM COLOR (Lower clothing color) - 11 classes:
   black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

3. GENDER - 2 classes:
   male (0), female (1)

4. BAG (Bag presence) - 2 classes:
   not present (0), present (1)

5. HAT (Hat presence) - 2 classes:
   not present (0), present (1)

--------------------------------------------------------------------------
TRAINING PARAMETERS AND CONFIGURATIONS
--------------------------------------------------------------------------

Standard configurations used in experiments:

IMAGE_SIZE: 68, 128 or 224 (pixels)
SEED: 42
BATCH_SIZE: 128
LEARNING_RATE: 1e-4 (0.0001)
DEVICE: CUDA (GPU) when available, otherwise CPU

Data split:
- 90% of data for training
- 10% of data for testing

Preprocessing:
- Padding to make images square (PadToSquare)
- Resizing to defined size (BILINEAR interpolation)
- RGB conversion
- Save quality: 95 (JPEG)

Column indices in labels file:
- TOP_COL_IDX: 1
- BOTTOM_COL_IDX: 2
- GENDER_COL_IDX: 3
- BAG_COL_IDX: 4
- HAT_COL_IDX: 5

--------------------------------------------------------------------------
ARTIFACTS DESCRIPTION
--------------------------------------------------------------------------

NOTEBOOKS (.ipynb) - TRAINING SCRIPTS

0-CNN-ResNet18-DataAugmentation.ipynb
  Function: Implementation of data augmentation techniques on the dataset
  Objective: Increase data variability to improve generalization

0.1-Baixando a qualidade das imagens.ipynb
  Function: Controlled reduction of custom image quality (ood_padded_images)
  Objective: Generate lower quality versions of family images for robustness tests
  Output: Directory ./ood_low_quality_images/

1-CNN-ResNet18-individual.ipynb
  Function: Training 5 separate models using ResNet18
  Approach: One independent model for each attribute
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 5 .pth files (one per attribute)

1.1-CNN-ResNet18-todos.ipynb
  Function: Multi-task model training using ResNet18
  Approach: Single model that predicts all 5 attributes simultaneously
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 1 multi-task .pth file

2-CNN-EfficientNetB0-individual.ipynb
  Function: Training 5 separate models using EfficientNetB0
  Approach: One independent model for each attribute
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 5 .pth files (one per attribute)

2.1-CNN-EfficientNetB0-todos.ipynb
  Function: Multi-task model training using EfficientNetB0
  Approach: Single model that predicts all 5 attributes simultaneously
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 1 multi-task .pth file

3-CNN-MobileNetV2-individual.ipynb
  Function: Training 5 separate models using MobileNetV2
  Approach: One independent model for each attribute
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 5 .pth files (one per attribute)

3.1-CNN-MobileNetV2-todos.ipynb
  Function: Multi-task model training using MobileNetV2
  Approach: Single model that predicts all 5 attributes simultaneously
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 1 multi-task .pth file

4-ViT-individual.ipynb
  Function: Training 5 separate models using Vision Transformer
  Approach: One independent model for each attribute
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 5 .pth files (one per attribute)

4.1-ViT-todos.ipynb
  Function: Multi-task model training using Vision Transformer
  Approach: Single model that predicts all 5 attributes simultaneously
  Parameters: IMAGE_SIZE (variable), EPOCHS (variable), LR=1e-4, BATCH_SIZE=128
  Output: 1 multi-task .pth file

5-CNN-ResNet18&EfficientNetB0-todos.ipynb
  Function: Comparison and ensemble analysis
  Approach: Combination of ResNet18 and EfficientNetB0 architectures
  Objective: Evaluate performance gains with multiple models


--------------------------------------------------------------------------
NOTE ABOUT FILES NOT INCLUDED
--------------------------------------------------------------------------

The following files are NOT included in this repository due to size and 
privacy concerns:

- Trained models (.pth): 15-43 MB each
- PDF reports with experimental results
- OOD images (ood_padded_images/ and ood_low_quality_images/)
- gabarito_imagens.txt file


DIRECTORIES AND DATASETS

training_set/
  Description: Original training dataset directory
  Repository: https://github.com/MatheusKozak/Clothing-Detection-Challenge
  Content: Original images and training_set.txt file with labels

validation_set/
  Description: Original validation dataset directory
  Repository: https://github.com/MatheusKozak/Clothing-Detection-Challenge

training_set_resized_[SIZE]/
  Description: Resized training dataset
  Function: Preprocessed images for training
  Generation: Automatically created by training notebooks
  [SIZE]: 68, 128 or 224 according to configuration

teste_set_resized_[SIZE]/
  Description: Resized test dataset
  Function: Preprocessed images for evaluation
  Generation: Automatically created by notebooks (10% of dataset)
