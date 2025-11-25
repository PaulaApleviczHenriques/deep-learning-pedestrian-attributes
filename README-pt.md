--------------------------------------------------------------------------
AVALIAÇÃO DE MODELOS PROFUNDOS PARA RECONHECIMENTO DE ATRIBUTOS DE PEDESTRES
--------------------------------------------------------------------------

DESCRIÇÃO GERAL DO PROJETO:
Este projeto implementa e avalia modelos de Deep Learning para reconhecimento de 
atributos de pedestres em imagens. Foram testadas diferentes arquiteturas de 
redes neurais convolucionais (CNN) e Vision Transformers (ViT) para classificar 
5 atributos: cor da roupa superior, cor da roupa inferior, gênero, presença de 
bolsa e presença de chapéu.

Dataset utilizado: PAR2025
Repositório do dataset: https://github.com/MatheusKozak/Clothing-Detection-Challenge

--------------------------------------------------------------------------
ESTRUTURA DE NOMENCLATURA DOS ARQUIVOS
--------------------------------------------------------------------------

NOTEBOOKS (.ipynb):
Os notebooks seguem o padrão de numeração e descrição:
[NÚMERO]-[ARQUITETURA]-[MODELO]-[TIPO].ipynb

Onde:
- NÚMERO: ordem de desenvolvimento/experimentação (0, 1, 1.1, 1.2, 2, 2.1, etc.)
- ARQUITETURA: CNN ou ViT (Vision Transformer)
- MODELO: ResNet18, EfficientNetB0, MobileNetV2
- TIPO: "todos" ou "individual"
  * "individual": 5 modelos separados, um para cada atributo
  * "todos": modelo único multi-tarefa que classifica os 5 atributos simultaneamente

Exemplo: "1-CNN-ResNet18-individual.ipynb"
Estrutura interna: Os notebooks são organizados em seções markdown como:
- Padronizando os tamanhos da imagem
- Deep learning - CNN - ResNet18
- Parte superior da roupa
- Parte inferior da roupa
- Gênero
- Bolsa
- Chapéu

--------------------------------------------------------------------------
ATRIBUTOS CLASSIFICADOS
--------------------------------------------------------------------------

1. TOP COLOR (Cor da roupa superior) - 11 classes:
   black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

2. BOTTOM COLOR (Cor da roupa inferior) - 11 classes:
   black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

3. GENDER (Gênero) - 2 classes:
   male (0), female (1)

4. BAG (Presença de bolsa) - 2 classes:
   not present (0), present (1)

5. HAT (Presença de chapéu) - 2 classes:
   not present (0), present (1)

--------------------------------------------------------------------------
PARÂMETROS E CONFIGURAÇÕES DE TREINAMENTO
--------------------------------------------------------------------------

Configurações padrão utilizadas nos experimentos:

IMAGE_SIZE: 68, 128 ou 224 (pixels)
SEED: 42
BATCH_SIZE: 128
LEARNING_RATE: 1e-4 (0.0001)
DEVICE: CUDA (GPU) quando disponível, caso contrário CPU

Split dos dados:
- 90% dos dados para treinamento
- 10% dos dados para teste

Pré-processamento:
- Padding para tornar imagens quadradas (PadToSquare)
- Redimensionamento para o tamanho definido (BILINEAR interpolation)
- Conversão para RGB
- Qualidade de salvamento: 95 (JPEG)

Índices das colunas no arquivo de labels:
- TOP_COL_IDX: 1
- BOTTOM_COL_IDX: 2
- GENDER_COL_IDX: 3
- BAG_COL_IDX: 4
- HAT_COL_IDX: 5

--------------------------------------------------------------------------
DESCRIÇÃO DOS ARTEFATOS
--------------------------------------------------------------------------

NOTEBOOKS (.ipynb) - SCRIPTS DE TREINAMENTO

1-CNN-ResNet18-individual.py
  Função: Treinamento de 5 modelos separados usando ResNet18
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

1.1-CNN-ResNet18-todos.py
  Função: Treinamento de modelo multi-tarefa usando ResNet18
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

2-CNN-EfficientNetB0-individual.py
  Função: Treinamento de 5 modelos separados usando EfficientNetB0
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

2.1-CNN-EfficientNetB0-todos.py
  Função: Treinamento de modelo multi-tarefa usando EfficientNetB0
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

3-CNN-MobileNetV2-individual.py
  Função: Treinamento de 5 modelos separados usando MobileNetV2
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

3.1-CNN-MobileNetV2-todos.py
  Função: Treinamento de modelo multi-tarefa usando MobileNetV2
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

4-ViT-individual.py
  Função: Treinamento de 5 modelos separados usando Vision Transformer
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

4.1-ViT-todos.py
  Função: Treinamento de modelo multi-tarefa usando Vision Transformer
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

5-CNN-ResNet18&EfficientNetB0-todos.py
  Função: Comparação e análise de ensemble
  Abordagem: Combinação das arquiteturas ResNet18 e EfficientNetB0
  Objetivo: Avaliar ganhos de desempenho com múltiplos modelos


--------------------------------------------------------------------------
NOTA SOBRE ARQUIVOS NÃO INCLUÍDOS
--------------------------------------------------------------------------

Os seguintes arquivos NÃO estão incluídos neste repositório por questões de 
tamanho e privacidade:

- Modelos treinados (.pth): 15-43 MB cada
- Relatórios PDF com resultados experimentais
- Imagens OOD (ood_padded_images/ e ood_low_quality_images/)
- Arquivo gabarito_imagens.txt


DIRETÓRIOS E DATASETS

training_set/
  Descrição: Diretório original do dataset de treinamento
  Repositório: https://github.com/MatheusKozak/Clothing-Detection-Challenge
  Conteúdo: Imagens originais e arquivo training_set.txt com labels

validation_set/
  Descrição: Diretório original do dataset de validação
  Repositório: https://github.com/MatheusKozak/Clothing-Detection-Challenge

training_set_resized_[SIZE]/
  Descrição: Dataset de treino redimensionado
  Função: Imagens pré-processadas para treinamento
  Geração: Criado automaticamente pelos notebooks de treinamento
  [SIZE]: 68, 128 ou 224 conforme configuração

teste_set_resized_[SIZE]/
  Descrição: Dataset de teste redimensionado
  Função: Imagens pré-processadas para avaliação
  Geração: Criado automaticamente pelos notebooks (10% do dataset)
