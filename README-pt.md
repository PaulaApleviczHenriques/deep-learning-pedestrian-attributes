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
Repositório da equipe: https://github.com/MatheusKozak/Clothing-Detection-Challenge

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

RELATÓRIOS PDF:
Os arquivos PDF seguem o padrão de nomenclatura detalhado:
[TAMANHO]-[ÉPOCAS]-[ARQUITETURA]-[MODELO]-[TIPO].pdf

Onde:
- TAMANHO: dimensão das imagens (68x68, 128x128, 224x224)
- ÉPOCAS: número de épocas de treinamento (5, 10, 20, 40)
- ARQUITETURA: CNN ou ViT (Vision Transformer)
- MODELO: ResNet18, EfficientNetB0, MobileNetV2
- TIPO: "todos" ou "individual"

Exemplo: "128x128-5-CNN-ResNet18-individual.pdf"
Significa: imagens 128x128, 5 épocas, CNN com ResNet18, modelos individuais

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

0-CNN-ResNet18-DataAugmentation.ipynb
  Função: Implementação de técnicas de data augmentation no dataset
  Objetivo: Aumentar a variabilidade dos dados para melhorar generalização

0.1-Baixando a qualidade das imagens.ipynb
  Função: Redução controlada da qualidade das imagens customizadas (ood_padded_images)
  Objetivo: Gerar versões de menor qualidade das imagens de familiares para testes de robustez
  Saída: Diretório ./ood_low_quality_images/

1-CNN-ResNet18-individual.ipynb
  Função: Treinamento de 5 modelos separados usando ResNet18
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

1.1-CNN-ResNet18-todos.ipynb
  Função: Treinamento de modelo multi-tarefa usando ResNet18
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

1.2-CNN-ResNet18-todos.ipynb
  Função: Variação experimental do treinamento multi-tarefa ResNet18
  Abordagem: Testes com ajustes de hiperparâmetros

2-CNN-EfficientNetB0-individual.ipynb
  Função: Treinamento de 5 modelos separados usando EfficientNetB0
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

2.1-CNN-EfficientNetB0-todos.ipynb
  Função: Treinamento de modelo multi-tarefa usando EfficientNetB0
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

2.2-CNN-EfficientNetB0-todos.ipynb
  Função: Variação experimental do treinamento multi-tarefa EfficientNetB0
  Abordagem: Testes com ajustes de hiperparâmetros

3-CNN-MobileNetV2-individual.ipynb
  Função: Treinamento de 5 modelos separados usando MobileNetV2
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

3.1-CNN-MobileNetV2-todos.ipynb
  Função: Treinamento de modelo multi-tarefa usando MobileNetV2
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

4-ViT-individual.ipynb
  Função: Treinamento de 5 modelos separados usando Vision Transformer
  Abordagem: Um modelo independente para cada atributo
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 5 arquivos .pth (um por atributo)

4.1-ViT-todos.ipynb
  Função: Treinamento de modelo multi-tarefa usando Vision Transformer
  Abordagem: Modelo único que prediz os 5 atributos simultaneamente
  Parâmetros: IMAGE_SIZE (variável), EPOCHS (variável), LR=1e-4, BATCH_SIZE=128
  Saída: 1 arquivo .pth multi-tarefa

5-CNN-ResNet18&EfficientNetB0-todos.ipynb
  Função: Comparação e análise de ensemble
  Abordagem: Combinação das arquiteturas ResNet18 e EfficientNetB0
  Objetivo: Avaliar ganhos de desempenho com múltiplos modelos


MODELOS TREINADOS (.pth)

best_efficientnet_multitask.pth
  Descrição: Melhor modelo EfficientNetB0 multi-tarefa
  Tamanho: 15,7 MB
  Função: Pesos do modelo treinado para classificação dos 5 atributos
  Uso: Carregar com torch.load() para inferência ou fine-tuning

best_multitask_resnet18_full.pth
  Descrição: Melhor modelo ResNet18 multi-tarefa completo
  Tamanho: 42,8 MB
  Função: Pesos do modelo treinado completo
  Uso: Carregar com torch.load() para inferência

best_multitask_resnet18_state.pth
  Descrição: State dict do melhor modelo ResNet18 multi-tarefa
  Tamanho: 42,8 MB
  Função: Dicionário de estados do modelo (apenas pesos)
  Uso: Carregar com model.load_state_dict(torch.load())


DIRETÓRIOS E ARQUIVOS AUXILIARES

./ood_padded_images/
  Descrição: Pasta contendo imagens de familiares processadas com padding
  Função: Dataset customizado com imagens cortadas e normalizadas
  Conteúdo: Imagens em formato .jpg/.png de teste em ambiente real

./ood_low_quality_images/
  Descrição: Pasta com versões de qualidade reduzida das imagens customizadas
  Função: Dataset para teste de robustez dos modelos com imagens de menor qualidade
  Origem: Geradas a partir das imagens em ./ood_padded_images/ usando o notebook "0.1-Baixando a qualidade das imagens.ipynb"

gabarito_imagens.txt
  Descrição: Arquivo de gabarito com labels manuais
  Tamanho: 3 KB
  Função: Ground truth para as imagens customizadas (ood_padded_images)
  Formato: Texto com labels anotados manualmente pela equipe
  Uso: Validação e teste dos modelos em imagens reais

2025-2-BCC-PT2-TemplatePoster/
  Descrição: Pasta com template do pôster científico
  Tamanho: 6 MB
  Função: Material para apresentação do projeto

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
