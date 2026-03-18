# 📌 Projeto: Detecção de Objetos com OpenCV

## 👨‍💻 Autores

* Juan Pinheiro de França — RM552202
* Kaiky Alvaro de Miranda — RM98118

---

## 📖 Descrição

Este projeto implementa um pipeline de processamento de imagens utilizando a biblioteca OpenCV para realizar **segmentação, limpeza e detecção de objetos** em uma imagem.

O fluxo inclui técnicas clássicas de visão computacional, como:

* Binarização automática (Otsu)
* Operações morfológicas
* Detecção de contornos
* Geração de bounding boxes

---

## ⚙️ Tecnologias Utilizadas

* Python
* OpenCV (`cv2`)
* Matplotlib
* OS

---

## 🧠 Etapas do Pipeline

### 1. Leitura da Imagem

A imagem é carregada em escala de cinza:

```python
image = cv2.imread(path_image, 0)
```

---

### 2. Segmentação com Otsu

Aplicação do método de limiarização automática de Otsu:

```python
cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

✔ Objetivo: separar fundo e objetos automaticamente.

---

### 3. Operações Morfológicas

Aplicação de fechamento (closing) para remover ruídos:

```python
cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
```

✔ Objetivo: melhorar a segmentação, unindo regiões próximas.

---

### 4. Detecção de Contornos

Identificação dos contornos externos na imagem binária:

```python
cv2.findContours(...)
```

✔ Retorna a quantidade de objetos detectados.

---

### 5. Desenho dos Contornos

Os contornos são desenhados na imagem original para visualização.

---

### 6. Bounding Boxes

Criação de caixas delimitadoras ao redor dos objetos detectados:

```python
cv2.boundingRect(contorno)
```

✔ Inclui filtro para ignorar pequenos ruídos (área < 100).

---

### 7. Visualização do Pipeline

Exibição de todas as etapas:

* Imagem original
* Histograma
* Segmentação
* Pós-processamento
* Resultado final com bounding boxes

---

## 📊 Saídas do Sistema

O sistema exibe:

* Quantidade de contornos encontrados
* Quantidade total de objetos detectados
* Visualização completa do pipeline

---

## 🎯 Objetivo do Projeto

Demonstrar um fluxo completo de:

* Processamento de imagem
* Segmentação automática
* Extração de características
* Detecção de objetos

---

## 🚀 Possíveis Melhorias

* Ajuste dinâmico do tamanho do kernel
* Uso de filtros adicionais (Gaussian Blur, Median)
* Classificação dos objetos detectados
* Aplicação em tempo real (vídeo)

---

## 📌 Conclusão

O projeto mostra de forma prática como técnicas clássas de visão computacional podem ser combinadas para resolver problemas de detecção de objetos de forma eficiente e interpretável.

---
