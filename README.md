# Stock Market Prediction – Deep Learning Models

This project predicts stock market movements (Up/Down) based on daily news headlines using deep learning models.

---

## 🔧 Features

- Preprocessing of financial news headlines
- Text vectorization using TF-IDF
- Deep Learning models built and evaluated:
  - BiLSTM
  - CNN (1D Convolution)
  - BiGRU
- Confusion matrices, classification reports, and full metric evaluations (Accuracy, Precision, Recall, F1 Score)

---

## ⚙️ Tech Stack

- Python 3.11
- TensorFlow / Keras
- Scikit-Learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SriVenkateshMani/Stock_prediction.git cd Stock_prediction
```


### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```


### 3. Run the notebook

- Open `Stock_models.ipynb` using Jupyter Notebook.
- Run all cells sequentially to preprocess the data, train the models, and evaluate results.

---

## 📈 Models Implemented

| Model | Test Accuracy | Precision | Recall | F1 Score |
|:------|:--------------|:----------|:-------|:---------|
| BiLSTM | 0.5151 | 0.5479 | 0.4882 | 0.5163 |
| CNN (1D Convolution) | 0.5251 | 0.5390 | 0.7204 | 0.6166 |
| BiGRU | 0.5327 | 0.5676 | 0.4976 | 0.5303 |

---

## 📊 Key Observations

- Predicting stock movement purely from headlines is extremely challenging due to news volatility.
- Different architectures show different strengths: CNN achieved the best Recall, while BiGRU had the best overall Accuracy.
- Preprocessing quality and model design are critical in such financial prediction tasks.

---

## 📁 Project Structure

- `dataset_mp.csv` – Input dataset (news headlines + stock movement labels)
- `Stock_models.ipynb` – Complete project notebook (preprocessing, models, evaluations)

---

## 🌟 Future Work

- Incorporating pre-trained embeddings (GloVe, Word2Vec) for better language understanding.
- Exploring Transformer-based models like BERT for deeper contextual modeling.
- Combining news data with stock price indicators for improved prediction accuracy.

---

## 👨‍💻 Author

**Sri Venkatesha Mani**  
[GitHub Profile](https://github.com/SriVenkateshMani)
