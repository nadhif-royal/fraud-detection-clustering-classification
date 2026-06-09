# Bank Transaction Analysis: Clustering & Classification Pipeline 🏦📊

An end-to-end Machine Learning project applying both **Unsupervised Learning (Clustering)** and **Supervised Learning (Classification)** to analyze bank transaction data, uncover customer personas, and build predictive models. 

> **Note:** This project is the final submission for the **"Belajar Machine Learning untuk Pemula"** (Machine Learning for Beginners) course by **Dicoding Indonesia**.

---

## 🎯 Project Overview

The main objective of this project is to integrate two fundamental Machine Learning approaches to solve a business problem:
1. **Unsupervised Learning:** Applying K-Means Clustering to an unlabeled bank transaction dataset to segment customers based on their financial behavior and transaction patterns.
2. **Supervised Learning:** Using the clustered labels from the first stage as the target variable to train Classification models (Decision Tree & Random Forest) capable of predicting the segment of new, incoming transaction data.

## 🗂️ Directory Structure

```text
├── data/                   # Datasets used and generated in the project
│   ├── bank_transactions_data_edited.csv
│   ├── data_clustering.csv
│   └── data_clustering_inverse.csv
├── models/                 # Saved Machine Learning models (.pkl and .h5)
│   ├── model_clustering.pkl
│   ├── PCA_model_clustering.h5
│   ├── decision_tree_model.h5
│   ├── explore_RandomForest_classification.h5
│   └── tuning_classification.h5
├── notebooks/              # Jupyter Notebooks containing the code
│   ├── [Clustering]_Submission.ipynb
│   └── [Klasifikasi]_Submission.ipynb
└── README.md

```

## 🛠️ Methodology & Tech Stack

* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Yellowbrick.
* **Phase 1 - Clustering:** * Extensive Exploratory Data Analysis (EDA) and handling outliers using the IQR method.
* Feature engineering including Binning and One-Hot Encoding.
* Determining the optimal 'K' using the **Elbow Method**.
* Model training using **K-Means** and **PCA (Principal Component Analysis)** for 2D visualization.


* **Phase 2 - Classification:** * Data splitting and stratifying.
* Training baseline model using **Decision Tree**.
* Exploring advanced ensemble models using **Random Forest Classifier**.
* Optimizing model performance using **Hyperparameter Tuning (GridSearchCV)**.



## 📈 Key Insights

Through the inverse transformation of the clustered data, the K-Means algorithm successfully identified three distinct customer personas based strictly on their categorical behaviors (such as Transaction Type, Channel, and Device used), rather than purely numerical balances:

* **Cluster 0:** Moderate-income users with stable balances, representing a low-risk, highly reliable segment.
* **Cluster 1:** Senior and highly conservative users focusing on low-value transactions.
* **Cluster 2:** Younger, highly active transactional users with high turnover rates, ideal for lifestyle banking promotions.

The tuned Random Forest classification model successfully predicted these clusters with an exceptional **Accuracy and F1-Score of 99%**.

## 👨‍💻 Author

**Nadhif Rif'at Rasendriya** Undergraduate Student of Informatics Engineering

Universitas Brawijaya (FILKOM UB)

* GitHub: [@nadhif-royal](https://github.com/nadhif-royal)
