# Emotion Recognition in Speech: A Data Mining Project

This repository documents a comprehensive data mining project focused on **recognizing human emotions from audio features**. Using the RAVDESS dataset, we apply a full-cycle data mining pipeline, from data preparation and exploratory analysis to predictive modeling, clustering, and pattern discovery.

The project was developed for the "Data Mining" course at the **University of Pisa (UniPi)**.

[![Read the Report](https://img.shields.io/badge/Read_the_Full-Report-red?style=for-the-badge&logo=adobeacrobatreader)](report.pdf)

---

## 📝 Table of Contents

- [Project Goal: Can We Hear Emotions?](#-project-goal-can-we-hear-emotions)
- [Our Approach: A Full Data Mining Pipeline](#-our-approach-a-full-data-mining-pipeline)
- [Technical Stack & Methodologies](#-technical-stack--methodologies)
- [Dataset: The RAVDESS Audio Dataset](#-dataset-the-ravdess-audio-dataset)
- [Project Workflow & Implemented Techniques](#-project-workflow--implemented-techniques)
- [Key Findings & Results](#-key-findings--results)
- [Repository Structure](#-repository-structure)
- [How to Run This Project](#-how-to-run-this-project)
- [Authors](#-authors)

---

## 🎯 Project Goal: Can We Hear Emotions?

The ability to automatically recognize emotions from speech has profound implications for human-computer interaction, mental health monitoring, and customer service analysis. This project tackles this challenge by exploring a rich dataset of audio recordings to answer the question: **"Can we build robust models to classify and understand the emotional content of human speech using only its acoustic features?"**

---

## 💡 Our Approach: A Full Data Mining Pipeline

Instead of focusing on a single task, this project implements a complete, end-to-end data mining workflow. We treat the problem holistically, performing in-depth analysis at each stage to extract meaningful insights. Our pipeline includes:

1.  **Data Understanding & Preparation**: Rigorous cleaning, preprocessing, and feature engineering.
2.  **Predictive Modeling (Classification)**: Training and evaluating models to predict the emotion expressed in an audio clip.
3.  **Pattern Discovery**: Using clustering and association rule mining to uncover hidden structures and relationships within the data.
4.  **Advanced Data Handling**: Utilizing regression models for sophisticated missing value imputation.

---

## 💻 Technical Stack & Methodologies

-   **Language**: **Python 3.x**
-   **Core Libraries**:
    -   **Pandas** & **NumPy**: For data manipulation, cleaning, and numerical operations.
    -   **Matplotlib** & **Seaborn**: For comprehensive data visualization and exploratory data analysis (EDA).
    -   **scikit-learn**: The primary library for implementing our machine learning pipeline, including preprocessing (scaling, encoding), classification, regression, and clustering algorithms.
    -   **`mlxtend`**: Used specifically for efficient frequent pattern mining (Apriori algorithm).
    -   **Jupyter Notebook**: The environment for all analysis, providing a clear, step-by-step narrative of our process.

---

## 📊 Dataset: The RAVDESS Audio Dataset

The core of this project is the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. The dataset consists of audio recordings from 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent.

-   **Emotions**: The recordings capture 8 distinct emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.
-   **Features**: For this project, we use a pre-processed version where various acoustic features have already been extracted from the audio files (e.g., MFCCs, chroma, spectral contrast). The final dataset contains **60 acoustic features**.

---

## ⚙️ Project Workflow & Implemented Techniques

The project is broken down into five key phases, each in its own dedicated folder.

1.  **Data Understanding & Preparation**
    -   **Activities**: In-depth EDA, visualization of feature distributions, correlation analysis, and handling of missing values. We apply outlier detection and feature scaling (Standardization) to prepare the data for modeling.
    -   **Notebooks**: `data_understanding_v2.ipynb`, `data_preparation.ipynb`

2.  **Classification (Emotion Prediction)**
    -   **Goal**: To predict the emotion from the 60 acoustic features.
    -   **Models Implemented**:
        -   **Decision Trees**: A simple yet powerful and interpretable model.
        -   **K-Nearest Neighbors (KNN)**: A distance-based classification algorithm.
        -   **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
    -   **Process**: For each model, we perform extensive hyperparameter tuning using Grid Search with cross-validation and evaluate performance using metrics like Accuracy, Precision, Recall, and F1-score.

3.  **Clustering**
    -   **Goal**: To determine if the audio samples form natural groupings that correspond to emotions or other underlying patterns.
    -   **Algorithms Used**:
        -   **Hierarchical Clustering**: To build a hierarchy of clusters and visualize it with dendrograms.
        -   **Centroid-Based (K-Means)**: To partition the data into a pre-defined number of clusters.
        -   **Density-Based (DBSCAN)**: To identify clusters of arbitrary shape and handle noise.

4.  **Pattern Mining**
    -   **Goal**: To discover frequent co-occurring patterns (itemsets) and association rules among the discretized acoustic features.
    -   **Method**: We use the **Apriori algorithm** to identify rules like *"if a sound has low spectral contrast and high MFCC_1, it is often associated with a 'sad' emotion."*

5.  **Regression (for Data Imputation)**
    -   **Goal**: As an advanced data preparation step, we explored using regression models to impute missing values in one of the key features (`'chroma_stft_mean'`).
    -   **Models Used**: Linear Regression, KNN Regressor, and Decision Tree Regressor. This allowed us to compare imputation strategies and prepare a more robust dataset for classification.

---

## 📈 Key Findings & Results

-   **Best Classification Model**: The **Decision Tree classifier** consistently achieved the best performance, reaching an **accuracy of approximately 75%** after hyperparameter tuning. This suggests that a rule-based separation of the feature space is highly effective for this task.
-   **Clustering Insights**: Clustering algorithms were able to identify some structure in the data, but the clusters did not perfectly align with the 8 emotion labels. This indicates that while acoustic features provide a strong signal, the boundaries between emotions can be acoustically ambiguous.
-   **Pattern Mining Discoveries**: Association rule mining successfully extracted interesting patterns, linking specific ranges of feature values to particular emotions, providing interpretable insights into the audio characteristics of emotional speech.
-   **Data Preparation is Crucial**: The performance of all models was heavily dependent on proper data preparation, particularly feature scaling and the strategic handling of outliers.

---

## 📂 Repository Structure

```

.
├── Dataset/
│   ├── ravdess\_features.csv            \# The primary dataset
│   └── dataset\_description.txt         \# Description of the features
├── Data Understanding \_ Preparation/
│   ├── data\_understanding\_v2.ipynb     \# Exploratory Data Analysis
│   ├── data\_preparation.ipynb          \# Preprocessing and cleaning pipeline
│   └── Data cleaned/ & Data prepared final/ \# Output CSVs from this stage
├── Classification/
│   ├── decision\_tree.ipynb
│   ├── KNN.ipynb
│   └── naive\_bayes.ipynb
├── Clustering/
│   ├── hierarchical\_clustering.ipynb
│   ├── centroid\_based\_clustering.ipynb
│   └── density\_based\_clustering.ipynb
├── Pattern Mining/
│   ├── pattern\_mining.ipynb            \# Association rule mining
│   └── ...
├── Regression/
│   ├── knn\_regression\_missing\_values.ipynb \# Imputation notebook
│   └── ...
├── report.pdf                            \# The final, detailed project report
└── README.md                             \# This file

````

---

## 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/danieleborghe/data_mining_1_project_UniPi.git](https://github.com/danieleborghe/data_mining_1_project_UniPi.git)
    cd data_mining_1_project_UniPi
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Explore the Notebooks:**
    -   It is recommended to start with the notebooks in the `Data Understanding _ Preparation` folder to understand the data.
    -   You can then run the notebooks in the `Classification`, `Clustering`, `Pattern Mining`, and `Regression` folders independently to replicate the analyses.

---

## 👥 Authors

- **Daniele Borghesi**
- **Lucrezia Labardi**
- **Vincenzo Sammartino**
