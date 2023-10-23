# Fake News Detection using NLP and Machine Learning

## Overview

This Jupyter Notebook project is dedicated to the detection of fake news using Natural Language Processing (NLP) techniques. Two machine learning algorithms, namely Logistic Regression and Passive Aggressive Classifier, are employed to classify news articles as either real or fake.

## Algorithms Used

1. **Logistic Regression**
   - Accuracy: 54.01%
   - Logistic Regression is a fundamental classification algorithm used in this project. While it provides a baseline accuracy, the Passive Aggressive Classifier outperforms it.

2. **Passive Aggressive Classifier**
   - Accuracy: 65.20%
   - The Passive Aggressive Classifier is utilized to achieve higher accuracy in fake news detection. It's a popular choice for text classification tasks and demonstrates superior performance.

## Dataset

The project utilizes a labeled dataset containing both real and fake news articles. This dataset is crucial for training and evaluating the machine learning models.

## Project Structure

The Jupyter Notebook is structured as follows:

- **Data Preprocessing**: This section covers data cleaning, tokenization, and text normalization.

- **Feature Extraction**: Text data is transformed into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

- **Model Building**: Here, both Logistic Regression and Passive Aggressive Classifier models are trained and evaluated.

- **Model Evaluation**: The models' performances are assessed using accuracy as the primary metric. Confusion matrices and classification reports provide additional insights.

- **Conclusion**: A summary of the project's results and insights.

## Usage

To run the Jupyter Notebook and test the fake news detection models, follow these steps:

1. Ensure you have Jupyter Notebook installed. If not, you can install it using [Jupyter Notebook Installation Guide](https://jupyter.org/install).

2. Download the dataset and the Jupyter Notebook file.

3. Open the Jupyter Notebook in your preferred Python environment.

4. Run the cells sequentially to execute data preprocessing, feature extraction, model training, and evaluation.

5. Analyze the results and gain insights into the fake news detection accuracy.

## Results and Discussion

The project results show that the Passive Aggressive Classifier outperforms Logistic Regression, achieving an accuracy of 65.20% compared to 54.01%. This improvement in accuracy is significant in identifying fake news articles.

## Future Enhancements

- Incorporate more advanced NLP techniques, such as word embeddings (Word2Vec, GloVe) and deep learning models (LSTM, BERT), to further boost detection accuracy.

- Explore additional features like source credibility, sentiment analysis, and temporal analysis for better fake news identification.

- Optimize hyperparameters and fine-tune the models for better performance.

- Consider deploying the model as a web application or API for real-time fake news detection.

