
# 📧 Email/SMS Spam Classifier  

This project is a **machine learning-based classifier** designed to detect whether an email or SMS message is spam or ham (not spam). It utilizes **natural language processing (NLP) techniques** to preprocess text data by removing stopwords, stemming, and vectorizing the text using **TF-IDF** or **CountVectorizer**. Various **machine learning algorithms** such as **Naïve Bayes** and **Logistic Regression** are used to classify messages, and the model's performance is evaluated based on metrics like accuracy, precision, recall, and F1-score.  

The project follows a structured approach, with dedicated folders for datasets, model training, and application deployment. The dataset used for training is the **SMS Spam Collection** from the UCI Machine Learning Repository. The classifier is implemented in **Python** using libraries such as **Scikit-learn, Pandas, NumPy, and NLTK** for text processing. If applicable, an **interactive web interface** is built using **Streamlit** or **Flask**, allowing users to input messages and instantly check whether they are spam or not.  

To use the project, clone the repository and install the required dependencies listed. The web interface is provided, it can be launched using `streamlit run app.py`. The classifier achieves high accuracy, with **Naïve Bayes** model performing at **97-98% accuracy**. 
