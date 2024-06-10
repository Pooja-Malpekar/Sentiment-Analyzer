# Sentiment-Analyzer
# YouTube Sentiment Analysis
This project focuses on analyzing sentiment from YouTube video comments. Leveraging natural language processing (NLP) techniques and machine learning algorithms, we classify comments into positive, neutral, or negative sentiments.

Key Features:

-YouTube Comment Retrieval: Fetches comments from any YouTube video using its URL.
-Data Preprocessing: Cleans and preprocesses comments to make them suitable for sentiment analysis.
-Sentiment Classification: Utilizes machine learning models to categorize comments into positive, neutral, or negative sentiments.
-Visualizations: Provides graphical representations including pie charts and bar plots to illustrate the sentiment distribution.
-Word Cloud Generation: Creates word clouds for positive, neutral, and negative sentiments, offering insights into commonly used words.

Technologies Used:

-Python: The core programming language used for data processing, machine learning, and web development.
-Flask: A lightweight web framework for creating the web interface and handling HTTP requests.
-Pandas: For data manipulation and analysis.
-Scikit-learn: For implementing machine learning models such as Random Forest, Logistic Regression, Naive Bayes, and SVM.
-Matplotlib: For generating pie charts and bar plots.
-WordCloud: For generating word clouds based on comment sentiment.
-Bootstrap: For responsive and aesthetically pleasing web design.

Project Workflow:

-Input: Users provide a YouTube video URL.
-Comment Retrieval: The project fetches comments from the video using the YouTube API.
-Preprocessing: Comments are cleaned and transformed into a format suitable for sentiment analysis.
-Sentiment Analysis: The processed comments are classified using trained machine learning models.
-Visualization: Sentiment distributions are displayed via pie charts and bar plots. Additionally, word clouds are generated for each sentiment category.
-Analysis: Users can view detailed sentiment analysis results and visualizations on the web interface.

Comparative Study:
We compared several machine learning algorithms and found that Logistic Regression provided the best performance with 80% accuracy, making it the chosen model for this project.

Contributing:
We welcome contributions! Please fork the repository and submit a pull request with your changes.
