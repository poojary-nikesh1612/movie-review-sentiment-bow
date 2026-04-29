# Movie Review Sentiment Analysis (Bag of Words)

This project classifies movie reviews as **positive** or **negative** using **Bag of Words (CountVectorizer)** and compares three machine learning models.

## Models Used

* Multinomial Naive Bayes
* K-Nearest Neighbors (KNN)
* Random Forest Classifier

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Workflow

1. Load dataset (`movies_sentiment_data.csv`)
2. Convert sentiment labels into numerical values

   * positive → 1
   * negative → 0
3. Split data into training and testing sets
4. Convert text into numerical features using `CountVectorizer`
5. Train three different models
6. Evaluate models using `classification_report`

## Project Structure

```text id="g2m8qp"
movie-review-sentiment-bow/
│
├── movie_review_sentiment.py
├── movies_sentiment_data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

```bash id="k8v2lm"
git clone your-repo-link
cd movie-review-sentiment-bow

pip install -r requirements.txt
python movie_review_sentiment.py
```

## Result

The performance of the three models is compared using precision, recall, and f1-score to identify the best model for movie sentiment classification. Multinomial Naive Bayes and Random Forest show good and similar performance, while KNN performs comparatively lower.

