import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import warnings
import pickle
warnings.filterwarnings("ignore")

"""## Importing Datasets"""

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

"""## Identifying data types of different columns"""

train.info()

"""## Handling duplicates"""

train = train.drop_duplicates()

"""## Handling outliers"""

numerical_columns = ["latitude", "longitude"]

outlier_indices_to_drop = set()

for column in numerical_columns:
    Q1 = train[column].quantile(0.25)
    Q3 = train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_condition = (train[column] < lower_bound) | (train[column] > upper_bound)
    outlier_indices_to_drop.update(train[outliers_condition].index.tolist())

train = train.drop(outlier_indices_to_drop)
print(f"There are {len(outlier_indices_to_drop)} outliers in the train dataset")
print(f"Shape of the train_data after dropping outliers is {train.shape}")

"""## Data Cleaning and Preparation"""

# Combine both datasets to create a consistent store_id mapping
combined = pd.concat([train[['latitude', 'longitude', 'store_address']].dropna().drop_duplicates(),
                      test[['latitude', 'longitude']].dropna().drop_duplicates()], axis=0)


combined['store_id'] = combined.groupby(['latitude', 'longitude']).ngroup()

# Create the store_id_map
store_id_map = combined.dropna().set_index('store_address')['store_id'].to_dict()

# Prepare the training data with the new store_id
train = train.merge(combined[['latitude', 'longitude', 'store_id']].drop_duplicates(), on=['latitude', 'longitude'], how='left')
train = train.dropna(subset=['store_id'])

test = test.merge(combined[['latitude', 'longitude', 'store_id']].drop_duplicates(), on=['latitude', 'longitude'], how='left')
test = test.dropna(subset=['store_id'])

# Handle rating count
train['rating_count'] = pd.to_numeric(train['rating_count'], errors='coerce').fillna(0)
test['rating_count'] = pd.to_numeric(test['rating_count'], errors='coerce').fillna(0)

train.drop(columns=['id', 'store_address', 'category', 'store_name', 'latitude', 'longitude'], inplace=True)
test.drop(columns=['id', 'store_address', 'category', 'store_name', 'latitude', 'longitude'], inplace=True)

train.isnull().sum()

test.isnull().sum()

"""## Preprocessing Pipeline"""

# !pip install transformers torch

class TextReviewTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # New function to parse text-based review times
        def parse_review_time(text):
            today = pd.Timestamp.today()
            match = re.match(r"^(a|\d+)\s+(year|month|week|day|hour)s?\s+ago$", str(text).strip())
            if not match:
                return np.nan
            value_str, unit = match.groups()
            value = 1 if value_str == "a" else int(value_str)

            if unit == "year":
                delta = timedelta(days=365 * value)
            elif unit == "month":
                delta = timedelta(days=30 * value)
            elif unit == "week":
                delta = timedelta(weeks=value)
            elif unit == "day":
                delta = timedelta(days=value)
            elif unit == "hour":
                delta = timedelta(hours=value)
            else:
                return np.nan
            return today - delta

        # Apply the parsing and calculate days ago
        review_date = X_copy['review_time'].apply(parse_review_time)
        days_ago = (pd.Timestamp.today() - review_date).dt.days
        days_ago = days_ago.fillna(0)

        return pd.DataFrame(days_ago, columns=['review_time'])

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment", max_length=128, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode texts batch-wise for performance
        texts = X.iloc[:, 0].tolist()
        batch_size = 16
        vectors = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoding = self.tokenizer(batch_texts,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt")
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                # Use the CLS token representation as sentence embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(cls_embeddings)
        return np.vstack(vectors)

X_train, Y_train = train.drop(["rating"], axis=1), train[["rating"]]

time_transformer = Pipeline(steps=[
    ('time', TextReviewTimeTransformer()),
    ('scaler', StandardScaler())
])

preprocessor  = ColumnTransformer(
    transformers=[
        ('time', time_transformer, ['review_time']),
        ('bert', BertVectorizer(), ['review']),
        ('numeric_features', StandardScaler(), ['rating_count', 'store_id']),
    ],
    remainder='passthrough'
)

"""## Model Training"""

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf_base = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

xgb_base = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)

mlp_base = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    alpha=1e-4,
    learning_rate='adaptive',
    max_iter=300,
    random_state=42
)

rf_cal = CalibratedClassifierCV(rf_base, method='sigmoid', cv=3)
xgb_cal = xgb_base
mlp_cal = CalibratedClassifierCV(mlp_base, method='sigmoid', cv=3)

estimators = [
    ('rf', rf_cal),
    ('xgb', xgb_cal),
    ('mlp', mlp_cal),
]

meta_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_xgb,
    cv=5,
    passthrough=True,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stack_clf)
])

pipeline.fit(X_train, Y_train)

"""## Pickle the pipeline and store ID map"""

with open('final_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('store_map.pkl', 'wb') as f:
    pickle.dump(store_id_map, f)

print("Model and new store ID map have been pickled successfully!")

"""## Prediction"""

predictions = pipeline.predict(test)

submission_df = pd.DataFrame({'id': range(len(predictions)), 'rating': predictions})
submission_df.to_csv('submission.csv', index=False)
print("✅ Submitted csv file successfully!")

