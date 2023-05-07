import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, NormalPredictor, KNNBasic
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

print('Loading data sets...')
ratings_df = pd.read_csv('datasets/ratings.csv').head(1000)
# ratings_df = pd.read_csv('datasets/ratings.csv')
movies_df = pd.read_csv('datasets/movies.csv')

print('merging data sets...')
# Merge the datasets on the movieId column
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

print('printing first 5 rows of merged data set...')
# Print the first few rows of the merged dataset to verify the merge
print(merged_df.head())

# Check for missing data
print('check for null values...')
print(merged_df.isnull().sum())

# Drop rows with missing data
print('Removing any null values...')
merged_df = merged_df.dropna()

# Check the shape of the cleaned dataset
print('Checking shape...')
print(merged_df.shape)

# Split the dataset into training and testing sets
print('Splitting data into training and testing sets...')
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

# Print the number of rows in the training and testing sets
print('Printing rows in training and testing sets...')
print("Number of rows in training set:", train_df.shape[0])
print("Number of rows in testing set:", test_df.shape[0])

# Create a Surprise dataset from the ratings dataframe
print('Creating Surprise dataset from ratings dataframe...')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(merged_df[['userId', 'movieId', 'rating']], reader)

# Create an item-based similarity matrix
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

# print('Performing KNNBasic...')
# sim_options = {'name': 'cosine', 'user_based': True}
# algo_sim = KNNBasic(k=100, sim_options=sim_options)
# print('Fitting algo_sim model to trainset...')
# algo_sim.fit(trainset)
# print('Cross Validating algo_sim')
# cross_validate(algo_sim, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#
# # Predict ratings for the testset
# predictions = algo_sim.test(testset)
#
# # Compute the RMSE and MAE of the predictions
# rmse = accuracy.rmse(predictions)
# mae = accuracy.mae(predictions)
#
# print(f"RMSE: {rmse}")
# print(f"MAE: {mae}")

# # Define the SVD algorithm with L2 regularization
print('Defining the SVD algorithm with L2 regularization...')
param_grid = {'n_factors': [50, 100, 150], 'n_epochs': [20, 30], 'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)

print('Printing the best RMSE score and parameters...')
# Print the best RMSE score and parameters
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
params = gs.best_params['rmse']

# Train the SVD model using the best parameters found from GridSearchCV with 5-fold cross-validation
print('Training the SVD model using the best parameters found from GridSearchCV with 5-fold cross-validation...')
# algo = gs.best_estimator['rmse']
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
svd_df = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'],
             lr_all=params['lr_all'], reg_all=params['reg_all'])
cross_validate(svd_df, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# cross_validate(algo_sim, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# random = NormalPredictor()

svd_df.fit(trainset)

predictions = svd_df.test(testset)
print('Predictions type: ', type(predictions))

# Generate predictions using item-based collaborative filtering


# predictions_item_based_filtering = algo.test(testset)
# predictions_algo = algo.test(testset)
# print("RMSE of svd_df on test set:", accuracy.rmse(predictions))
# print("RMSE of algo on test set:", accuracy.rmse(predictions_algo))
# print("RMSE of predictions_item_based_filtering on test set:", accuracy.rmse(predictions_item_based_filtering))

# Save the merged_df, train_df, test_df, movies_df, svd_df, and algo objects to disk
print('Saving model to disk...')
with open('data10.pkl', 'wb') as f:
    # pickle.dump((merged_df, train_df, test_df, movies_df, svd_df, algo), f)
    # pickle.dump((merged_df, train_df, test_df, movies_df, algo), f)
    pickle.dump((merged_df, train_df, test_df, movies_df, svd_df), f)
    # pickle.dump((merged_df, train_df, test_df, movies_df, algo_sim), f)

# # Define file name and path to save the model
# filename = 'svd_model3.pkl'
#
# # Open the file in write binary mode
# with open(filename, 'wb') as file:
#     # Save the trained model to the file using pickle's dump method
#     pickle.dump(svd, file)
#     # Close the file
#     file.close()

# Notes on RSME values:
#
# The RMSE values you are getting suggest that your model is performing better on the training data than on the
# validation data. The training RMSE of around 0.63 means that the model is able to predict the ratings for the
# movies in the training set with an average error of 0.63 stars, which is quite good. However, the validation
# RMSE of around 1 means that the model is not performing as well on the validation set, which suggests that it
# may be overfitting to the training data.
#
# Overfitting occurs when the model is too complex and starts to fit the noise in the training data, instead of the
# underlying patterns. This leads to poor performance on new, unseen data. To address overfitting, you can try
# adjusting the hyperparameters of your model, such as the learning rate or regularization strength, or you can try
# using a simpler model.
#
# It's important to keep in mind that the RMSE is just one metric and should be used in conjunction with other
# evaluation metrics, such as precision and recall, to get a more complete picture of your model's performance.
# It's also a good idea to compare your model's performance to a baseline model, such as a model that always predicts
# the average rating, to see how much of an improvement your model is providing.


# Data4.pkl terminal results

# Loading data sets...
# merging data sets...
# printing first 5 rows of merged data set...
#    userId  movieId  ...                                             title  genres
# 0       1      307  ...  Three Colors: Blue (Trois couleurs: Bleu) (1993)   Drama
# 1       6      307  ...  Three Colors: Blue (Trois couleurs: Bleu) (1993)   Drama
# 2      56      307  ...  Three Colors: Blue (Trois couleurs: Bleu) (1993)   Drama
# 3      71      307  ...  Three Colors: Blue (Trois couleurs: Bleu) (1993)   Drama
# 4      84      307  ...  Three Colors: Blue (Trois couleurs: Bleu) (1993)   Drama
#
# [5 rows x 6 columns]
# check for null values...
# userId       0
# movieId      0
# rating       0
# timestamp    0
# title        0
# genres       0
# dtype: int64
# Removing any null values...
# Checking shape...
# (27753444, 6)
# Splitting data into training and testing sets...
# Printing rows in training and testing sets...
# Number of rows in training set: 22202755
# Number of rows in testing set: 5550689
# Creating Surprise dataset from ratings dataframe...
# Defining the SVD algorithm with L2 regularization...
# Printing the best RMSE score and parameters...
# 0.8065389469344793
# {'n_factors': 50, 'reg_all': 0.02}
# Printing the best RMSE score and parameters...
# 0.8065389469344793
# {'n_factors': 50, 'reg_all': 0.02}
# Training the SVD model using the best parameters found from GridSearchCV with 5-fold cross-validation...
# Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
#
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
# RMSE (testset)    0.8072  0.8068  0.8068  0.8061  0.8066  0.8067  0.0003
# MAE (testset)     0.6104  0.6102  0.6099  0.6096  0.6100  0.6100  0.0003
# Fit time          244.41  250.00  255.44  247.41  250.90  249.63  3.68
# Test time         75.70   71.44   72.02   64.76   81.41   73.07   5.46
# RMSE: 0.7144
# RMSE on test set: 0.7143978808308666
# Saving model to disk...
