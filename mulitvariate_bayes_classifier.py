# Naive Bayes On The Iris Dataset
from csv import reader
from random import seed
from random import randrange
from math import exp
from math import pi
import numpy as np


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:

        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None # set class to 0 for validation since we don't know it yeet
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate mean arrays
def means(datalist):
    mean = np.array([])
    for i in range(len(datalist)):
        this_mean = sum(datalist[i])/len(datalist[i])
        mean = np.append(mean, this_mean)
    return mean


# Calculate covariance arrays
def covariance_creator(datalist,mean):
    covariances = np.zeros((4, 4))
    for i in range(len(datalist)):
        for j in range(len(datalist)):
            for k in range(len(datalist[1])):
                r = ((datalist[i][k] - mean[i]) * (datalist[j][k] - mean[j]))
                covariances[i][j] = covariances[i][j] + r/len(datalist[1])
    return covariances


# Calculate the mean, cov and count for each dataset
def summarize_dataset(dataset):
    columns = [column for column in zip(*dataset)]
    del(columns[-1])	  # remove classification from array
    mean = means(columns)
    length = len(columns[1])
    cov = covariance_creator(columns, mean)
    summaries = [mean, cov, length]
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    multiply1 = np.dot(np.transpose(x-mean),inv_cov)
    multiply2 = np.dot(multiply1, (x-mean))
    exponent = exp(-.5*multiply2)
    return (1 / ((2 * pi)**(4/2) * (np.linalg.det(cov))**.5)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    del(row[-1])
    total_rows = sum([summaries[label][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][2]/float(total_rows)
        mean, cov, _ = class_summaries
        probabilities[class_value] *= calculate_probability(row, mean, cov)	 # discriminant function
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Naive Bayes Algorithm
def multivariate_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# Test Naive Bayes on Iris Dataset
seed(1)
filename = "iris-original.data"
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, multivariate_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))