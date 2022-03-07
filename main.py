import dataset
import time
import model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Define Params
# ----------------
model_type = 'knn'
dataset_name = 'wine-quality'

model_args = {

    #nn params
    #"hidden_layer_sizes": (100, 100),
    #"solver": "adam",
    #"random_state": 100,
    #"learning_rate": "constant",
    #"learning_rate_init": 1e-2,

    #dtree params
    #"max_depth": 3,
    #"criterion": "gini"

    #boosted dtree
    #"classifier": {
    #    "max_depth": 3,
    #    "criterion": "gini"
    #},
    #"booster": {
    #    "learning_rate": .01,
    #    "n_estimators": 600
    #}

    #SVM
    #"C": 200,
    #"gamma": .01

    #KNN
    #"weights": "distance",
    "n_neighbors": 3
    

}

dataset_args = {
    "test_size": 0.3
}

cv_folds = 5
# ----------------


# Create model and dataset
m = model.Model(model_type, **model_args)
d = dataset.Dataset(dataset_name, **dataset_args)

# split data into test/train
X_train, X_test, y_train, y_test = d.get_dataset(dataset_name)

# perform cross validation
cross_val_scores = cross_val_score(m.model, X_train, y_train, cv=cv_folds)
m.compile_stats(X_train, y_train, cv_folds)


# fit model
train_st = time.time()
m.model.fit(X_train, y_train)
train_time = time.time() - train_st

#plt.plot(range(len(m.model.estimator_errors_)), m.model.train_score_, color='r')


# get prediction
predict_st = time.time()
pred = m.model.predict(X_test)
predict_time = time.time() - predict_st

# get accuracy of prediction
accuracy = accuracy_score(y_test, pred)
class_accuracy = m.get_class_accuracy(y_test, pred)
#m.print_nn_stats()

print(class_accuracy)

print('-----------------------------')
print('Model: ', model_type)
print('Dataset: ', dataset_name)
print('model params: ', model_args)
print('-------RESULTS-------')
print('Test Accuracy: ', round(accuracy*100, 2), '%')
print('Train time: ', round((train_time*1000), 2), 'ms')
print('Predict time:', round((predict_time*1000), 2), 'ms')
print('-----------------------------')
print()