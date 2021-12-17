import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv("data.csv")
label = data["spam"]

features = data[["money",'free','for','gambling','fun','machine','learning']]
print(features)

k = int(input("Input the k value for KNN model "))
# model = KNeighborsClassifier(metric='euclidean')  # Question 1
# model = KNeighborsClassifier(n_neighbors=k, metric='euclidean') # Question 2
# model = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='euclidean') # Question 3
model = KNeighborsClassifier(n_neighbors=k,metric='cosine') #Question 4

# Train the model using the training sets
model.fit(features,label)

#Predict Output
# Email to be predicted = machine learning for free = [0,1,1,0,0,1,1]
test = [0, 1, 1, 0, 0, 1, 1]
# test = ['free', 'for', 'machine', 'learning']
predicted= model.predict([test])
print(predicted)
