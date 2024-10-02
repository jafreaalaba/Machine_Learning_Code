import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre']) #dropping the target values

y = music_data['genre']  #target or output values



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#model.fit(X_train, y_train)

#joblib.dump(model, 'music-recommender.joblib')


model = joblib.load('music-recommender.joblib')


predictions = model.predict([[21, 1]])

print(predictions)

#score = accuracy_score(y_test, predictions)

#print(score)

tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True,
                     )



