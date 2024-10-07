import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'document': [
        'I Love playing football',
        'The new smartphone is amazing',
        'Foootball match tonight',
        'Latest technology trends',
        'Smartphone rreviews',
        'Football fans cheering'
    ],
    'category':[0, 1, 0, 1, 1, 0] # 0 = Sports category, 1 = Technology category
    
}

df = pd.DataFrame(data)
    
print(df)

X = df['document']
y = df['category']

#CountVectorizer tokeniz the text (splits into words) and creates binary vectors
#representing the present (1) or  absence (0) of each word.
vectorizer = CountVectorizer(binary=True)
X_vectorized = vectorizer.fit_transform(X)    

print(X_vectorized.toarray())

#splitting the data into training and test sets
 
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42) 

#Initialize the Naive Bayes classifier
clf = MultinomialNB()

#Train the classifier with the training data
clf.fit(X_train, y_train)

#Make predictions on the test set

y_pred = clf.predict(X_test)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:" ,accuracy)

#Create two new documents and clarrify them either as Sports or Technology

Sample_documents =[
    'Football fever grips the nation',
    'Exciting new smartphone launch event'
]

#Convert the sample documents into binary feature vetors
sample_X_documents = vectorizer.transform(Sample_documents)

print(sample_X_documents.toarray())

#Use the trained model to predict the category for these new documents
predictions = clf.predict(sample_X_documents)

print("Predictions for the sample documents", predictions)