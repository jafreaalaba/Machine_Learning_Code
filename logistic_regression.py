import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#Create the dataset
data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Pass': [0,0,0,0,1,1,1,1,1,1] # 0 - fail, 1 - pass
}

df = pd.DataFrame(data)

X = df['Hours_Studied']

y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20, random_state=42)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)



y_pred = log_reg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy: .2f}')
print(f'Confusion: {conf_matrix}')

plt.scatter(X['Hours_Studied'], y, color='red', label='Actual Data')
X_range = np.linspace(0, 10, 1000).reshape(-1,1)
y_prob = log_reg_model.predict_proba(X_range)[:,1]

plt.plot(X_range, y_prob, color="blue", label="Logistic Regression Curve")
plt.xlabel('Hours_Studied')
plt.ylabel('Probability of Passing')
plt.title('Hours Studied vs Probability of Passing')
plt.legend()
plt.show()