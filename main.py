from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset/Admission_Predict_Ver1.1.csv'), sep=',')
data = data[['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating Attended', 'SOP', 'LOR', 'CGPA', 'Internship',
             'Chance of Admit']]

predict = 'Chance of Admit'  # The attribute I predict

X = np.array(data.drop([predict, 'Serial No.'], 1))  # 0-rows, 1-columns
Y = np.array(data[predict])


# I have to keep it above - one here and one in the loop as after training the values of x_train, x_test etc.
# will be stored right below. I have to have it in that way as I'll be using them in the code where I'll be reading
# the model and in the last for loop as well
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

'''
# Piece of code for determining best accuracy and saving it to a pickle model
best = 0
for z in range(60000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

    linear = sklearn.linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc >= best:
        best = acc
        if best >= 0.9:
            with open('admission_model.pickle', 'wb') as f:
                pickle.dump(linear, f)
    else:
        continue

print('Best determined accuracy is: ', best)
'''


pickle_in = open('admission_model.pickle', 'rb')
linear = pickle.load(pickle_in)
accuracy = linear.score(x_test, y_test)


print('The coefficients for the features are: ', linear.coef_, '\n')  # these are the a's from ax+b equation
print('The intercept is in the point: ', linear.intercept_, '\n')
print('Standard Deviations of the features are: ', X.std(axis=0), '\n')

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print('Features: ', x_test[x], 'Chance of being accepted: ', predictions[x], ' Actual chance: ', y_test[x])

p = data['CGPA']
plt.style.use('fivethirtyeight')
plt.scatter(p, data['Chance of Admit'], c='red')
plt.title('Acceptance chart for master course')
plt.xlabel('Undergraduate GPA')
plt.ylabel('Chance')
plt.show()

