import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
msg=pd.read_csv('P6-D.csv',names=['message','label'])
print('The dimensions are ',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
print('\n the total num of training data:',ytrain.shape,'\n the total test data:',ytest.shape)
cv=CountVectorizer()
xtrain_dtm=cv.fit_transform(xtrain)
xtest_dtm=cv.transform(xtest)
print('\n words or tokens are\n',cv.get_feature_names())
df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())
print(df)
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
from sklearn import metrics
print('\n accuracy of classifier is ',metrics.accuracy_score(ytest,predicted))
print('\n confusion matrix',metrics.confusion_matrix(ytest,predicted))
print('\n value of precision',metrics.precision_score(ytest,predicted))
print('\n value of recall ',metrics.recall_score(ytest,predicted))