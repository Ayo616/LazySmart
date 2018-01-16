import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


filename = "./shiyan_all.csv"
targetname = 'people'
none_feature = 'time'

## save data
train = []
label = []



def load_data(filename,targetname,none_feature):
    data = pd.read_csv(filename)
    raw = pd.DataFrame(data)
    labels = []
    feature = []
    for item in raw.columns:
        if item == targetname:
            labels.append(item)
            print(labels)
        else:
            if item != none_feature:
                feature.append(item)

    train = pd.DataFrame(data,columns=feature)
    label = pd.DataFrame(data,columns=labels)
    print('===================================')
    print('** train data for ten examples **')
    print('===================================')
    print(train.head(),'\n')

    print('===================================')
    print('** label data for ten examples **')
    print('===================================')
    print(label.head(),'\n')

    return train,label

'''
do preprocess here
'''
def preprocess_data(train,label):

    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(train)
    print('==========================================')
    print('>>>>>> standstard preprocess done >>>>>')
    print('==========================================','\n')
    train = train
    label = label

    return train,label


def visiable_data(train,label,targetname):
    import matplotlib.pyplot as plt
    import seaborn as sns
    raw = pd.concat([train,label],axis=1)
    sns.set()#使用默认配色
    sns.pairplot(raw,hue=targetname)#hue 选择分类列
    plt.show()

def split_data(train,label):
    from sklearn.cross_validation import train_test_split
    train_data,predi_data,train_label,predi_label = \
        train_test_split(train,label,test_size=0.3,random_state=0)
    print('==========================================')
    print('>>>>>> split data process done >>>>>')
    print('>>>>> train amount',len(train_data),'>>>>>>>>')
    print('>>>>> test amount',len(predi_data),'>>>>>>>>')
    print('==========================================','\n')
    return train_data,train_label,predi_data,predi_label


def algorithm(train_data,train_label,predi_data,predi_label):
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn import linear_model

    models = []
    model_GaussianNB = GaussianNB()
    model_LogisticRegression = LogisticRegression()
    model_KNeighborsClassifier = KNeighborsClassifier()
    model_DecisionTreeClassifier = DecisionTreeClassifier()
    model_SVC = SVC()
    models.append(model_DecisionTreeClassifier)
    models.append(model_GaussianNB)
    models.append(model_LogisticRegression)
    models.append(model_SVC)
    models.append(model_KNeighborsClassifier)

    for model in models:
        model.fit(train_data, train_label)
        # make predictions
        expected = predi_label
        predicted = model.predict(predi_data)
        # summarize the fit of the model
        print('=========================================================')
        print(type(model))
        print('=========================================================')
        print(model)
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted),'\n')


def tpot(X_train, y_train,X_test, y_test):
    from tpot import TPOTClassifier
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))



if __name__ == '__main__':
    train,label = load_data(filename,targetname,none_feature)
    preprocess_data(train,label)
    # visiable_data(train,label,targetname)
    train_data,train_label,predi_data,predi_label = split_data(train,label)
    algorithm(train_data,train_label,predi_data,predi_label)
    tpot(train_data,train_label,predi_data,predi_label)
