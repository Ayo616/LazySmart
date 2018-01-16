import itchat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import  sys

from LazySmart.Logger import Logger
from LazySmart.wechat import callme


class lazsmt(object):


        def __init__(self):
            self.data =  None
            self.train = None
            self.label = None
            self.train_data = None
            self.train_label = None
            self.predi_data = None
            self.predi_label = None
            self.filename = "./shiyan_all.csv"
            self.targetname = 'people'
            self.none_feature = 'time'




        def load_data(self):
            self.data = pd.read_csv(self.filename)
            raw = pd.DataFrame(self.data)
            labels = []
            feature = []
            for item in raw.columns:
                if item == self.targetname:
                    labels.append(item)
                else:
                    if item != self.none_feature:
                        feature.append(item)

            self.train = pd.DataFrame(self.data,columns=feature)
            self.label = pd.DataFrame(self.data,columns=labels)
            print('===================================')
            print('** train data for ten examples **')
            print('===================================')
            print(self.train.head(),'\n')

            print('===================================')
            print('** label data for ten examples **')
            print('===================================')
            print(self.label.head(),'\n')

            return self.train,self.label

        '''
        do preprocess here
        '''
        def preprocess_data(self):

            min_max_scaler = preprocessing.MinMaxScaler()
            a = min_max_scaler.fit_transform(self.train)
            print('==========================================')
            print('>>>>>> standstard preprocess done >>>>>')
            print('==========================================','\n')
            self.train = a
            self.label = self.label

            return self.train,self.label


        def visiable_data(self):
            import matplotlib.pyplot as plt
            import seaborn as sns
            raw = pd.concat([self.data,self.label],axis=1)
            sns.set()#使用默认配色
            sns.pairplot(raw,hue=self.targetname)#hue 选择分类列
            plt.show()

        def split_data(self):
            from sklearn.cross_validation import train_test_split
            self.train_data,self.predi_data,self.train_label,self.predi_label = \
                train_test_split(self.train,self.label,test_size=0.3,random_state=0)
            print('==========================================')
            print('>>>>>> split data process done >>>>>')
            print('>>>>> train amount',len(self.train_data),'>>>>>>>>')
            print('>>>>> test amount',len(self.predi_data),'>>>>>>>>')
            print('==========================================','\n')
            return self.train_data,self.train_label,self.predi_data,self.predi_label


        def algorithm(self):
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
                model.fit(self.train_data, self.train_label)
                # make predictions
                expected = self.predi_label
                predicted = model.predict(self.predi_data)
                # summarize the fit of the model
                print('=========================================================')
                print(type(model))
                print('=========================================================')
                print(model)
                print(metrics.classification_report(expected, predicted))
                print(metrics.confusion_matrix(expected, predicted),'\n')


        def tpot(self):
            from tpot import TPOTClassifier
            tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
            tpot.fit(self.train_data, self.train_label)
            print(tpot.score(self.predi_data, self.predi_label))


if __name__ == '__main__':

    # Send Me
    itchat.login()


    # save log
    sys.stdout = Logger("./data.txt")
    lazysmart = lazsmt()
    lazysmart.load_data()
    lazysmart.preprocess_data()
    # visiable_data(train,label,targetname)
    lazysmart.split_data()
    lazysmart.algorithm()
    # lazysmart.tpot()



    # Send Me Now
    #想给谁发信息，先查找到这个朋友,name后填微信备注即可,deepin测试成功
    users =itchat.search_friends(name='陈超')
    #获取好友全部信息,返回一个列表,列表内是一个字典
    # print(users)
    #获取`UserName`,用于发送消息
    userName = users[0]['UserName']
    #文件地址
    itchat.send_file('data.txt',toUserName = userName)
    print("send already")