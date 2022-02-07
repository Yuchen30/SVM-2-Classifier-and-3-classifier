import sys
from numpy import *
from svm import *
from os import listdir
from plattSMO import PlattSMO
import pickle

class LibSVM:

    def __init__(self,data=[],label=[],C=0,toler=0,maxIter=0,):
        self.classlabel = unique(label)
        self.classNum = len(self.classlabel)
        #train k(k-1)/2 SVM2 classifier
        self.classfyNum = (self.classNum * (self.classNum-1))/2
        self.classfy = []
        self.dataSet={}
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        m = shape(data)[0]
        for i in range(m):
            if label[i] not in self.dataSet.keys():
                self.dataSet[label[i]] = []
                self.dataSet[label[i]].append(data[i][:])
            else:
                self.dataSet[label[i]].append(data[i][:])
    def train(self):
        # An SVM binary classifier is trained for every two classes in the sample. loop iteratively join the classifier
        num = self.classNum
        for i in range(num):
            for j in range(i+1,num):
                data = []
                label = [1.0]*shape(self.dataSet[self.classlabel[i]])[0]
                label.extend([-1.0]*shape(self.dataSet[self.classlabel[j]])[0])
                data.extend(self.dataSet[self.classlabel[i]])
                data.extend(self.dataSet[self.classlabel[j]])
                svm = PlattSMO(array(data),array(label),self.C,self.toler,self.maxIter)
                svm.smoP()
                self.classfy.append(svm)
        self.dataSet = None
    def predict(self,data,label):
        m = shape(data)[0]
        num = self.classNum
        print(str(num)+"classification：")
        classlabel = []
        count = 0.0
        for n in range(m):
            result = [0] * num
            index = -1
            #Output the respective classifier accuracy
            for i in range(num):
                for j in range(i + 1, num):
                    index += 1
                    s = self.classfy[index]
                    t = s.predict([data[n]])[0]
                    if t > 0.0:
                        result[i] +=1
                    else:
                        result[j] +=1
                print("iter:"+str(n)+"\n"+str(i+1)+"the respective classifier accuracy:", count / m)
            classlabel.append(result.index(max(result)))
            if classlabel[-1] != label[n]:
                count +=1
                #print (label[n],classlabel[n])
        #print classlabel
        print("Total accuracy:",count / m)
        return classlabel
    def save(self,filename):
        fw = open(filename,'wb')
        pickle.dump(self,fw,2)
        fw.close()

    @staticmethod
    def load(filename):
        fr = open(filename,"rb")
        svm = pickle.load(fr)
        fr.close()
        return svm

def loadImage(dir,maps = None):
    dirList = listdir(dir)
    data = []
    label = []
    for file in dirList:
        label.append(file.split('_')[0])
        lines = open(dir +'/'+file).readlines()
        row = len(lines)
        col = len(lines[0].strip())
        line = []
        for i in range(row):
            for j in range(col):
                line.append(float(lines[i][j]))
        data.append(line)
        if maps != None:
            label[-1] = float(maps[label[-1]])
        else:
            label[-1] = float(label[-1])
    return data,label
def main():
    data,label = loadImage('./dataset/3/trainingDigits')
    svm = LibSVM(data, label, 200, 0.0001, 10000)
    svm.train()
    svm.save("svm.txt")
    svm = LibSVM.load("svm.txt")
    test,testlabel = loadImage('./dataset/3/testDigits')
    svm.predict(test,testlabel)

if __name__ == "__main__":
    sys.exit(main())


