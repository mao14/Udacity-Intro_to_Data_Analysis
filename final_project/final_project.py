import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodecsv

"""
plt.subplots() is a function that returns a tuple containing a figure and axes object(s).
Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax.
Having fig is useful if you want to change figure-level attributes or save the figure as
an image file later (e.g. with fig.savefig('yourfilename.png').
You certainly don't have to use the returned figure object but many people do use it later
so it's common to see. Also, all axes objects (the objects that have plotting methods),
have a parent figure object anyway, thus:

    fig, ax = plt.subplots()
is more concise than this:

    fig = plt.figure()
    ax = fig.add_subplot(111)
"""

# import the data
data = pd.read_csv('titanic_data.csv')

# Analysis for survival
survived = data.sum()['Survived']
print(survived)

print('% of survival')
print(float(survived)/len(data))


# ### Information by class

data.Pclass.plot(kind='hist')
plt.xlabel("Class")
plt.title("Class Distribution")


# ### Information by Age

data.Age.plot(kind='hist')
plt.xlabel('Age')
plt.title('Age Distribution')


# ### Information by port of embarcation

data_by_port = data.groupby('Embarked')
total = data_by_port['Survived'].count()

index = np.arange(len(data_by_port))
x_labels = ['Cherbourg','Queenstown','Southampton']

plt.xlabel('Port')
plt.bar(index, total)
plt.xticks(index, x_labels)


# ### Analysis by Age and class

data.Age[data.Pclass == 3].plot(kind='kde')
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 1].plot(kind='kde')
plt.xlabel("Age")
plt.title("Age Distribution, within classes")
plt.legend(('3rd Class', '2nd Class','1st Class'))


# ### Analysis by Age and Sex

data.Age[data.Sex == 'male'].plot(kind='hist')
data.Age[data.Sex == 'female'].plot(kind='hist')
plt.xlabel('Age')
plt.title('Age distribution - by sex')
plt.legend(('Male', 'Female'))


# ### Analysis by Age and class

data.Age[data.Pclass == 1].plot(kind='kde')
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.title("Age Distribution, within classes")
plt.legend(('1st Class', '2nd Class','3rd Class'))


# ### Analysis Survived by Pclass

data_by_class = data.groupby('Pclass')
total = data_by_class['Survived'].count()
survived = data_by_class['Survived'].sum()
not_survived= total - survived
perc_survived = (survived) / (total)
print(perc_survived)

index = np.arange(len(data_by_class))
x_labels = ['Class1','Class2','Class3']

plt.bar(index,total,color='blue', width=0.3)
plt.bar(index+0.3,survived, color='grey',width=0.3)
plt.bar(index+0.6,not_survived, color='black',width=0.3)
plt.legend(('Total','Survived','Not Survived'),loc='best')
plt.xticks(index+0.3, x_labels)


# ### Analysis Survived by Age

data_by_age = data.groupby(pd.cut(data['Age'],[0,10,20,28,40,50,100]))
total = data_by_age['Survived'].count()
survived = data_by_age['Survived'].sum()
perc_survived = survived / total

print(perc_survived)

index = np.arange(len(data_by_age))
x_labels = ['0-10','10-20','20-28','28-40','40-50','50-100']

plt.bar(index, total, color = 'blue', width=0.4)
plt.bar(index+0.4, survived, color = 'gray', width=0.4)
plt.legend(('Total','Survived'),loc='best')
plt.xticks(index+0.2, x_labels)


# ### Analysis Survived by Sex

data_by_sex = data.groupby(['Sex'])
total = data_by_sex['Survived'].count()
survived = data_by_sex['Survived'].sum()
perc_survived = survived / total

print(perc_survived)
print(survived)


# ### Analysis Survived by Class and Sex

data_by_class_sex = data.groupby(['Pclass','Sex'])

total_males = []
total_females = []
males_survived = []
females_survived = []

for i in range(1,4):
    total_males.append(data_by_class_sex['Survived'].count()[i][1])
    total_females.append(data_by_class_sex['Survived'].count()[i][0])
    males_survived.append(data_by_class_sex['Survived'].sum()[i][1])
    females_survived.append(data_by_class_sex['Survived'].sum()[i][0])

print(total_males)
print(total_females)
print(males_survived)
print(females_survived)

#Information for females
index = np.arange(3)
ax2 = plt.subplot2grid((2,2),(0,0),colspan=2)

rects1 = ax2.bar(index,total_females,width=0.4,color='blue')
rects2 = ax2.bar(index+0.4,females_survived,width=0.4,color='green')
ax2.set_xticks(index+0.4)
ax2.set_xticklabels(('Class1','Class2','Class3'))
ax2.legend((rects1[0],rects2[0]),('Total','Survived'),loc='best')

#Information for males
index = np.arange(3)
ax2 = plt.subplot2grid((2,2),(0,0),colspan=2)

rects1 = ax2.bar(index,total_males,width=0.4,color='blue')
rects2 = ax2.bar(index+0.4,males_survived,width=0.4,color='green')
ax2.set_xticks(index+0.4)
ax2.set_xticklabels(('Class1','Class2','Class3'))
ax2.legend((rects1[0],rects2[0]),('Total','Survived'),loc='best')


# ### Analysis Survived by Port

data_by_port = data.groupby('Embarked')
total = data_by_port['Survived'].count()
survived = data_by_port['Survived'].sum()
perc_survived = survived / total

print(perc_survived)

index = np.arange(len(data_by_port))
x_labels = ['Cherbourg','Queenstown','Southampton']

plt.xlabel('Port')
plt.bar(index, total, width = 0.4, color= 'gray')
plt.bar(index+0.4, survived, width = 0.4, color= 'blue')
plt.legend(('Total','Survived'))
plt.xticks(index+0.4, x_labels)


# ### Analysis by Port and class

data_by_port_class = data.groupby(['Pclass','Embarked'])

total_class1 = data_by_port_class['Survived'].count()[1]
total_class2 = data_by_port_class['Survived'].count()[2]
total_class3 = data_by_port_class['Survived'].count()[3]

print(total_class1)
print(total_class2)
print(total_class3)

index = np.arange(len(data_by_port))
x_labels = ['Cherbourg','Queenstown','Southampton']

plt.bar(index, total_class1, color='blue', width=0.2)
plt.bar(index+0.2, total_class2, color='green', width=0.2)
plt.bar(index+0.4, total_class3, color='black', width=0.2)
plt.legend(('Class1','Class2','Class3'),loc='best')
plt.xticks(index+0.2, x_labels)
