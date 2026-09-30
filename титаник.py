import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import math
train_data = pd.read_csv("C:\\Users\\iturb\\PycharmProjects\\Mашинное обучение\\данные\\train.csv")
test_data = pd.read_csv("C:\\Users\\iturb\\PycharmProjects\\Mашинное обучение\\данные\\test.csv")
#обработка данных
train_data_one=train_data.loc[train_data['Age'].notna()]
train_data_male_one=train_data_one.loc[(train_data_one['Sex']=='male') & (train_data_one['Pclass']==1)]
train_data_male_too=train_data_one.loc[(train_data_one['Sex']=='male') & (train_data_one['Pclass']==2)]
train_data_male_fre=train_data_one.loc[(train_data_one['Sex']=='male') & (train_data_one['Pclass']==3)]
train_data_female_fre=train_data_one.loc[(train_data_one['Sex']=='female') & (train_data_one['Pclass']==3)]
train_data_female_too=train_data_one.loc[(train_data_one['Sex']=='female') & (train_data_one['Pclass']==2)]
train_data_female_one=train_data_one.loc[(train_data_one['Sex']=='female') & (train_data_one['Pclass']==1)]
sred_age_female_one=math.floor(train_data_female_one['Age'].mean())
sred_age_female_too=math.floor(train_data_female_too['Age'].mean())
sred_age_female_fre=math.floor(train_data_female_fre['Age'].mean())
sred_age_male_fre=math.floor(train_data_male_fre['Age'].mean())
sred_age_male_one=math.floor(train_data_male_one['Age'].mean())
sred_age_male_too=math.floor(train_data_male_too['Age'].mean())
#написанно qwen2.5
age_means = {
    'female_1': sred_age_female_one,
    'female_2': sred_age_female_too,
    'female_3': sred_age_female_fre,
    'male_1': sred_age_male_one,
    'male_2': sred_age_male_too,
    'male_3': sred_age_male_fre
}
train_data['Age'] = train_data['Age'].fillna(train_data.apply(lambda row: age_means[f"{row['Sex']}_{row['Pclass']}"], axis=1))
test_data['Age'] = test_data['Age'].fillna(test_data.apply(lambda row: age_means[f"{row['Sex']}_{row['Pclass']}"], axis=1))
#----------------------------------------------
print(sred_age_male_one, sred_age_female_one, sred_age_male_too,sred_age_female_too,sred_age_male_fre,sred_age_female_fre)
train_data.drop(columns=['Cabin'], inplace=True)
test_data.drop(columns=['Cabin'], inplace=True)
train_data=pd.get_dummies(train_data,columns=['Sex','Pclass'],dtype=int)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Pclass'], dtype=int)
sred_age=train_data['Age'].mean()
otkl_age=train_data['Age'].std()
train_data['Age']=(train_data['Age']-sred_age)/otkl_age
test_data['Age']=(test_data['Age']-sred_age)/otkl_age
y_train = train_data['Survived']
# Укажите признаки, на которых вы хотите обучить модель
features = ['Age','Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3']
X_train = train_data[features]
X_test = test_data[features]
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_tr, y_tr)
predict_val=model.predict(X_val)
correct_answers=(predict_val == y_val).sum()
total_answers = len(y_val)
accuracy = (correct_answers / total_answers) * 100
print( accuracy)
predict_test=model.predict(X_test)
#Написано gemini
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"], # Берем ID пассажиров из исходного теста
    "Survived": predict_test                 # Записываем предсказания модели
})
submission.to_csv("submission.csv", index=False)
