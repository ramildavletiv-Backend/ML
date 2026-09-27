import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)  # для воспроизводимости результатов

# Чтение файлов
train_data = pd.read_csv("C:/Users/ramil/Downloads/titanic/train.csv")
test_data = pd.read_csv("C:/Users/ramil/Downloads/titanic/test.csv")


print(test_data['Age'])
prediction = []


for i in range(len(test_data)):
    age = test_data['Age'][i]
    sex = test_data['Sex'][i]
    pclass = test_data['Pclass'][i]

    if sex == 'male'and age > 10 and age < 37 and pclass == 1:
        prediction.append (1)
    elif sex == 'male' and age > 0 and age < 16 and pclass == 2:
        prediction.append(1)
    elif sex == 'male' and age > 0 and age < 16 and pclass == 3:
        prediction.append(1)
    elif sex == 'male' and age > 29 and age < 31 and pclass == 3:
        prediction.append(1)
    elif sex == 'female' and age > 10 and age < 37 and pclass == 1:
        prediction.append(1)
    elif sex == 'female' and age > 0 and age < 16 and pclass == 2:
        prediction.append(1)
    elif sex == 'female' and age > 0 and age < 16 and pclass == 3:
        prediction.append(1)
    elif sex == 'female' and age > 29 and age < 31 and pclass == 3:
        prediction.append(1)
    else:
        prediction.append (0)


# Создание датафрейма
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': prediction,
})

# Сохранение результатов
submission.to_csv('submission.csv', index=False)



print(train_data.head()) # показывает первые 5 строк таблицы

print(train_data.shape) # возвращает кортеж (число строк, число столбцов). Позволяет сразу оценить размер датасета.

print(train_data.dtypes) # показывает тип данных каждой колонки. Помогает заметить, например, что числовая колонка случайно сохранилась как строка (object).

print(train_data.describe()) # азовые статистики (среднее, минимум, максимум, квартили) по числовым колонкам. Полезен для поиска некорректных значений: например, минимальная стоимость билета не может быть отрицательной.

print(train_data.info())  # свод по датафрейму: число строк, колонок, типы данных и количество непустых (non-null) значений в каждой колонке. Главный инструмент для быстрого поиска пропусков.
