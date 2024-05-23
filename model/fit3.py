import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных из CSV-файла
library = pd.read_csv('library.csv')

# Разделение данных на признаки (X) и целевую переменную (y)
X = library.drop(columns=['Books'])
y = library['Books']

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(X, y)

# Предсказание для новых данных
prediction = model.predict([[10, 2]])
print("Предсказание:", prediction)

# Экспорт модели с помощью Pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Модель сохранена в файл model.pkl")