from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
import pickle

# Генерируем синтетические данные
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Делим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем объект модели логистической регрессии
model = LogisticRegression()

# Обучаем модель на обучающем наборе
model.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе
y_pred = model.predict(X_test)

# Оцениваем качество модели
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
# Выводим метрики качества
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)