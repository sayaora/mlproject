import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify, Response, json
from requests import get
import numpy as np
import requests
import requests
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
app = Flask(__name__)
with open('model/model.pkl', 'rb') as file:
    loaded_model22 = pickle.load(file)

# Загрузка новых данных из CSV-файла

def calculate_slope(x, y):
    mx = x - x.mean()
    my = y - y.mean()
    return sum(mx * my) / sum(mx**2)
def get_params(x, y):
    a = calculate_slope(x, y)
    b = y.mean() - a * x.mean()
    return a, b
menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]
loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
loaded_model_2=pickle.load(open('model/2laba.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Абрамовым Александром Альбертовичем", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)

@app.route("/p_lab2",methods=['POST','GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)

    if request.method == 'POST':
        X_test=np.array([[float(request.form['list11']),
                          float(request.form['list21']),
                          float(request.form['list31']),
                          float(request.form['list41']),
                          float(request.form['list51']),
                          float(request.form['list61']),
                          float(request.form['list71']),
                          float(request.form['list81']),
                          float(request.form['list91']),
                          float(request.form['list101']),]])
        y_pred_loaded = loaded_model_2.predict(X_test)
        return render_template('lab2.html', title="логистическая регрессия", menu=menu,
                               class_model="Это: " + str(y_pred_loaded))


    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3", methods=["GET", "POST"])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu)
    if request.method == 'POST':
        list1 = int(request.form['list1'])
        list2 = int(request.form['list2'])
        predictions = loaded_model22.predict([[list1,list2]])
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + predictions)
@app.route('/api', methods=['GET'])
def get_sort():
    if request.method == 'GET':
        request_data = request.get_json()
        if 'age' not in request_data or 'gender' not in request_data:
            return jsonify(error="Необходимые параметры отсутствуют в запросе"), 400
        age = int(request_data['age'])
        gender = int(request_data['gender'])

        X_new = np.array([[age]])
        y_new = np.array([[gender]])

        if X_new.shape != (1, 1) or y_new.shape != (1, 1):
            return jsonify(error="Неверный формат входных данных"), 400

        pred = loaded_model22.predict([[X_new, y_new]])

        return jsonify(books=pred[0])
    else:
        return jsonify(error="Метод запроса не поддерживается"), 405


#http://localhost:5000/api?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2

@app.route('/api_v2', methods=['GET'])
def get_sort_v2():
    list1 = request.args.get('list1')
    list2 = request.args.get('list2')

    if list1 is None or list2 is None:
        return jsonify(error="Параметры 'list1' и 'list2' должны быть предоставлены"), 400

    X_new = np.array([[int(list1), int(list2)]])
    pred = loaded_model22.predict(X_new)
    gender = pred[0]

    return jsonify({"gender": gender})
#http://localhost:5000/api_v2?list1=101&list2=1


if __name__ == "__main__":
    app.run(debug=True)
