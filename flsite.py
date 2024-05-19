import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

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


@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
