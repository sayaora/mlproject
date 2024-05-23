from sklearn import  datasets
from sklearn.linear_model import LogisticRegression
x,y=datasets.make_classification(
    n_samples=20,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2
)
logreg=LogisticRegression()
logreg.fit(x,y)
pred=logreg.predict(x)