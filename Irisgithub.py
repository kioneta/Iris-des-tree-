from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.model_selection import cross_val_score
from IPython.display import HTML
style = "<style>svg{width: 30% !important; height: 60% !important;} </style>"
HTML(style)


df_train = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv', index_col=0)
df_test = pd.read_csv('https://stepik.org/media/attachments/course/4852/test_iris.csv', index_col=0)
df_train.head(10)
df_train.isnull().sum() #смотрим где много пропущено значений
X_train = df_train.iloc[:,:-1]
y_train = df_train.species
my_awesome_tree = tree.DecisionTreeClassifier(criterion='entropy')
X_test = df_test.iloc[:,:-1]
y_test = df_test.species
#X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
# Разбиваем выборку на тренировочную часть и тестовую
X_train.shape
X_test.shape
max_depth_values = range(1,100)
scores_data = pd.DataFrame()
cross_val_score(my_awesome_tree,X_train, y_train, cv=5).mean()
#разбиваем все точки на 5 групп и считаем их как тест и трейн
for max_depth in max_depth_values:
    my_awesome_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=max_depth)
    my_awesome_tree.fit(X_train, y_train)
    train_score = my_awesome_tree.score(X_train, y_train)
    test_score = my_awesome_tree.score(X_test, y_test)

    mean_cross_val_score = cross_val_score(my_awesome_tree, X_train, y_train, cv=5).mean()
    # записываем среднее значение и при кросс валидации
    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score],
                                    'cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)
# Прогоняем углубление дерева решения и для каждого варианта записываем дф
scores_data.head()
0]
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'],
                          var_name='set_type', value_name='score')
#преобразовываем в нужный формат данные
scores_data_long.query('set_type == "cross_val_score"').head(20)
plt.rcParams['figure.figsize']=(10,10)
sns.lineplot(x = 'max_depth',y  = 'score', hue='set_type', data = scores_data_long)
