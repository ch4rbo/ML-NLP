import pandas as pd 
import regex as re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

df_raw = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")

df_interin = df_raw.copy()

df_interin = df_interin.drop_duplicates().reset_index(drop = True)

def comas(text):
       
    return re.sub(',', ' ', text) # Elimina comas del texto

def espacios(text):
    
    return re.sub(r'(\n{2,})','\n', text) # Elimina enters dobles por un solo enter

def minuscula(text):
  
    return text.lower() # Cambia mayusculas a minusculas

def numeros(text):
    
    return re.sub('([\d]+)', ' ', text) # Sustituye los numeros

def caracteres_no_alfanumericos(text):
    
    return re.sub("(\\W)+"," ",text) # Sustituye caracteres raros, no digitos y letras Ej. hola 'pepito' como le va? -> hola pepito como le va

def comillas(text):
    
    return re.sub("'"," ", text) # Sustituye comillas por un espacio
    
def palabras_repetidas(text):
   
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text) # Sustituye palabras repetidas
  
def esp_multiple(text):
    
    return re.sub(' +', ' ',text) # Sustituye los espacios dobles entre palabras

def url(text):
   
    return re.sub(r'(https://www|https://)', '', text) # Remove https

df_interin['url_limpia'] = df_interin['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)

df_interin['is_spam'] = df_interin['is_spam'].apply(lambda x: 1 if x == True else 0)

df = df_interin.copy()

X = df['url_limpia']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state=2207)

vec = CountVectorizer()

X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

clf_SVM = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')

clf_SVM.fit(X_train, y_train)
predictions = clf_SVM.predict(X_test)
print(classification_report(y_test, predictions))

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

best_model = grid.best_estimator_

# Save it for future use

pickle.dump(best_model, open('../models/best_model.pickle', 'wb')) # save the model
# modelo = pickle.load(open('../models/best_model.pickle', 'rb')) # read the model in the future
