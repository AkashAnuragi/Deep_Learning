
# importing libraries
import pandas as pd
from utils import text_clean



# loading Dataset
print("Data Loading..")
neg = pd.read_csv('dataset/negative.csv')
pos = pd.read_csv('dataset/positive.csv')

# Concatinating Dataset
print("Data Merging..")
df = pd.concat([neg, pos])
df = df.drop(columns=["count"])

# Text Cleaning Process / Tokens / Stopwords / Lemmatization
print("Data Cleaning..")
df['text'] = df['text'].map(text_clean)


# Divide Data into X(dataset) and y(labels)
X = df['text']
y = df['status']
y = pd.get_dummies(y)
y = y.iloc[:,1].astype('int8')


# Create VectorDataset
print("Creating Vector Dataset..")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)
print("Vector Created!")
print(X.shape)


# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
print("Model Training Completed!")


# Model Save
import joblib
joblib.dump(model , "models/model.pkl")
joblib.dump(tfidf , 'models/vectorizer.pkl')
print("Model Saved Successfully!")



# Evaluation
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print("Model Evaluation")
print("Accuracy Score :",acc)
print("Confusion Matrix :\n",cm)
print("Classification Report :\n",cr)

file = open("model_training_report.txt",'w')
file.write("\n\n\t\tLOGISTIC REGRESSION\n\n")
file.write("Accuracy Score : "+str(acc))
file.write("\nConfusion Matrix\n"+str(cm))
file.write("\nClassification Report\n"+str(cr))
file.close()
print("Report Generated Successfully!")

