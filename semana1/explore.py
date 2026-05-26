from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Carregar o dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")




#Verificar estrutura os dados
print(X.shape)
print(y.shape)
print(data.target_names)
print(X.head())


#Explorar distruicao das classes
print(y.value_counts())
sns.countplot(x=y)
plt.show()

X.describe()


