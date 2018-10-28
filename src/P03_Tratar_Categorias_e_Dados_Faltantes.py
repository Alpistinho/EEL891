# -*- coding: utf-8 -*-
#==============================================================================
#  Tratamento de Dados Faltantes e de Atributos Categ�ricos
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_csv('../data/D03_Categorias_e_Dados_Faltantes.csv')

#------------------------------------------------------------------------------
#  Criar os arrays num�ricos correspondentes aos atributos e ao alvo
#  (dados vazios ou n�o-convers�veis para n�mero s�o armazenados como 'NaN')
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#------------------------------------------------------------------------------
#  Tratar os dados faltantes, preenchendo-os com valores estimados
#------------------------------------------------------------------------------

from sklearn.preprocessing import Imputer

# instanciar um Imputer para substituir todas as ocorr�ncias de 'NaN'
# pelo valor m�dio (ou pela mediana, ou pelo valor mais frequente,
# conforme par�metro "strategy") da linha ou da coluna (conforme
# par�metro "axis")

imputer = Imputer(
        missing_values = 'NaN',  # lista de valores a serem substituidos
        strategy = 'mean',       # pode ser tamb�m 'median' ou 'most_frequent'
        axis = 0                 # 0 para coluna, 1 para linha
)

# o m�todo "fit" ajusta os par�metros internos do Imputer, 
# conforme estrat�gia escolhida
 
imputer = imputer.fit(X[:, 1:3])

# o m�todo transform preenche os dados faltantes com os valores 
# determinados pela estrat�gia escolhida

Xold = X.copy()

X[:, 1:3] = imputer.transform(X[:, 1:3])

#------------------------------------------------------------------------------
#  Codificar o atributo categ�rico da coluna 0 (pa�s)
#------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# codifica os pa�ses da coluna 0 em r�tulos num�ricos 0, 1, 2, etc.

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Transforma a coluna 0 em um conjunto de colunas con conte�do bin�rio
# (uma coluna para cada valor distinto)

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X)
X = X.toarray()

#------------------------------------------------------------------------------
#  Codificar o alvo ('yes' ou 'no' - comprou ou n�o comprou)
#------------------------------------------------------------------------------

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


