#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados




import pandas as pd 
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib


# ### Consolidar Base de Dados




meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

# abril2018_df = pd.read_csv(r'dataset\abril2018.csv')
# display(abril2018_df)

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = pd.concat([base_airbnb, df])
    
display(base_airbnb)    


# - Como temos muitas colunas, nosso modelo pode acabar ficando muito lento.
# - Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão, por isso,
# vamos exluir algumas colunas da nossa base
# - Tipos de colunas que vamos excluir:
#     1. ID, links e infromações não relevantes para o modelo
#     2. Colunas repetidas ou extremamente parecida com outra que dão a mesa informação para o modelo, exemplo: Data X Ano/Mês
#     3. Colunas com texto livre, não rodaremos nenhuma análise com palavras ou algo do tipo
#     4. Colunas em que todos ou quase todos os valores são iguais 
# 
# - Para isso vamos criar um arquivo em excel com os mil primeiros registros e fazer uma análise qualitativa




print(list(base_airbnb.columns))
#base_airbnb.head(1000).to_csv('Primeiros Registros.csv3', sep = ';')


# ### Depois da análise qualitativa das colunas, levando em conta os critérios explicados acima, ficamos com as seguintes colunas: 




colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','street','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]

display(base_airbnb)


# ### Tratar Valores Faltando
# 
# - Visualizando os dados, percebemos que existe uma grande disparidade de dados faltantes. As colunas com mais de 300.000 valores Nan foram excluídas da análise
# - Para as outras colunas como temos muitos dados (mais de 900.000 linhas) vamos excluir as linhas que contém dados Nan




for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300_000:
        base_airbnb = base_airbnb.drop(coluna, axis =1)

print(base_airbnb.isnull().sum())





base_airbnb = base_airbnb.dropna()
base_airbnb = base_airbnb.drop('street', axis=1)
print(base_airbnb.isnull().sum())
print(base_airbnb.shape)


# ### Verificar Tipos de Dados em cada coluna




print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])


# - Como preço e extra_people estão sendo reconhecida como objeto ao (invés de ser um float) temos que mudar o tipo de variável da coluna 




#price 


base_airbnb['price'] = base_airbnb['price'].str.replace('$', '', regex = True)
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '', regex=True)
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy = False)

#extra_people

base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '', regex=True)
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '', regex=True)
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy = False)

print(base_airbnb.dtypes)








# ### Análise Exploratória e Tratar Outliers

# 
# - Vamos basicamente olhar feature por feature para:
#     1. Ver a correlação entre as features e decidir se manteremos todas as features que temos.
#     2. Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
#     
# - Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.
# 
# - Depois vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)
# 
# - Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.
# 
# MAS CUIDADO: não saia excluindo direto outliers, pense exatamente no que você está fazendo. Se não tem um motivo claro para remover o outlier, talvez não seja necessário e pode ser prejudicial para a generalização. Então tem que ter uma balança ai. Claro que você sempre pode testar e ver qual dá o melhor resultado, mas fazer isso para todas as features vai dar muito trabalho.
# 
# Ex de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listings_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do seu modelo. Pense sempre no seu objetivo

# ### Encoding




#print(base_airbnb.corr(numeric_only = True))

plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(numeric_only = True), annot=True, cmap='Greens')


# ### Definição de funções par Análise de Outliers 
# 
# Vamos definir algumas funções para ajudar na análise de outliers das colunas 




def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]  
    return df, linhas_removidas
    
    
    




print(limites(base_airbnb['price']))





def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x = coluna, ax = ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x = coluna, ax = ax2)



def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.histplot(data = base_airbnb, x = coluna, element = 'bars')

def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x =coluna.value_counts().index , y= coluna.value_counts())
    ax.set_xlim(limites(coluna))
    
    




diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas de imóveis de altíssimo luxo, que não é nosso objetivo principal. Por isso podemos excluir esses outliers.


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(f'{linhas_removidas} linhas removidas') 


print(histograma(base_airbnb['price']))
print(base_airbnb.shape)


# ### Extra People




diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])




histograma(base_airbnb['extra_people'])





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{linhas_removidas} linhas removidas')


# ### host_listings_count    




diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# Podemos excluir os outliers porque para o objetivo do nosso projeto porque hosts com mais de 6 imóveis no airbnb não atinge o público alvo do objetivo do projeto (imagino que sejam imobiliárias ou profissionais que gerenciam imóveis no airbnb)




base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas') 


# ### accommodates 



diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{linhas_removidas} linhas removidas') 


# ### bathrooms




diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())
           





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{linhas_removidas} linhas removidas')


# ### bedrooms



diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])






base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{linhas_removidas} linhas removidas')


# ### beds 




diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(f'{linhas_removidas} linhas removidas')


# ### guests_included  



#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barra(base_airbnb['guests_included'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Vamos remover essa feature da análise. Parece que os usuários do airbnb usam muito o valor padrão do airbnb como 1 guest included. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço, por isso me parece melhor excluir a coluna da análise.




base_airbnb = base_airbnb.drop('guests_included', axis=1)
print(base_airbnb.shape)


# ### minimum_nights  




diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas')


# ### maximum_nights



diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])





base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
print(base_airbnb.shape)


# ### number_of_reviews




diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])





base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
print(base_airbnb.shape)


# Tratamento de Colunas de Valores de Preço

# ### property_type




print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['property_type'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)




tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_others_agrupadas = []

for tipo_casa in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo_casa] < 2000:
        colunas_others_agrupadas.append(tipo_casa)
    
print(colunas_others_agrupadas)

for tipo in colunas_others_agrupadas:
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['property_type'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)


# ### room_type




print(base_airbnb['room_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['room_type'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)


# ### bed_type




print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['bed_type'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)



#agrupando categorias de bed_type

tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupadas1 = []

for tipo2 in tabela_bed.index:
    if tabela_bed[tipo2] < 10000:
        colunas_agrupadas1.append(tipo2)
    
print(colunas_agrupadas1)

for tipo3 in colunas_agrupadas1:
    base_airbnb.loc[base_airbnb['bed_type'] == tipo3, 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['bed_type'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)




# ### cancellation_policy




print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['cancellation_policy'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)


# agrupando categorias de cancellation_policy

tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupadas = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupadas.append(tipo)
    
print(colunas_agrupadas)

for tipo1 in colunas_agrupadas:
    base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo1, 'cancellation_policy'] = 'Strict'

print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico_property_type = sns.countplot(x=base_airbnb['cancellation_policy'])
grafico_property_type.tick_params(axis = 'x', rotation = 90)


# ### amenities
# Como temos uma diversidade muito grande de amenities e às vezes, as mesmas podem ser escritas de forma diferente vamos avaliar a quantidade de amenities como parâmetro para o nosso modelo.  




print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))

base_airbnb['numero_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)




base_airbnb = base_airbnb.drop('amenities', axis=1)
print(base_airbnb.shape)





diagrama_caixa(base_airbnb['numero_amenities'])
grafico_barra(base_airbnb['numero_amenities'])





base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'numero_amenities')
print(f'{linhas_removidas} linhas removidas')


# ### Visualização de mapa das propriedades 




amostra = base_airbnb.sample(n=50000)

centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
                        center = centro_mapa, zoom = 10,
                        mapbox_style='stamen-terrain')
mapa.show()


# ## Enconding

# Precisamos ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true e false, etc)
# - Features de valores True or False, vamos substituir True por 1 e False por 0
# - Features de categoria (features em que os valores da coluna são texto) vamos utilizar o método de encoding de variáveis dummies




colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod =  base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]== 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]== 'f', coluna] = 0

print(base_airbnb_cod.iloc[0])





colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod =pd.get_dummies(data = base_airbnb_cod, columns = colunas_categorias)

display(base_airbnb_cod.head())


# ### Modelo de Previsão

#  - Métricas de Avaliação




def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    rsme = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{rsme:.2f}'


# - Escolhas dos modelos a serem testados
#   1. Random Forest
#   2. Linear Regression
#   3. Extra Tree
# 
# 
# 




modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)


# - Separar os dados em treino e teste + Treino do Modelo



x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state = 10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(x_treino, y_treino)

    #testar
    previsao = modelo.predict(x_teste)
    print(avaliar_modelo(nome_modelo, y_teste, previsao))
    
    


# ### Análise do Melhor Modelo

# - Modelo Escolhido como Melhor Modelo: ExtraTrees
# 
#   Esse foi o modelo com o maior valor de R² e menor RSME (valor de erro quadrático médio) e como não tivemos uma notória diferença na velocidade de treino e de previsão desse modelo em comparação com o modelo de RandomForest (que teve resultados aproximados de R² e de RSME), vamos escolher o modelo ExtraTrees
# 
# - O modelo de Regressão Linear não obteve um resultado satisfatório, com valores de R² e RSME muito abaixo em comparação aos outros modelos.
# 
# - Resultados das Métricas de Avaliações no modelo vencendor:<br>
# - Modelo RandomForest:<br>
# R²:97.23%<br>
# RSME:44.12<br>
# - Modelo LinearRegression:<br>
# R²:32.70%<br>
# RSME:217.54<br>
# - Modelo ExtraTrees:<br>
# R²:97.50%<br>
# RSME:41.97<br>
# 

# ### Ajustes e Melhorias no Melhor Modelo




print(modelo_et.feature_importances_)
print(x_treino.columns)
importancia_features = pd.DataFrame(modelo_et.feature_importances_, x_treino.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
importancia_features.rename(columns = {0: 'Relação'}, inplace=True)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index , y=importancia_features['Relação'])
ax.tick_params(axis='x', rotation=90) 


# ### Ajustes Finais do Modelo
# - is_business_travel não parece ter muito impacto no nosso modelo. Por isso, para chegar em um modelo mais simples vamos excluir essa feature e testar o modelo sem ela.




base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis = 1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state = 10)

modelo_et.fit(x_treino, y_treino)

previsao = modelo.predict(x_teste)

print(avaliar_modelo('ExtraTrees', y_teste, previsao))
    






base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))





print(previsao)


# # Deploy do Projeto
# 
# - Passo 1 -> Criar arquivo do Modelo (joblib)<br>
# - Passo 2 -> Escolher a forma de deploy:
#     - Arquivo Executável + Tkinter
#     - Deploy em Microsite (Flask)
#     - Deploy apenas para uso direto Streamlit
# - Passo 3 -> Outro arquivo Python (pode ser Jupyter ou PyCharm)
# - Passo 4 -> Importar streamlit e criar código
# - Passo 5 -> Atribuir ao botão o carregamento do modelo
# - Passo 6 -> Deploy feito




x['price'] = y
x.to_csv('dados_tratados.csv')




joblib.dump(modelo_et, 'modelo.joblib')

