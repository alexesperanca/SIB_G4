# Notebook Monica

# Table of contents

1. [Introduction](#chapter1)
2. [Initial Data Exploration and Preprocessing](#chapter2)
    1. [Importing Packages](#section2_1)
    2. [Importing datasets](#section2_2)
    3. [Exploring the data](#section2_3)
3. [Non-Supervised Analysis](#chapter3)
4. [Machine Learning](#chapter4)
5. [Deep Learning ](#chapter5)
6. [Conclusions](#chapter6)

# Data Analysis using Machine Learning

This project consists in the analysis of dataset, through the use of machine learning algorithms, using Python as a programming language. This Jupyter Notebook is organized into sections, which include the steps of the analysis performed and explain very succinctly the procedures performed and decisions taken during the analysis.

<a class="anchor" id="chapter1"></a>

## 1. Introduction

**Data Selection and Context of this Project** 

For the execution of this project, the dataset of the "Novozymes Enzyme Stability Prediction" Competition on the Kaggle platform was selected. You can view the [Competition and Corresponding Data here](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data).

The main goal of this competition is to predict the thermostability of enzyme variants. The experimentally measured thermostability (melting temperature) data includes natural sequences, as well as engineered sequences with single or multiple mutations upon the natural sequences.

The test dataset contains experimental melting temperatures of over 2,413 single mutation variants of an enzyme (GenBank: KOC15878.1), obtained by Novozymes A/S. The wild-type amino acid sequence is:

    VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK

Para esta competição foram dado vários ficheiros para o dessenvolvimento deste trabalho. Nomeadamente:

- `train.csv`: o conjunto de dados de treino com as seguintes colunas (features):
    - `seq_id`: identificador único de cada variante da enzima
    - `protein_sequence`: sequência de aminoácidos de cada variante da enzima. A estabilidade da proteína (`tm`) é determinada pela sua sequência.
    - `pH`: escala de acidez em que a estabilidade de cada variante da enzima foi medida.
    - `data_source`: fonte onde os dados foram publicados
    - `tm`: feature alvo - estabilidade de cada variante da enzima (valores mais altos correspondem a uma maior estabilidade).
    
- `train_updates_20220929.csv`: ficheiro de correção dos dados de treino, onde algumas linhas têm os valores de `pH` e `tm` trocados, para além de ter identificadas as linhas com valores NaN. Para mais detalhes, é possível verificar a [explicação original aqui](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356251).

- `test.csv`: o conjunto de dados de teste com as colunas `seq_id` e `protein_sequence`, emq ue o objetivo é prever o valor de `tm` para cada variante da enzima.

- `sample_submission.csv`: um ficheiro de amostra de submissão, com a formatação correta em que o `seq_id` corresponde aos mesmos do ficheiro `test.csv`.

- `wildtype_structure_prediction_af2.pdb`: ficheiro que contém informação sobre a estrutura 3D da enzima, prevista por AlphaFold.

<a class="anchor" id="chapter2"></a>

## 2. Initial Data Exploration and Preprocessing

This step corresponds to the following objectives:
- description and characterization of the assigned data according to the existing documentation/literature;
- brief description of the characteristics of the data available from the initial exploratory analysis;
- description of data preparation and pre-processing steps;
- initial exploratory graphs that illustrate the main characteristics of the data.

<a class="anchor" id="section2_1"></a>

### 2.1 Importing Required Packages

**NOTA: não sei se faz sentido, mas poderiamos explicar pelo menos alguns dos packages, porque é que estamos a usa-los.**


```python
# pip install sgt
# pip install propy
# pip install nbconvert
```


```python
#Imports
# from utils.func import swap_ph_tm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.func import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from utils.kmers import KMer_prot
from sklearn.preprocessing import StandardScaler
# import propy
# from propy import PyPro
from sklearn.preprocessing import scale
from sklearn import preprocessing 
from sgt import SGT
```

<a class="anchor" id="section2_2"></a>

### 2.2 Importing datasets


```python
#Train dataframe
train = pd.read_csv("data/train.csv",index_col="seq_id")
#Validation dataframe (test dataset for the competition scoring)
validation = pd.read_csv("data/test.csv",index_col="seq_id")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>protein_sequence</th>
      <th>pH</th>
      <th>data_source</th>
      <th>tm</th>
    </tr>
    <tr>
      <th>seq_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLVG...</td>
      <td>7.0</td>
      <td>doi.org/10.1038/s41592-020-0801-4</td>
      <td>75.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR...</td>
      <td>7.0</td>
      <td>doi.org/10.1038/s41592-020-0801-4</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAAFSTPRATSYRILSSAGSGSTRADAPQVRRLHTTRDLLAKDYYA...</td>
      <td>7.0</td>
      <td>doi.org/10.1038/s41592-020-0801-4</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAASGLRTAIPAQPLRHLLQPAPRPCLRPFGLLSVRAGSARRSGLL...</td>
      <td>7.0</td>
      <td>doi.org/10.1038/s41592-020-0801-4</td>
      <td>47.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAATKSGPRRQSQGASVRTFTPFYFLVEPVDTLSVRGSSVILNCSA...</td>
      <td>7.0</td>
      <td>doi.org/10.1038/s41592-020-0801-4</td>
      <td>49.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
validation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>protein_sequence</th>
      <th>pH</th>
      <th>data_source</th>
    </tr>
    <tr>
      <th>seq_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31390</th>
      <td>VPVNPEPDATSVENVAEKTGSGDSQSDPIKADLEVKGQSALPFDVD...</td>
      <td>8</td>
      <td>Novozymes</td>
    </tr>
    <tr>
      <th>31391</th>
      <td>VPVNPEPDATSVENVAKKTGSGDSQSDPIKADLEVKGQSALPFDVD...</td>
      <td>8</td>
      <td>Novozymes</td>
    </tr>
    <tr>
      <th>31392</th>
      <td>VPVNPEPDATSVENVAKTGSGDSQSDPIKADLEVKGQSALPFDVDC...</td>
      <td>8</td>
      <td>Novozymes</td>
    </tr>
    <tr>
      <th>31393</th>
      <td>VPVNPEPDATSVENVALCTGSGDSQSDPIKADLEVKGQSALPFDVD...</td>
      <td>8</td>
      <td>Novozymes</td>
    </tr>
    <tr>
      <th>31394</th>
      <td>VPVNPEPDATSVENVALFTGSGDSQSDPIKADLEVKGQSALPFDVD...</td>
      <td>8</td>
      <td>Novozymes</td>
    </tr>
  </tbody>
</table>
</div>



<a class="anchor" id="section2_3"></a>

### 2.3 Exploring the data


```python
print(f"Train data is divided in {train.shape[0]} samples and {train.shape[1]} features")
print(f"Validation data is divided in {validation.shape[0]} samples and {validation.shape[1]} features")
print(f"Labels for train: {[labels for labels in train.columns]}")
```

    Train data is divided in 31390 samples and 4 features
    Validation data is divided in 2413 samples and 3 features
    Labels for train: ['protein_sequence', 'pH', 'data_source', 'tm']



```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pH</th>
      <th>tm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>31104.000000</td>
      <td>31390.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.892339</td>
      <td>49.147337</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.612225</td>
      <td>14.010089</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.990000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>42.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>53.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.900000</td>
      <td>130.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
validation.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2413.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



With this information, we can make some observations:

- the `data_source` variable is not nominal, so we can't more information about this feature.
- the `validation` dataset corresponds to ~7% of total samples, while `train` dataset corresponds to ~93%.
- the `validation` dataset corresponds to the test dataset for the competition scoring, so it misses the `tm` feature (the variable to be predicted).
- the `pH` variable on the `train` dataset has a maximum value of 64.9.
- the `pH` variable on the `train` dataset has 31104 samples, while `tm` has 31390 (which is the total number of samples of that dataset).
- the `pH` variable on the `validation` dataset has all the samples with the same value of 8. 

#### Data sources

Para verificar a distribuição das sources na bases de dados, converte-se para uma variável nominal e observa-se a distribuição.


```python
len(pd.unique(train["data_source"]))
```




    325




```python
print(train.data_source)
```

    seq_id
    0        doi.org/10.1038/s41592-020-0801-4
    1        doi.org/10.1038/s41592-020-0801-4
    2        doi.org/10.1038/s41592-020-0801-4
    3        doi.org/10.1038/s41592-020-0801-4
    4        doi.org/10.1038/s41592-020-0801-4
                           ...                
    31385    doi.org/10.1038/s41592-020-0801-4
    31386    doi.org/10.1038/s41592-020-0801-4
    31387    doi.org/10.1038/s41592-020-0801-4
    31388    doi.org/10.1038/s41592-020-0801-4
    31389    doi.org/10.1038/s41592-020-0801-4
    Name: data_source, Length: 31390, dtype: object



```python
sources = {}
count = 1
for n, i in enumerate(train.data_source):
    if i not in sources.keys():
        sources[i] = count
        train.data_source[n] = sources[i]
        count += 1
    else:
        train.data_source[n] = sources[i]

# print(train.data_source)
```

    /tmp/ipykernel_36200/722519737.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train.data_source[n] = sources[i]
    /tmp/ipykernel_36200/722519737.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train.data_source[n] = sources[i]



```python
print(pd.Series(train.data_source).describe())
```

    count     31390
    unique      325
    top           1
    freq      24525
    Name: data_source, dtype: int64



```python
unique, counts = np.unique(train.data_source, return_counts=True)
dicion = dict(zip(unique, counts))

for i,a in dicion.items():
    if a > 300: # 1% do total counts
        print('Data Source: ', i, 'Counts: ', a)
        print('Percentage of total counts: ', round((a*100)/train.data_source.describe()[0], 2), '%')

```

    Data Source:  1 Counts:  24525
    Percentage of total counts:  78.13 %
    Data Source:  2 Counts:  3347
    Percentage of total counts:  10.66 %



```python
plt.hist(train.data_source, bins=325)
plt.show()
```


    
![png](Monica_notebook_files/Monica_notebook_27_0.png)
    


We can observe that 78% of the dataset corresponds to one unique data source.

Although this feature (`data source`) is interesting to explore how the data was obtained, it contains a large number of null values (NA) and is not essential to the main goal of this project (**to predict `tm`**).
So, it was decided to drop this column from the dataset (presented further ahead).

#### Swap pH e tm (to correct data)

Here it is possible to verify that the `pH` variable has errors in the "train" dataset since its maximum is 64.9 (a value impossible to obtain). According to the source of the data, the variable pH and tm have some examples with the values changed. Thus, it is necessary to change these values in the identified sequences (dataset "train_updates").

Here it is possible to verify that the `pH` variable has errors in the `train` dataset, since its maximum is 64.9 (impossible pH value to obtain). According to the [source of the competition](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356251), the variable `pH` and `tm` has some samples with the values swapped. Thus, it is necessary to change these values in the identified sequences (dataset `train_updates`).


```python
update_train = pd.read_csv("data/train_updates_20220929.csv",index_col="seq_id")
train = swap_ph_tm(train,update_train)
```


```python
print(f"Labels: {[labels for labels in train.columns]}")
print(f"Labels: {[labels for labels in validation.columns]}")
```

    Labels: ['protein_sequence', 'pH', 'data_source', 'tm']
    Labels: ['protein_sequence', 'pH', 'data_source']


#### Drop NA Values

As said above, the column `data_source` will be eliminated from the dataset, as well as the rest of the samples with null values.


```python
print("Remove data_source")
train = train.drop(columns="data_source")
validation = validation.drop(columns="data_source")
print(f"Train data is divided in {train.shape[0]} lines and {train.shape[1]} col")
print(f"Validation data is divided in {validation.shape[0]} lines and {validation.shape[1]} col")
print("We want to predict tm values for test data")
```

    Remove data_source
    Train data is divided in 28981 lines and 3 col
    Validation data is divided in 2413 lines and 2 col
    We want to predict tm values for test data



```python
print(train.isnull().sum().sort_values(ascending=False))
print(validation.isnull().sum().sort_values(ascending=False))
```

    pH                  286
    protein_sequence      0
    tm                    0
    dtype: int64
    protein_sequence    0
    pH                  0
    dtype: int64



```python
missing_data = train[train["pH"].isnull()]
# missing_data
```


```python
train= train.drop((missing_data).index)
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>protein_sequence</th>
      <th>pH</th>
      <th>tm</th>
    </tr>
    <tr>
      <th>seq_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLVG...</td>
      <td>7.0</td>
      <td>75.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR...</td>
      <td>7.0</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAAFSTPRATSYRILSSAGSGSTRADAPQVRRLHTTRDLLAKDYYA...</td>
      <td>7.0</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAASGLRTAIPAQPLRHLLQPAPRPCLRPFGLLSVRAGSARRSGLL...</td>
      <td>7.0</td>
      <td>47.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAATKSGPRRQSQGASVRTFTPFYFLVEPVDTLSVRGSSVILNCSA...</td>
      <td>7.0</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31385</th>
      <td>YYMYSGGGSALAAGGGGAGRKGDWNDIDSIKKKDLHHSRGDEKAQG...</td>
      <td>7.0</td>
      <td>51.8</td>
    </tr>
    <tr>
      <th>31386</th>
      <td>YYNDQHRLSSYSVETAMFLSWERAIVKPGAMFKKAVIGFNCNVDLI...</td>
      <td>7.0</td>
      <td>37.2</td>
    </tr>
    <tr>
      <th>31387</th>
      <td>YYQRTLGAELLYKISFGEMPKSAQDSAENCPSGMQFPDTAIAHANV...</td>
      <td>7.0</td>
      <td>64.6</td>
    </tr>
    <tr>
      <th>31388</th>
      <td>YYSFSDNITTVFLSRQAIDDDHSLSLGTISDVVESENGVVAADDAR...</td>
      <td>7.0</td>
      <td>50.7</td>
    </tr>
    <tr>
      <th>31389</th>
      <td>YYVPDEYWQSLEVAHKLTFGYGYLTWEWVQGIRSYVYPLLIAGLYK...</td>
      <td>7.0</td>
      <td>37.6</td>
    </tr>
  </tbody>
</table>
<p>28695 rows × 3 columns</p>
</div>




```python
print(train.isnull().sum().sort_values(ascending=False))
print(validation.isnull().sum().sort_values(ascending=False))
```

    protein_sequence    0
    pH                  0
    tm                  0
    dtype: int64
    protein_sequence    0
    pH                  0
    dtype: int64


#### Exploring Protein Sequence feature


```python
# Tamanho das sequencias de proteina e distribuição
lista = [len(train['protein_sequence'].iat[i]) for i in range(len(train))]
pd.Series(lista).describe()
```




    count    28695.000000
    mean       451.729535
    std        416.889872
    min          5.000000
    25%        210.000000
    50%        352.000000
    75%        537.000000
    max       8798.000000
    dtype: float64




```python
plt.hist(lista, bins=75)
plt.show()

#adicionar titulos aos eixos e grafico
```


    
![png](Monica_notebook_files/Monica_notebook_41_0.png)
    


#### Summary


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pH</th>
      <th>tm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28695.000000</td>
      <td>28695.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.872467</td>
      <td>51.385604</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.793184</td>
      <td>12.076609</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.990000</td>
      <td>25.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>43.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>48.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>54.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
      <td>130.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Train data is divided in {train.shape[0]} samples and {train.shape[1]} features")
print(f"Labels for train: {[labels for labels in train.columns]}")
```

    Train data is divided in 28695 samples and 3 features
    Labels for train: ['protein_sequence', 'pH', 'tm']


Now, other observations can be made:

- the `train` dataset without null values has a total of 28695 samples and only 3 features: `protein_sequence`, `pH` and `tm`.
- the `pH` feature has values between 1.9 and 11 but, the majority of samples have a pH value of 7.
- the `tm` feature has values between 25 and 130, but 50% of samples have a smaller range: 44-55.
- the `protein_sequence` samples have a large range of lenghts (between 5 and 8798), but 50% of samples have lenghts between 210 and 537). The mean lenght of the 28695 sequences is 451.
- to use some machine learning techniques it is needed to transform the feature `protein_sequence` into descriptors (multiple features), such as the frequency of the aminoacids, composition, physicochemical properties of the protein, and many others.


```python

```

<a class="anchor" id="chapter3"></a>

## 3. Non-Supervised Analysis

<a class="anchor" id="chapter4"></a>

## 4. Machine Learning

<a class="anchor" id="chapter5"></a>

## 5. Deep Learning

<a class="anchor" id="chapter6"></a>

## 6. Conclusions
