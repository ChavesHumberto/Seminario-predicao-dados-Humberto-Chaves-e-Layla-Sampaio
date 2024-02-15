# Seminário Predição dados - Humberto Chaves (218060062) e Layla Sampaio (119060009)

# Usando VAR (vetor autorregressivo) para prever dados sobre economia


## Preparandinho o terreno


```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
```

    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
data = sm.datasets.macrodata.load_pandas().data #cria o dataset puxando da API
data.head(6) # esse dataset se refere a economia de um dado país, no intervalo de tempo de 1959 a 2009 dividido por bimestre
```





  <div id="df-f0a1444f-bf0b-4ea3-a438-016710c7bdf3">
    <div class="colab-df-container">
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
      <th>year</th>
      <th>quarter</th>
      <th>realgdp</th>
      <th>realcons</th>
      <th>realinv</th>
      <th>realgovt</th>
      <th>realdpi</th>
      <th>cpi</th>
      <th>m1</th>
      <th>tbilrate</th>
      <th>unemp</th>
      <th>pop</th>
      <th>infl</th>
      <th>realint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1959.0</td>
      <td>1.0</td>
      <td>2710.349</td>
      <td>1707.4</td>
      <td>286.898</td>
      <td>470.045</td>
      <td>1886.9</td>
      <td>28.98</td>
      <td>139.7</td>
      <td>2.82</td>
      <td>5.8</td>
      <td>177.146</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1959.0</td>
      <td>2.0</td>
      <td>2778.801</td>
      <td>1733.7</td>
      <td>310.859</td>
      <td>481.301</td>
      <td>1919.7</td>
      <td>29.15</td>
      <td>141.7</td>
      <td>3.08</td>
      <td>5.1</td>
      <td>177.830</td>
      <td>2.34</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1959.0</td>
      <td>3.0</td>
      <td>2775.488</td>
      <td>1751.8</td>
      <td>289.226</td>
      <td>491.260</td>
      <td>1916.4</td>
      <td>29.35</td>
      <td>140.5</td>
      <td>3.82</td>
      <td>5.3</td>
      <td>178.657</td>
      <td>2.74</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1959.0</td>
      <td>4.0</td>
      <td>2785.204</td>
      <td>1753.7</td>
      <td>299.356</td>
      <td>484.052</td>
      <td>1931.3</td>
      <td>29.37</td>
      <td>140.0</td>
      <td>4.33</td>
      <td>5.6</td>
      <td>179.386</td>
      <td>0.27</td>
      <td>4.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1960.0</td>
      <td>1.0</td>
      <td>2847.699</td>
      <td>1770.5</td>
      <td>331.722</td>
      <td>462.199</td>
      <td>1955.5</td>
      <td>29.54</td>
      <td>139.6</td>
      <td>3.50</td>
      <td>5.2</td>
      <td>180.007</td>
      <td>2.31</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1960.0</td>
      <td>2.0</td>
      <td>2834.390</td>
      <td>1792.9</td>
      <td>298.152</td>
      <td>460.400</td>
      <td>1966.1</td>
      <td>29.55</td>
      <td>140.2</td>
      <td>2.68</td>
      <td>5.2</td>
      <td>180.671</td>
      <td>0.14</td>
      <td>2.55</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f0a1444f-bf0b-4ea3-a438-016710c7bdf3')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f0a1444f-bf0b-4ea3-a438-016710c7bdf3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f0a1444f-bf0b-4ea3-a438-016710c7bdf3');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Descrição do problema

O modelo VAR é um processo ESTOCASTICO que representa um grupo de variáveis dependentes de TEMPO como uma FUNÇÂO LINEAR dos valores passados delas próprias e dos valores passados de todas as outras variáveis do grupo.


Por exemplo, podemos considerar uma análise de série temporal bivariada que descreve uma relação entre temperatura de hora em hora e a velocidade do vento em função de valores passados ​​[2]:

temp(t) = a1 + w11* temp(t-1) + w12* vento(t-1) + e1(t-1)

vento(t) = a2 + w21* temp(t-1) + w22* vento(t-1) +e2(t-1)

onde a1 e a2 são constantes; w11, w12, w21 e w22 são os coeficientes; e1 e e2 são os termos de erro

Em nosso problema, usamos o VAR para fazer a previsão do PROODUTO INTERNO BRUTO  e da RENDA PESSOAL DESCARTAVEL num período de tempo de 10 DATAS nesse caso BIMESTRES.


```python

```

## Apresentação da metodologia

### Tratamento de dados

Nossos dados continham uma série de dados de séries temporais. A solução do Machine Hack escolheu apenas duas variáveis dependentes de tempo (realgdp e realdpi) para fazer o experimento e usou a coluna "year" como índice dos dados

Antes de aplicar o modelo VAR, precisamos verificar se nossas variaveis eram estacionárias (apresentavam média e variânca constantes ao longo do tempo).

Para isso, usamos o teste Augmented Dickey-Fuller (ADF) para encontrar a estacionariedade da série usando os critérios AIC. O teste ADF é um teste de raiz unitária em séries temporais. A estatística ADF, usada no teste, é um número negativo, e quanto mais negativo, mais indicativo o teste se torna de rejeitar a hipótese nula de que existe raiz unitária na série.

Como ambas as séries não são estacionárias, realizamos a diferenciação e posteriormente verificamos a estacionaridade.

Os dados passam a ser estacionários.


```python
data1 = data[["realgdp", 'realdpi']]
data1.index = data["year"]
data1.plot(figsize = (8,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f377dd5f490>




    
![png](README_files/README_12_1.png)
    



```python
adfuller_test = adfuller(data1['realgdp'], autolag= "AIC")
print("ADF test statistic: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1]))
```

    ADF test statistic: 1.7504627967647186
    p-value: 0.9982455372335032
    


```python
adfuller_test = adfuller(data1['realdpi'], autolag= "AIC")
print("ADF test statistic: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1]))
```

    ADF test statistic: 2.9860253519546855
    p-value: 1.0
    


```python
data_d = data1.diff().dropna()
```


```python
adfuller_test = adfuller(data_d['realgdp'], autolag= "AIC")
print("ADF test statistic: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1]))
```

    ADF test statistic: -6.305695561658105
    p-value: 3.327882187668224e-08
    


```python
adfuller_test = adfuller(data_d['realdpi'], autolag= "AIC")
print("ADF test statistic: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1]))
```

    ADF test statistic: -8.864893340673007
    p-value: 1.4493606159108096e-14
    


```python
data_d.plot(figsize=(8,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3773f57b10>




    
![png](README_files/README_18_1.png)
    


### Sobre o modelo

No processo de modelagem do VAR, o site optou por empregar o Critério de Informação Akaike (AIC) como critério de seleção do modelo para realizar a identificação ótima do modelo. Em termos simples, selecionamos a ordem (p) do VAR com base na melhor pontuação do AIC. O AIC, em geral, penaliza os modelos por serem muito complexos, embora os modelos complexos possam ter um desempenho ligeiramente melhor em algum outro critério de seleção de modelos. Assim, esperamos um ponto de inflexão na busca da ordem (p), significando que, a pontuação do AIC deve diminuir à medida que a ordem (p) aumenta até uma certa ordem e então a pontuação começa a aumentar. Para isso, realizamos grid-search para investigar a ordem ótima (p).

Realizamos a divisão de teste de treinamento dos dados e mantemos as últimas 10 datas como dados de teste. Treinamos o modelo VAR com os dados de treinamento, que foram os separados 10 ultimos bimestres do dataset.

A partir do gráfico, a pontuação de AIC mais baixa é alcançada na ordem de 2 e, em seguida, as pontuações de AIC mostram uma tendência crescente à medida que a ordem p aumenta. Assim, selecionamos o 2 como a ordem ótima do modelo VAR. Consequentemente, ajustamos a ordem 2 ao modelo de previsão.

Ao executar um teste de hipótese, você usa a estatística T com um valor p . O valor-p informa quais são as chances de que seus resultados possam ter acontecido por acaso.

Como não conseguimos compreender muito bem os dados apresentados no sumário, criamos algumas hipoteses:

1) Nos nossos testes, a t-stat pode ser considerada baixa, por isso, consideramos nossa previsão boa.

2) a coluna "prob" pode estar relacionada a probabilidade de erro, e por termos numeros baixos aqui também, imaginamos que nossa previsão tenha se aproximado do real resultado.


```python
train = data_d.iloc[:-10,:]
test = data_d.iloc[-10:,:]
```


```python
forecasting_model = VAR(train)
results_aic = []
for p in range(1,10):
  results = forecasting_model.fit(p)
  results_aic.append(results.aic)
```

    /usr/local/lib/python3.7/dist-packages/statsmodels/tsa/base/tsa_model.py:215: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    


```python
plt.plot(list(np.arange(1,10,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()
```


    
![png](README_files/README_24_0.png)
    



```python
results = forecasting_model.fit(2)
results.summary()
```




      Summary of Regression Results   
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Wed, 26, Jan, 2022
    Time:                     19:25:54
    --------------------------------------------------------------------
    No. of Equations:         2.00000    BIC:                    15.5043
    Nobs:                     190.000    HQIC:                   15.4026
    Log likelihood:          -1985.87    FPE:                4.56270e+06
    AIC:                      15.3334    Det(Omega_mle):     4.33171e+06
    --------------------------------------------------------------------
    Results for equation realgdp
    =============================================================================
                    coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------
    const             23.807343         6.111430            3.896           0.000
    L1.realgdp         0.176227         0.078131            2.256           0.024
    L1.realdpi         0.213713         0.085309            2.505           0.012
    L2.realgdp         0.211259         0.075926            2.782           0.005
    L2.realdpi         0.018103         0.087131            0.208           0.835
    =============================================================================
    
    Results for equation realdpi
    =============================================================================
                    coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------
    const             29.557677         5.688065            5.196           0.000
    L1.realgdp         0.246371         0.072718            3.388           0.001
    L1.realdpi        -0.182692         0.079399           -2.301           0.021
    L2.realgdp         0.048001         0.070667            0.679           0.497
    L2.realdpi         0.091316         0.081095            1.126           0.260
    =============================================================================
    
    Correlation matrix of residuals
                realgdp   realdpi
    realgdp    1.000000  0.386669
    realdpi    0.386669  1.000000
    
    



## Discussão dos resultados e conclusões finais




Usamos 2 como a ordem ótima no ajuste do modelo VAR. Assim, tomamos as 2 etapas finais nos dados de treinamento para prever a próxima etapa imediata (ou seja, a primeira data dos dados de teste).

Depois de ajustar o modelo, prevemos para os dados de teste em que as últimas 2 datas de dados de treinamento foram definidas como valores defasados ​​e as steps definidas como 10 datas conforme queremos prever para os próximos 10 dias.

O realdpi original e o realdpi previsto mostram um padrão semelhante fora das datas previstas. Para realgdp: a primeira metade dos valores previstos apresenta um padrão semelhante aos valores originais, por outro lado, a última metade dos valores previstos não segue um padrão semelhante.



```python
lagged_values = train.values[-2:]#Usamos 2 como a ordem ótima no ajuste do modelo VAR.
                                # Assim, tomamos as 2 ultimas datas dos dados de treinamento para prever a próxima etapa imediata
forecast = pd.DataFrame(results.forecast(y= lagged_values, steps=10),# prevemos para os dados de teste em que as últimas 2 datas de DADOS DE TREINAMENTO
                        index = test.index, columns= ['realgdp_1d', 'realdpi_1d'])#foram definidas como valores defasados ​​e as steps definidas como 10 datas conforme queremos prever para os próximos 10 dias.
forecast
```





  <div id="df-0fff4149-3492-4741-b142-64ae40415eb1">
    <div class="colab-df-container">
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
      <th>realgdp_1d</th>
      <th>realdpi_1d</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007.0</th>
      <td>61.872982</td>
      <td>47.739232</td>
    </tr>
    <tr>
      <th>2007.0</th>
      <td>53.948996</td>
      <td>41.742951</td>
    </tr>
    <tr>
      <th>2007.0</th>
      <td>56.171082</td>
      <td>42.552316</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.953081</td>
      <td>42.023999</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>55.109616</td>
      <td>42.001007</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.865410</td>
      <td>41.937065</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.841362</td>
      <td>41.893996</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.775171</td>
      <td>41.878378</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.754309</td>
      <td>41.859837</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.732404</td>
      <td>41.853481</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0fff4149-3492-4741-b142-64ae40415eb1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0fff4149-3492-4741-b142-64ae40415eb1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0fff4149-3492-4741-b142-64ae40415eb1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
forecast.plot(figsize=(8,5))

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3771ec5d50>




    
![png](README_files/README_30_1.png)
    



```python
forecast["realgdp_forecasted"] = data1["realgdp"].iloc[-10-1] + forecast['realgdp_1d'].cumsum() #Derivação da previsão
forecast["realdpi_forecasted"] = data1["realdpi"].iloc[-10-1] + forecast['realdpi_1d'].cumsum() #que soma acumulativa dos valores dos 10 dias anteriores anteriores
```


```python
forecast
```





  <div id="df-f837c268-f3b0-4ef2-94be-e2c8e01e075e">
    <div class="colab-df-container">
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
      <th>realgdp_1d</th>
      <th>realdpi_1d</th>
      <th>realgdp_forecasted</th>
      <th>realdpi_forecasted</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007.0</th>
      <td>61.872982</td>
      <td>47.739232</td>
      <td>13161.773982</td>
      <td>9877.939232</td>
    </tr>
    <tr>
      <th>2007.0</th>
      <td>53.948996</td>
      <td>41.742951</td>
      <td>13215.722978</td>
      <td>9919.682183</td>
    </tr>
    <tr>
      <th>2007.0</th>
      <td>56.171082</td>
      <td>42.552316</td>
      <td>13271.894060</td>
      <td>9962.234500</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.953081</td>
      <td>42.023999</td>
      <td>13326.847141</td>
      <td>10004.258499</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>55.109616</td>
      <td>42.001007</td>
      <td>13381.956757</td>
      <td>10046.259506</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.865410</td>
      <td>41.937065</td>
      <td>13436.822166</td>
      <td>10088.196571</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>54.841362</td>
      <td>41.893996</td>
      <td>13491.663528</td>
      <td>10130.090566</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.775171</td>
      <td>41.878378</td>
      <td>13546.438699</td>
      <td>10171.968945</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.754309</td>
      <td>41.859837</td>
      <td>13601.193008</td>
      <td>10213.828781</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>54.732404</td>
      <td>41.853481</td>
      <td>13655.925412</td>
      <td>10255.682262</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f837c268-f3b0-4ef2-94be-e2c8e01e075e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f837c268-f3b0-4ef2-94be-e2c8e01e075e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f837c268-f3b0-4ef2-94be-e2c8e01e075e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
forecast.plot(figsize=(8,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3771dea150>




    
![png](README_files/README_33_1.png)
    



```python
data1_10 = data1.iloc[-10:,:]
forecasted = forecast[["realgdp_forecasted", "realdpi_forecasted"]]
prev = pd.concat([data1_10, forecasted])
prev.plot(figsize=(8,5)) #PROJEÇÃO CORRETA(modelo da previsao) COM PROPORÇÃO ERRADA
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f37719bd410>




    
![png](README_files/README_34_1.png)
    

