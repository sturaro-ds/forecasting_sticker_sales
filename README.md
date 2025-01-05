# Series Temporais | Forecasting Sticker Sales

![Kaggle Competition](https://sigmoidal.ai/wp-content/uploads/2022/06/Forecasting-2-1024x626.png)

Este projeto é parte de uma competição do Kaggle com o objetivo de prever as vendas de adesivos em diferentes países. Para este desafio, apliquei uma abordagem robusta de modelagem de séries temporais, utilizando o algoritmo **XGBRegressor** do pacote **xgboost**. O modelo foi otimizado para minimizar o **MAPE** (Mean Absolute Percentage Error), que é a principal métrica de avaliação da competição.

## O que foi feito:

1. **Processo de ELT (Extração, Transformação e Carga) e Engenharia de Atributos:**
   - Realizei um processo completo de ELT para garantir que os dados estivessem no formato adequado para a modelagem.
   - A transformação incluiu a engenharia de atributos de data, extraindo variáveis como ano, mês, dia da semana e dummizaçao das demais variáveis preditoras para melhorar a previsão das vendas.

2. **Algoritmo Utilizado:**
   - O algoritmo escolhido foi o **XGBRegressor**, que é conhecido por sua alta performance em problemas de regressão.
   - Realizei uma busca automatizada para otimizar a combinação de diversos parâmetros e alcançar o melhor **MAPE** possível.

![XGBoost](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*6HDinkTP5BbizoL_xKxToA.jpeg)

## Resultados:

Com o processo de otimização e a modelagem de séries temporais, consegui melhorar a precisão das previsões, resultando em uma solução eficaz para prever as vendas de adesivos em múltiplos países, com MAPE de 8,09%, que indica que o modelo erra em média apenas 8,09% das previsões considerando a base de treino e teste.

### Ferramentas Utilizadas:
- Python
- Pandas
- XGBoost
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusão:

Este projeto proporcionou uma excelente oportunidade de aplicar técnicas de modelagem de séries temporais, além de aprofundar o uso de algoritmos de regressão como o **XGBRegressor**. A experiência com o processo de ELT e a engenharia de atributos foi fundamental para melhorar a precisão das previsões. Foi uma ótima forma de entender como diferentes fatores impactam as previsões e otimizar os resultados para um desafio prático no Kaggle.

[Veja as informações da Competição](https://www.kaggle.com/competitions/playground-series-s5e1)
