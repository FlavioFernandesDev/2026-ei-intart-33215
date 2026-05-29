## Exploracao inicial

Depois de uma primeira abordagem ao dataset reparei que este problema e binario: o modelo tem de distinguir casos malignos de casos benignos. O dataset tem 569 amostras e 30 variaveis numericas extraidas de imagens de biopsias mamarias. Isto quer dizer que os dados ja nao sao a imagem original, mas sim uma representacao numerica criada a partir dela.

A distribuicao das classes nao esta perfeitamente equilibrada: existem 212 casos malignos e 357 casos benignos. Usei um split estratificado para manter esta proporcao no treino e no teste.

O modelo baseline foi uma LogisticRegression com StandardScaler. Usei o StandardScaler porque as variaveis estao em escalas diferentes, e isso influencia modelos lineares. Com random_state=42, o modelo obteve uma accuracy de cerca de 0.9825 no conjunto de teste.

## Reflexao

Quando uma imagem e reduzida a um conjunto de numeros perde-se parte da informacao visual original. As 30 caracteristicas resumem medidas como raio, textura, area e concavidade, mas deixam de mostrar a forma completa da biopsia, a distribuicao espacial dos tecidos e outros detalhes visuais que podem ser importantes para um diagnostico.

Na imagem original pode existir informacao que estas caracteristicas nao capturam bem, por exemplo padroes locais, irregularidades pequenas, zonas suspeitas muito especificas ou relacoes espaciais entre diferentes partes da imagem. Ao trabalhar apenas com numeros, ficamos dependentes das caracteristicas que alguem decidiu extrair previamente.

O erro que considero mais grave na confusion matrix e prever um caso maligno como benigno. Neste dataset isso corresponde a uma linha real "malignant" ser classificada na coluna "benign". Este falso benigno e perigoso porque pode atrasar o diagnostico e o tratamento de uma pessoa que realmente tem cancro. Um falso maligno tambem e um erro importante, porque pode causar ansiedade e exames desnecessarios, mas em contexto clinico deixar passar um caso maligno parece-me mais grave.
