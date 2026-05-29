## Exploracao inicial

Depois de uma primeira abordagem ao dataset reparei que este problema é um problema binario, isto é, o modelo tem de distinguir casos malignos de casos benignos (M/B). O dataset tem 569 amostras e 30 variaveis numericas extraidas de imagens de biopsias mamarias. Isto quer dizer que os dados ja nao sao a imagem original, mas sim uma representacao numerica criada a partir dela.

A distribuicao das classes não está perfeitamente equilibrada: existem 212 casos malignos e 357 casos benignos. Usei um split estratificado para manter esta proporcao no treino e no teste.

O modelo baseline foi uma LogisticRegression com StandardScaler. Usei o StandardScaler porque as variaveis estao em escalas diferentes, e isso influencia modelos lineares. Com random_state=42, o modelo obteve uma accuracy de cerca de 0.9825 no conjunto de teste.

## Teoria que apliquei

Este trabalho usa aprendizagem supervisionada, porque o dataset ja vem com a resposta certa para cada amostra. Cada linha representa uma biopsia e cada target indica se o caso é maligno ou benigno. O objetivo do modelo é aprender padrões nos dados de treino para depois prever a classe de exemplos que ainda nao viu.

O problema é de classificacao, nao de regressao, porque queremos prever uma categoria e nao um valor continuo. Neste caso ha duas classes, por isso é uma classificacao binaria. A classe `malignant` representa casos malignos e a classe `benign` representa casos benignos.

As 30 variaveis sao caracteristicas numericas extraidas das imagens. Isto torna o problema mais facil para modelos classicos de machine learning, porque em vez de trabalhar diretamente com pixels o modelo trabalha com medidas ja resumidas. Ao mesmo tempo, isto tambem limita o modelo, porque ele so consegue usar a informacao que esta nessas variaveis.

Antes de treinar o modelo dividi os dados em treino e teste. O treino serve para o modelo aprender. O teste serve para avaliar o modelo em dados que nao foram usados durante o treino. Isto é importante porque um modelo pode decorar os dados de treino e parecer muito bom, mas depois falhar em dados novos. Usei `random_state=42` para o resultado ser reproduzivel.

Tambem usei `stratify=y` no train_test_split. Isto faz com que a proporcao entre casos malignos e benignos seja mantida no treino e no teste. Como ha mais casos benignos do que malignos, isto ajuda a evitar uma divisao azarada em que uma das classes fique mal representada.

Usei StandardScaler porque as variaveis nao estao todas na mesma escala. Algumas medidas podem ter valores maiores do que outras, e um modelo linear pode dar importancia exagerada a variaveis so porque têm numeros maiores. A normalizacao coloca as variaveis numa escala mais comparavel.

A LogisticRegression é um modelo simples e interpretavel para classificacao. Apesar do nome ter "regression", neste caso ela é usada para prever classes. Ela tenta encontrar uma fronteira de decisao que separa os casos malignos dos benignos com base nas variaveis.

A accuracy indica a percentagem total de previsoes certas. Neste caso foi alta, mas a accuracy sozinha nao chega para avaliar bem um problema medico. Se uma classe for mais frequente, um modelo pode ter boa accuracy mesmo falhando casos importantes da classe menos frequente.

Por isso tambem olhei para a confusion matrix. Ela mostra os acertos e os erros por classe. As linhas representam a classe real e as colunas representam a classe prevista. Assim consigo perceber nao so quantos erros existem, mas tambem que tipo de erro o modelo comete.

Neste caso, com as labels `[malignant, benign]`, a matriz foi:

```text
[[41  1]
 [ 1 71]]
```

Isto significa que o modelo acertou 41 casos malignos e 71 benignos. Errou 1 caso maligno como benigno e 1 caso benigno como maligno.

## Reflexao critica

Quando uma imagem é reduzida a um conjunto de numeros perde-se parte da informacao visual original. As 30 caracteristicas resumem medidas como raio, textura, area e concavidade, mas deixam de mostrar a forma completa da biopsia, a distribuicao espacial dos tecidos e outros detalhes visuais que podem ser importantes para um diagnostico.

Na imagem original pode existir informacao que estas caracteristicas nao capturam bem, por exemplo padroes locais, irregularidades pequenas, zonas suspeitas muito especificas ou relacoes espaciais entre diferentes partes da imagem. Ao trabalhar apenas com numeros, ficamos dependentes das caracteristicas que alguem decidiu extrair previamente.

O erro que considero mais grave na confusion matrix é prever um caso maligno como benigno. Neste dataset isso corresponde a uma linha real "malignant" ser classificada na coluna "benign". Este falso benigno é perigoso porque pode atrasar o diagnostico e o tratamento de uma pessoa que realmente tem cancro.

Um falso maligno tambem é um erro importante, porque pode causar ansiedade, exames extra e custos desnecessarios. Mesmo assim, em contexto clinico, deixar passar um caso maligno parece-me mais grave do que sinalizar um caso benigno como suspeito.

Tambem é importante nao confiar cegamente no modelo so porque a accuracy é alta. Este dataset é pequeno e limpo, por isso os resultados podem parecer melhores do que seriam num hospital real. Na pratica seria preciso testar com mais dados, dados de hospitais diferentes e validacao feita por especialistas antes de usar isto para apoiar decisoes clinicas.

Para mim, este modelo serve como baseline: é um primeiro ponto de comparacao. Ele mostra que dados tabulares ja conseguem bons resultados, mas tambem mostra a limitacao de nao usar a imagem original. Nas semanas seguintes, ao trabalhar com imagens, vou conseguir comparar se os pixels ou uma CNN conseguem capturar informacao que estas 30 variaveis nao capturam.
