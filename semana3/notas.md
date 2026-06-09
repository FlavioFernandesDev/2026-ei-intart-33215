# Semana 3 - Modelos classicos sobre pixels

Nesta semana o objetivo era perceber uma forma mais arcaica/antiga de trabalhar com imagens, antes de usar CNNs.  
A ideia foi pegar nas imagens do **BreastMNIST**, cujas imagens são 28x28 e a preto e branco, e transformar cada imagem numa lista de numeros.

Ou seja, uma imagem que antes era uma matriz 28x28 passou a ser um vetor com 784 valores (28 x 28 = 78).  
Isto foi feito com `.reshape(n, -1)`. Assim o modelo deixa de ver uma imagem como imagem e passa a ver uma linha de uma tabela, so que em vez de ter 30 caracteristicas como na Semana 1, tem 784 pixeis.

Eu usei o dataset BreastMNIST porque continua o tema do cancro da mama, tal como na semana 1 e na semana 2. O dataset ja vem dividido em treino, validacao e teste:

| Split | Total | Malignant | Normal/benign |
|---|---:|---:|---:|
| Treino | 546 | 147 | 399 |
| Validacao | 78 | 21 | 57 |
| Teste | 156 | 42 | 114 |

As imagens vinham com valores de 0 a 255. Eu normalizei para 0 a 1, porque assim os valores ficam numa escala mais pequena e mais facil para os modelos. Além disso faz com que consiga igualar os pesos. Depois testei modelos classicos de `scikit-learn`, ou seja, modelos que nao sao redes neuronais convolucionais.

Usei:

- `LogisticRegression` com os pixeis achatados;
- `SVC` com kernel RBF com os pixeis achatados;
- `LogisticRegression` com HOG, como teste extra.

O HOG é uma tecnica que tenta resumir a imagem olhando para as direçoes de contornos/gradientes. Isto quer dizer que em vez de dar os pixeis todos ao modelo, damos uma espécie de resumo dos contornos da imagem. A ideia parecia fazer sentido porque em imagens medicas os contornos podem ser importantes, mas neste caso nao correu melhor.

## Resultados obtidos

| Modelo | Accuracy | Balanced accuracy | Recall maligno | Erros |
|---|---:|---:|---:|---:|
| LogisticRegression pixels | 0.7821 | 0.7531 | 0.6905 | 34 |
| SVC RBF pixels | 0.7949 | 0.7393 | 0.6190 | 32 |
| LogisticRegression HOG | 0.7308 | 0.6504 | 0.4762 | 42 |

A **accuracy** mede a percentagem total de previsoes certas. O SVC teve a melhor accuracy geral, com 0.7949.

Mas neste problema a accuracy sozinha não chega, porque o dataset está desiquilibrado. Como já foir falado desta importância em semanas anteriores.  
Existem mais casos `normal/benign` do que `malignant`. Por isso tambem olhei para o **recall maligno**, que mede, dos casos malignos que existiam, quantos o modelo conseguiu apanhar. Aqui a LogisticRegression com pixeis foi melhor, porque teve recall maligno de 0.6905.

Isto e importante porque no contexto medico nao me interessa apenas acertar muitos casos no total. Interessa-me muito perceber se o modelo esta a deixar passar casos malignos como benignos.

As confusion matrices foram:

```text
LogisticRegression pixels
[[29 13]
 [21 93]]

SVC RBF pixels
[[26 16]
 [16 98]]

LogisticRegression HOG
[[20 22]
 [20 94]]
```

Aqui a ordem das classes e `[malignant, normal/benign]`. As linhas sao a classe real e as colunas sao a classe prevista. Por exemplo, na LogisticRegression, o modelo acertou 29 malignos, mas classificou 13 malignos como benignos. Estes 13 sao o erro mais preocupante.

## Reflexão

### 1. Que impacto tem usar imagens em vez de caracteristicas extraidas manualmente da Semana 1?

Na Semana 1 eu usei o dataset `load_breast_cancer`, este tinha 30 caracteristicas numéricas. Essas caracteristicas já tinham sido extraidas das imagens por alguem. Por exemplo, valores relacionados com raio, textura, area, concavidade, etc. Ou seja, na Semana 1 eu nao estava a trabalhar com a imagem original, estava a trabalhar com um resumo da imagem.

Nesta semana 3 a situação mudou. Aqui usei as imagens mais diretamente. Cada pixel entrou como um número. Isto quer dizer que em vez de confiar apenas nas 30 medidas que alguem decidiu calcular, o modelo recebeu muita mais informação bruta da imagem.

Podemos pensar assim: na Semana 1 era como se alguem tivesse lido a imagem por mim e me tivesse dado um resumo com 30 pontos importantes. Na Semana 3 eu tentei dar quase a imagem ao modelo, mas de uma forma muito simples: transformei a imagem numa lista de 784 numeros.

Isto tem vantagens. A principal e que posso estar a dar ao modelo informacao que as 30 caracteristicas da Semana 1 nao captavam. Pode haver padroes visuais, sombras, zonas mais claras ou mais escuras, ou pequenas formas que nao aparecem num conjunto pequeno de caracteristicas manuais.

Mas tambem tem desvantagens. Passar de 30 valores para 784 valores torna o problema mais dificil. O modelo tem muito mais numeros para analisar e pode ser mais facil confundir-se, principalmente porque o dataset BreastMNIST nao e muito grande.

Outra desvantagem e que, ao achatar a imagem, eu perco parte da noçao espacial da imagem. Para nos, quando olhamos para uma imagem, vemos zonas, formas, contornos e texturas. Mas quando faço `.reshape(n, -1)`, o modelo recebe uma lista de numeros. Ele sabe o valor dos pixeis, mas nao "ve" a imagem da mesma forma que nos vemos. A relacao entre um pixel e os pixeis vizinhos fica menos natural.

Por isso, usar imagens dá mais informação, mas tambem aumenta a dificuldade. Nao basta ter mais dados se o modelo nao for bom a interpretar a estrutura desses dados.

### 2. Que padroes visuais sao dificeis para um classificador linear?

Um classificador linear, como a `LogisticRegression`, tenta separar as classes com uma regra relativamente simples. No fundo, ele da pesos aos pixeis e tenta decidir se a soma desses pesos aponta mais para maligno ou para benigno.

Isto funciona melhor quando as classes sao separaveis de forma mais simples. Por exemplo, se todos os malignos fossem sempre muito escuros numa zona especifica e todos os benignos fossem sempre claros nessa mesma zona, talvez fosse mais facil.

Mas imagens medicas nao sao assim tao simples. O que distingue maligno de benigno pode depender de varios detalhes ao mesmo tempo:

- a forma da lesao;
- os contornos serem mais ou menos definidos;
- a textura da zona;
- o contraste entre a lesao e o fundo;
- pequenas diferencas entre zonas proximas da imagem;
- partes escuras ou claras que aparecem em sitios diferentes.

Isto e dificil para um classificador linear porque ele nao entende bem "formas". Ele trabalha com os valores dos pixeis, mas nao tem uma forma natural de aprender que um conjunto de pixeis juntos forma um contorno ou uma massa suspeita. Ele olha para a lista de numeros, nao para a imagem como uma estrutura.

Os casos mais dificeis sao aqueles em que:

- a lesao e pequena;
- o contraste e baixo;
- a imagem esta muito escura;
- os contornos nao sao claros;
- um caso benigno parece ter uma zona suspeita;
- um caso maligno nao tem um aspeto muito obvio.

Na minha opiniao, isto ajuda a perceber porque e que na Semana 4 vai fazer sentido usar CNNs. Uma CNN olha para regioes pequenas da imagem e aprende filtros, por isso consegue aproveitar melhor a estrutura espacial. Ja os modelos desta semana tratam a imagem quase como uma tabela gigante.

### 3. Os erros fazem sentido visualmente? Que casos sao mais ambiguos?

O script gerou imagens com os erros dos modelos:

- `figuras/erros_logisticregression_pixels.png`
- `figuras/erros_svc_rbf_pixels.png`
- `figuras/erros_logisticregression_hog.png`

Ao olhar para esses erros, acho que alguns fazem sentido visualmente. Ha imagens em que eu tambem nao consigo dizer facilmente se parecem malignas ou benignas. Algumas sao muito escuras, outras tem pouco contraste, outras tem zonas que parecem suspeitas mas podem nao ser malignas.

Os casos mais ambiguos parecem ser aqueles em que existe uma zona mais escura ou uma especie de "caroco", mas sem ser muito claro se aquilo e realmente sinal de malignidade. Tambem ha imagens benignas que parecem ter uma mancha ou uma zona mais marcada, e isso pode levar o modelo a prever maligno.

Por outro lado, tambem ha casos malignos que o modelo classificou como benignos. Estes sao os que me preocupam mais. Na LogisticRegression isto aconteceu 13 vezes, no SVC aconteceu 16 vezes e no HOG aconteceu 22 vezes.

Este erro chama-se falso benigno: a pessoa tinha um caso maligno, mas o modelo disse que era benigno. Tal como ja tinha referido na Semana 1 e na Semana 2, este e o erro mais perigoso no contexto clinico, porque pode atrasar o diagnostico e o tratamento.

Tambem existem falsos malignos, ou seja, casos benignos que foram classificados como malignos. Isto tambem e mau, porque pode causar ansiedade, exames extra e custos. Mas, na minha opiniao, continua a ser menos grave do que deixar passar um cancro como se estivesse tudo bem.

O HOG foi o pior neste ponto, porque falhou 22 dos 42 casos malignos no teste. Isto mostra que uma tecnica que parece boa em teoria nem sempre melhora os resultados. Neste dataset, com esta configuracao, os pixeis em bruto deram melhor resultado do que o HOG.

## Conclusao

Esta semana ajudou-me a perceber que trabalhar com imagens não é simplesmente "dar mais informacao ao modelo". As imagens têm mais informação do que as 30 caracteristicas da Semana 1, mas essa informação vem de forma mais dificil de interpretar.

Os modelos clássicos conseguem aprender alguns padroes, mas continuam limitados. A LogisticRegression teve melhor recall maligno, enquanto o SVC teve melhor accuracy geral. Como estamos num problema médico, eu valorizo bastante o recall maligno, porque quero reduzir os falsos benignos, como ja chegamos a essa conclusão nas semanas anteriores.

Para mim, esta semana fica como uma ponte entre a Semana 1 e a Semana 4 espreitando já o que vai ser feito posteriormente. Na Semana 1 usei caracteristicas ja extraidas. Nesta semana usei pixeis achatados. Na Semana 4, com CNNs, espero conseguir aproveitar melhor a estrutura da imagem em vez de a transformar apenas numa lista de números.
