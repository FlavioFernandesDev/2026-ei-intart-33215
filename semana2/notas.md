

# Semana 2 — Exploracao de imagens medicas (MedMNIST)

Nesta semana o objetivo era passar de dados/numa tabela da semana 1 para imagens medicas mesmo a serio onde existe informação em bruto e ninguem fez o trabalho de calcular certas variaveis por mim.  
Explorei os dois datasets propostos pelo professor, o BreastMNIST e o PathMNIST.  
O dataset BreastMNIST é de imagens (a preto e branco) ultrasons a mama para detetar cancro da mama assim como na 1ªSemana, enquantoque o PathMNIST é de visualização microsopica de tecido de partes do intestino (a cores). 



## Exploração do dataset

Antes de irmos para as perguntas desta semana vamos analisar alguma estatística imporante destes datasets, isto é importante para percebermos com o que estamos a trabalhar!  
Corri então o `explore.py`, que calcula as estatisticas dos pixeis e tambem gera imagens/graficos que vao para a pasta `figuras/` e que daqui a bocado já as vamos analisar durante a reposta às perguntas desta semana.  
Temos a segur um breve resumo em foramto tabela:

| Característica | BreastMNIST | PathMNIST |
|---|---|---|
| Nº de imagens | 780 | 107 180 |
| Canais | 1 (preto e branco) | 3 (RGB / cores) |
| Tamanho | 28×28 | 28×28 |
| Nº de classes | 2 (maligno / benigno) | 9 (tipos de tecido) |
| Split (treino/val/teste) | 546 / 78 / 156 | 89 996 / 10 004 / 7 180 |
| Distribuição das classes | 210 malignos (27%) / 570 benignos (73%) | equilibrado (8,9% a 14,5%) |
| Pixeis | 0 a 255, média 84.25 | 0 a 255, média 168.23 |

Aqui podemos reparar logo que, e vamos confirmar mais à frente, o BreastMNIST é relativamente pequeno em comparação com o PathMNIST.  
O BreastMNIST é a preto e branco e desiquilibrado enquanto que o PathMNIST é a cores e equilibrado.
Vamos analisar melhor estas diferneças na respostas às perguntas a seguir.  


## Reflexao 

### 1. As classes sao visivelmente diferentes? O que distingue maligno de benigno?

No BreastMNIST:  
As duas classes são na minha opinião levemente diferentes, se olharmos para a imagem `breastmnist_grelha.png`, podemos ver que os da segunda linha que sao os benignos tem uma imagem mais uniforme do que os da primeira linha que em alguma parte da iamgem existe uma diferença de tonalidade de cinzento maior isto é parece um caroço com contornos mais bem definidos, nos benigmos a imagem tende a ser mais lisa e sem este "caroço" visivel. Por outtro lado, ao olharmos para o ultimo da segunda linha parece me que tem um "caroço" o que pode indicar que as diferenças não são assim tão significativas mas existem, contudo caso a caso pode confundir. 
Podemos tambem reparar na `breastmnist_media_classe.png`, que a "imagem média" das classes tem diferenças, as do benigno são ligeiramente mais claras do que as do maligno. A média dos pixeis em maligno é 79.62 e em benigno 85.95.  
Os pixeis vao de valor 0(preto) até valor 255 (branco), temos o `breastmnist_histograma.png` que podemos ver no geral se as imagens têm mais pixeis para o branco ou preto o que concluimos atraves da curva que são mais escuras, média global de 84.25. 

No PathMNIST: As varias classes sao bastante distintas entre si com cores diferentes, padroes completamente diferentes de umas para as outras classes. 

### 2. O dataset esta equilibrado? Que implicacoes para o treino e a avaliacao?

No BreastMNIST:  
O Dataset não está de todo equilibrado, muito pelo contrário ele está desiquilibrado (~27% maligno e ~73% benigno, 210 malignos e 570 benignos). Isto tem implicações no treino e para perceber isto vamos olhar para o problema como pensando como o modelo aprende. Ele aprende tentando errar o menos possível e para isso o modelo pode se inclinar a responder quase sempre benigno porque vai acertar bastantes vezes neste dataset enviesado, é como se o modelo ignorasse a classe que é mais rara porque não é tao importante aprender sobre ela porque são poucos casos e então ao errar nesses erra poucos. Ora, isto não é de todo correto e ainda por cima a classe que ele está a errar são falsos benignos que foi o pior caso que foi identificado na semana anterior que é o caso de alguem ter cancro mas ele dizer que está tudo ok.  
Na avaliação, ou seja depois de treinado e a medir a sua avaliação, aqui o problema não é do modelo em si mas sim da forma como nós o avaliamos. Se usarmos uma metrica errada estamos nos a enganar a pensar que tem uma boa avalição.  
A "accuracy" é a percentagem de acertos, e quando exisite um desiquilibrio no dataset ela pode induzir em erro. Imaginemos um modelo que responde benigno a todas as imagens, ele vai ter 73% de accuracy neste exemplo, o que à primeira vista parece ótimo. Mas é muito perigoso porque na verdade nunca apanhou um unico caso de cancro!  
Desta forma, na avaliação não basta o acuracy temos de usar a "confusion matrix" que separa os erros por classes, quanto malignos foram classificados como benignos e vice versa.  
E fazer um recall da classe maligna, isto é, dos casos malignos que exisitam quantos é que o modelo consegiu acertar. Esta é a metrica importante porque mede o maior risco nos casos clinicos.  
Resumindo, no treino o desiquilibrio faz o modelo aprender de forma errada na avaliação existe a possbilidade de nos enganarmos a fazer a medição correta, contudo isto tem solução que é dar mais peso à classe rara em vez de igual peso às duas. Ou então manter uma estruturação de split como fiz na semana 1. 

No PathMNIST: 
O Dataset está equilibrado por isso estes problemas não se verificam. Como podemos ver em `pathmnist_distribuicao.png`. 


### 3. Que diferencas em relacao a Semana 1? O que se ganha ao trabalhar com imagens?

Na semana 1 nós tinhamos tabelas com valores, ou seja metricas/numeros que alguem decidiu calcular de alguma forma analisando as imagens. Nessa mesma semana nas notas eu apontei que reduzir uma imagem a 30 numeros fazia perder informação.
Nesta semana temos as imagens mesmo, ou seja sem falta de informação porque temos a informação toda da imagem. Na primeira semana cada "caso" tinha 30 variaveis/valores enquanto que nesta semana cada "caso" tem no BreastMNIST 784 valores que são os 28 por 28 pixeis e cada picel tem um valor de 0 a 255. Aqui ninguem mediu nada temos os valores em bruto.  
Podemos pensar como se na semana 1 alguém tivesse lido um livro e nós tinhamos um resumo de 30 páginas e nesta semana temos o livro com 784 paginas. Temos muita mais informação nesta semana mas também temos muito mais para ler!  

O que se ganha:  
- Temos muita mais informação para analisar
- Não estamos dependentes de alguma variavel que alguem achou que fazia sentido calcular, e podia ter escapado outra variavel/padrao importante e que se nao tivessmos a imagem nunca iamos conseguir ver, assim com a imagem conseguimos ver tudo e notar em padroes que ainda nao tinham sido descobertos. 
- Um modelo pode ser capaz de aprender a "ver" a forma e a textura real em vez de apenas analisar numeros como por exmeplo forma: 0,34 e textura: 0,37. (exemplos meramente explicativos).  

Desvantagens:  
- Temos muitos mais numeros (784 vs 30). Isto torna muito mais dificil de aprender e precisamos de muitos mais casos. 
- A informação pode conter algum ruido por exemplo do aparelho que tira a imagem. 
- O modelo que foi aplciado na semana1 como o de regressão logística deixa de funcionar aqui. Com 30 features e limpas funciona bem, mas com 784 pixeis vai ter dificuldade. Ou ainda pior se for em RGB como o pathmnsit que tem 3x mais porque cada pixel tem 3 valores. 
