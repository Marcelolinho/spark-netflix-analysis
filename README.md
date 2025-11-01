# Trabalho Prático - Análise Dataset Netflix com Apache Spark

## Contexto do Projeto

Desenvolvimento de análise de dados usando **Apache Spark (PySpark)** para processar o dataset da Netflix e criar recomendações hipotéticas baseadas em análise quantitativa.

## Configuração Inicial

**Arquivo principal**: `main.py`  
**Dataset**: `./dataset/netflix_titles.csv`  
**Tecnologias**: Python, PySpark, NLTK

## Objetivo

Analisar o dataset de streaming da Netflix para criar uma série e um filme hipotéticos, justificando com base em análises de dados e apresentando o código completo.

---

## Implementação Requerida

### Etapa 1: Carregar e Dividir Dataset

Criar dois DataFrames separados baseados na coluna `type`:

- `series_df`: filtra registros onde `type == "TV Show"`
- `movies_df`: filtra registros onde `type == "Movie"`

### Etapa 2: Processar Coluna Description

Criar nova coluna `description_scraped` em ambos os DataFrames:

- Remover stopwords da coluna `description` usando NLTK
- Manter apenas palavras significativas
- Aplicar transformação usando UDF (User Defined Function)

**Requisitos técnicos**:

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


---

## Sistema de Pontuação (4 Critérios)

### Critério 1: Top 10 Palavras em `description_scraped`

**Objetivo**: Identificar as 10 palavras mais frequentes nas descrições processadas

**Sistema de pontos** (inverso à colocação):
- 1º lugar → 10 pontos
- 2º lugar → 9 pontos
- 3º lugar → 8 pontos
- 4º lugar → 7 pontos
- 5º lugar → 6 pontos
- 6º lugar → 5 pontos
- 7º lugar → 4 pontos
- 8º lugar → 3 pontos
- 9º lugar → 2 pontos
- 10º lugar → 1 ponto

**Aplicação**: Cada linha recebe pontos acumulados baseados em quantas palavras do top 10 aparecem em sua `description_scraped`.

### Critério 2: Top 5 Gêneros em `listed_in`

**Objetivo**: Identificar os 5 gêneros/categorias mais frequentes

**Sistema de pontos** (valor dobrado):
- 1º lugar → 25 pontos (5 × 5)
- 2º lugar → 20 pontos (4 × 5)
- 3º lugar → 15 pontos (3 × 5)
- 4º lugar → 10 pontos (2 × 5)
- 5º lugar → 5 pontos (1 × 5)

**Aplicação**: Cada linha recebe pontos se seu `listed_in` corresponde a um dos top 5.

### Critério 3: Pontuação por `release_year`

**Objetivo**: Valorizar filmes/séries por período de lançamento

**Sistema de pontos**:
- 2000-2005 → 5 pontos
- 2006-2010 → 10 pontos
- 2011-2015 → 15 pontos
- 2016-2021 → 20 pontos
- Fora desses períodos → 0 pontos

### Critério 4: Top 10 Palavras em `title`

**Objetivo**: Identificar as 10 palavras mais frequentes nos títulos

**Sistema de pontos**: Idêntico ao Critério 1
- Pontuação inversa de 10 a 1 ponto
- Cada linha acumula pontos baseado nas palavras do top 10 presentes em seu título

---

## Cálculo Final e Saída

### Processamento Final

1. Somar pontos dos 4 critérios para cada linha (filme/série)
2. Ordenar por pontuação total em ordem decrescente
3. Selecionar:
   - **Top 15 séries** com maior pontuação
   - **Top 15 filmes** com maior pontuação

### Formato de Saída Esperado

Para cada item do ranking, exibir:

**Colunas obrigatórias**:
- `title` - Título do filme/série
- `listed_in` - Gênero(s)/Categoria(s)
- `release_year` - Ano de lançamento
- `description` - Descrição original

**Pontuação detalhada**:
- Pontos Critério 1 (description_scraped)
- Pontos Critério 2 (listed_in)
- Pontos Critério 3 (release_year)
- Pontos Critério 4 (title)
- **TOTAL DE PONTOS**

---

## Estrutura Sugerida do `main.py`

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, split, lower, regexp_replace
from pyspark.sql.types import StringType, IntegerType
from nltk.corpus import stopwords
import nltk

1. Configurar SparkSession
2. Carregar CSV do dataset
3. Dividir em series_df e movies_df
4. Criar coluna description_scraped (remover stopwords)
5. Calcular top 10 palavras mais frequentes em description_scraped
6. Calcular top 5 gêneros mais frequentes em listed_in
7. Calcular top 10 palavras mais frequentes em title
8. Criar função de pontuação para cada critério
9. Aplicar sistema de pontuação em cada linha
10. Somar pontos totais
11. Ordenar e selecionar top 15 filmes
12. Ordenar e selecionar top 15 séries
13. Exibir resultados com detalhamento de pontos

---

## Considerações de Implementação

### Tratamento de Dados

- Tratar valores nulos/vazios em todas as colunas
- Normalizar texto para lowercase antes de processar
- Remover pontuação e caracteres especiais
- Considerar que `listed_in` pode conter múltiplos gêneros separados por vírgula

### Otimizações PySpark

- Usar transformações distribuídas (map, filter, flatMap)
- Evitar collect() em grandes volumes
- Utilizar cache() para DataFrames reutilizados
- Aplicar UDFs com decorador @udf para melhor performance

### Stopwords em Inglês

stop_words = set(stopwords.words('english'))


### Análise de Frequência

Para contar palavras mais frequentes:
- Usar flatMap para separar palavras
- Aplicar map para contar ocorrências
- Usar reduceByKey para agregação
- Ordenar por frequência e limitar ao top N

---

## Entregável Final

1. **Código completo** do `main.py` funcional
2. **Saída formatada** com top 15 filmes e séries
3. **Justificativa** para série e filme hipotéticos baseada nos padrões identificados:
   - Palavras-chave mais valorizadas
   - Gêneros mais populares
   - Períodos de lançamento mais pontuados
   - Elementos de título mais comuns

---

## Exemplo de Análise Esperada

Com base nos resultados do ranking:
- **Série hipotética**: Deve incorporar palavras-chave top, gêneros valorizados e período otimizado
- **Filme hipotético**: Mesma lógica aplicada ao contexto de filmes
- **Justificativa**: Explicar como cada elemento identificado contribui para a pontuação máxima
