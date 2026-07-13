# Identificador de Fraudes + Assistente de IA — Banco Vórtex

Projeto para desafio de análise de dados em Python. Tem duas partes que se
conectam:

1. **Pipeline de detecção de fraudes**: gera dados sintéticos de transações,
   faz análise exploratória e treina um modelo de machine learning para
   sinalizar transações suspeitas.
2. **Assistente de IA (chat)**: uma interface conversacional para o banco
   fictício "Vórtex", que usa os números gerados pelo pipeline para explicar
   a situação de fraudes a **clientes** e **profissionais do banco**, em dois
   modos de linguagem diferentes.

---

## Estrutura do projeto

```
fraud_detector/
├── generate_data.py         # gera o dataset sintético de transações
├── fraud_detection.py       # EDA, modelagem e avaliação
├── assistente_fraude.html   # assistente de IA (chat) do Banco Vórtex
├── requirements.txt
├── README.md
├── data/
│   └── transacoes.csv       # gerado por generate_data.py
└── output/                  # gráficos e resultados gerados por fraud_detection.py
    ├── eda_visao_geral.png
    ├── matriz_confusao_baseline.png
    ├── matriz_confusao_random_forest.png
    ├── curva_roc.png
    ├── importancia_features.png
    └── transacoes_suspeitas.csv
```

---

## Parte 1 — Pipeline de detecção de fraudes

### Como rodar

```bash
pip install -r requirements.txt
python generate_data.py       # cria data/transacoes.csv
python fraud_detection.py     # roda a análise completa
```

### 1.1. Geração de dados (`generate_data.py`)

Cria **20.000 transações** sintéticas, das quais **2% (400) são fraude** —
uma proporção desbalanceada, como costuma acontecer em dados reais de fraude.

Cada transação tem os seguintes atributos:

| Coluna | Descrição |
|---|---|
| `id_transacao` | identificador único |
| `valor` | valor da transação em R$ |
| `hora_do_dia` | hora em que a transação ocorreu (0–23) |
| `distancia_do_titular_km` | distância entre o local da transação e o local habitual do titular |
| `qtd_transacoes_ultima_hora` | quantas transações o cartão fez na última hora |
| `transacao_internacional` | se a transação foi feita fora do país (0/1) |
| `categoria_estabelecimento` | tipo de comércio (mercado, restaurante, eletrônicos, etc.) |
| `fraude` | rótulo real: 1 = fraude, 0 = legítima |

As transações fraudulentas seguem um padrão propositalmente diferente:
valores mais altos, horários de madrugada, maior distância do titular e
maior frequência — o que a análise exploratória e o modelo devem conseguir
capturar.

### 1.2. Análise exploratória (`analise_exploratoria` em `fraud_detection.py`)

Gera `output/eda_visao_geral.png`, com 4 gráficos comparando o comportamento
de transações legítimas e fraudulentas:
- distribuição do valor da transação;
- distribuição por horário do dia;
- distribuição da distância do titular;
- taxa de fraude por categoria de estabelecimento.

### 1.3. Engenharia de atributos (`preparar_features`)

- Cria a variável binária `horario_madrugada` (1 se a transação ocorreu
  entre 22h e 6h).
- Converte `categoria_estabelecimento` em variáveis binárias (one-hot
  encoding), para uso no modelo.

### 1.4. Dois modelos de detecção

**Baseline por regras de negócio** (`baseline_regras`): sinaliza como
suspeita qualquer transação que combine, ao mesmo tempo, valor alto (top
10%), horário de madrugada e distância alta do titular (top 10%). Serve
como ponto de comparação simples, sem machine learning.

**Random Forest** (`treinar_modelo`): modelo de classificação treinado com
`class_weight="balanced"` para lidar com o desbalanceamento entre fraude e
não-fraude. Os dados são padronizados (`StandardScaler`) antes do
treinamento, e o conjunto é dividido em treino (75%) e teste (25%) de forma
estratificada, preservando a proporção de fraudes nos dois conjuntos.

### 1.5. Avaliação

Para os dois modelos são calculados: **precisão**, **recall**, **F1-score**
e **matriz de confusão**; para o Random Forest, também a **curva ROC** e a
**importância de cada variável** no modelo. Métricas de recall e precisão
são priorizadas em vez de acurácia simples, já que a classe de interesse
(fraude) é rara — um modelo que "prevê tudo como legítimo" teria 98% de
acurácia e zero utilidade prática.

Resultado obtido nesta execução de referência (dados sintéticos):

| Modelo | Precisão (fraude) | Recall (fraude) | F1 (fraude) |
|---|---|---|---|
| Baseline (regras) | 0,973 | 0,708 | 0,819 |
| Random Forest | 0,990 | 0,970 | 0,980 |

> Esses números são altos porque os dados são sintéticos e foram gerados com
> padrões bem separáveis entre fraude e não-fraude. Em um dataset real, o
> desempenho tende a ser mais modesto — ver seção "Próximos passos".

### 1.6. Saída final

`output/transacoes_suspeitas.csv`: lista as transações do conjunto de teste
ordenadas pela probabilidade de fraude estimada pelo modelo, simulando uma
fila de revisão manual para a equipe de risco.

---

## Parte 2 — Assistente de IA (`assistente_fraude.html`)

Um artefato HTML autocontido que simula a central de atendimento sobre
fraudes de um banco fictício, o **Banco Vórtex**. Pode ser aberto
diretamente no navegador (ou dentro do painel de Artifacts do Claude).

### 2.1. O que ele faz

É um chat que responde dúvidas sobre fraude bancária, adaptando a
linguagem conforme quem está perguntando:

| Modo | Público | Estilo de resposta |
|---|---|---|
| **Sou cliente** | clientes do banco | linguagem simples, acolhedora, sem jargão técnico; foco em orientação prática (contestar transação, bloquear cartão, prazos) |
| **Sou profissional do banco** | analistas de fraude, risco, atendimento | linguagem técnica: métricas do modelo (precisão, recall), como interpretar o score de risco, próximos passos operacionais (revisão manual, bloqueio preventivo, escalonamento) |

### 2.2. Como o assistente usa os dados do modelo

O HTML embute, em uma constante (`CONTEXTO_DADOS`), um resumo dos números
gerados pelo pipeline da Parte 1 (total de transações analisadas, taxa de
fraude, valores médios, desempenho do modelo, principais sinais de risco).
Esse contexto é injetado no *system prompt* enviado à API, para que as
respostas do assistente sejam consistentes com os dados do projeto, em vez
de inventar números.

### 2.3. Funcionalidades da interface

- **Faixa de status** no topo, com indicadores em tempo real (transações
  monitoradas, taxa de fraude, recall e precisão do modelo).
- **Alternância de modo** (cliente / profissional), que troca o *system
  prompt* e as sugestões de pergunta exibidas.
- **Chips de perguntas rápidas**, diferentes para cada modo, para facilitar
  o primeiro contato.
- **Histórico de conversa** mantido durante a sessão (enviado a cada nova
  mensagem, já que a API não guarda memória entre chamadas).
- **Indicador de "digitando"** enquanto a resposta é gerada.
- Aviso fixo no rodapé deixando claro que o Banco Vórtex é fictício, que o
  assistente não acessa contas reais e não substitui os canais oficiais de
  contestação de fraude.

### 2.4. Limites intencionais (por design do prompt)

- Nunca pede dados sensíveis (senha, código completo do cartão etc.).
- No modo cliente, evita citar termos técnicos do modelo.
- No modo profissional, evita inventar números fora do contexto fornecido
  e recomenda confirmar políticas internas específicas com a área
  responsável.
- Deixa claro, se perguntado, que é um projeto de demonstração.

---

## Próximos passos possíveis (para evoluir o desafio)

- Trocar o Random Forest por XGBoost/LightGBM e comparar desempenho.
- Testar detecção de anomalias não-supervisionada (Isolation Forest, LOF)
  para cenários sem rótulos de fraude confirmados.
- Ajustar o limiar de decisão (`threshold`) em vez de usar 0,5 fixo,
  otimizando para o custo de falso negativo vs falso positivo.
- Validar com validação cruzada estratificada em vez de um único split.
- Trocar o dataset sintético por um dataset real (ex: Kaggle "Credit Card
  Fraud Detection").
- Conectar o assistente de IA a `output/transacoes_suspeitas.csv` de
  verdade (hoje ele só recebe um resumo estático dos números), permitindo
  perguntas sobre transações específicas.
- Adicionar um terceiro modo no assistente (ex: "supervisor/gestor"), com
  visão consolidada de indicadores em vez de casos individuais.
