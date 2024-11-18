
# Implementação do Sistema de Processamento Fonético

## 1. Classe Vocab (vocab.py)

### 1.1 Estrutura do Arquivo de Vocabulário
O arquivo `data/lang/phone.vocab` deve seguir o formato:
```text
<blk> 0
p1 1
p2 2
p3 3
...
```

### 1.2 Requisitos da Classe
A classe `Vocab` deve implementar três métodos:
- `encode`: Similar ao SentencePieceProcessor
- `decode`: Similar ao SentencePieceProcessor
- `__len__`: Retorna o número total de símbolos fonéticos

## 2. Preparação dos Dados

### 2.1 Requisitos dos Manifests
- Criar arquivos manifest para:
  - Treino (train)
  - Desenvolvimento (dev)
  - Teste (test)
- Formato: [Lhotse](https://lhotse.readthedocs.io/en/latest/corpus.html#supervision-manifest) (seguindo exemplo do [icefall](https://github.com/k2-fsa/icefall/tree/master/egs))
- Campos obrigatórios:
  - `text`: Transcrição
  - `extra`: Dados fonéticos

## 3. Treinamento do Modelo

### 3.1 Modelo Baseline
```bash
# Comando de treinamento
python train.py ... --ctc-loss-scale-phones 0
```

### 3.2 Modelo Hierárquico
```bash
# Comando de treinamento
python train.py ... --ctc-loss-scale-phones > 0
```

## 4. Avaliação dos Modelos

### 4.1 Modelo Baseline
```bash
python decode.py # Não requer argumentos adicionais no decode.py
```

### 4.2 Modelo Hierárquico
```bash
# Requer especificação do arquivo de vocabulário
python decode.py --phone-vocab [caminho_do_arquivo]
```