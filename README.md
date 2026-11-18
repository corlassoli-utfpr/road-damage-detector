# Detecção de Danos em Pavimento (YOLOv8 vs Faster R-CNN) - RDD2022

Projeto unificado para treinamento, gerenciamento de checkpoints e testes comparativos visuais entre **YOLOv8** e **Faster R-CNN** utilizando o dataset de danos em pavimento (**RDD2022**).

---

## 📁 Estrutura do Projeto

```text
teste_modelos_rdd/
├── README.md               # Documentação completa do projeto
├── requirements.txt        # Dependências do projeto (PyTorch, Ultralytics, etc.)
├── .gitignore              # Regras de exclusão para o Git (ignora dados brutos e arquivos >100MB)
├── scripts/                # Scripts Python enxutos e comentados
│   ├── treinar.py          # Treinador unificado com proteção térmica e checkpoints por época
│   ├── testar_modelos.py   # Testador visual comparativo com arquivamento sem sobrescrita
│   └── gerar_manifests.py  # Gerador automático de manifestos dos modelos
├── archive/
│   ├── colab/              # Dados e métricas antigas do Google Colab / Drive
│   └── records/            # Histórico de testes comparativos anteriores (imagens e manifestos)
├── imagens_teste/          # Imagens exclusivas para teste (nunca alteradas pelos scripts)
├── modelos/
│   ├── manifests/          # Manifestos descritivos com histórico e métricas de cada modelo
│   │   ├── yolo/           # manifest.txt do YOLO
│   │   └── faster/         # manifest.txt do Faster R-CNN
│   └── weights/            # Arquivos binários de pesos (.pt e .pth)
│       ├── yolo/           # best.pt, last.pt, yolov8s.pt
│       └── faster/         # modelo_ultimo.pth (hospedado externamente / ignorado pelo Git)
├── resultados/             # Imagens comparativas e manifesto do ÚLTIMO teste realizado
└── dataset_rdd/            # Dataset RDD2022 (configuração data.yaml e imagens de treino/val)
```

---

## ⚙️ Instalação e Pré-requisitos

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏷️ Classes de Danos (RDD2022)

Os modelos detectam 4 categorias principais de danos em vias asfálticas:

| Código | Nome da Classe | Descrição |
| :---: | :--- | :--- |
| **D00** | Rachadura Longitudinal | Fissuras paralelas ao sentido da via |
| **D10** | Rachadura Transversal | Fissuras perpendiculares ao fluxo de tráfego |
| **D20** | Rachadura Tipo Jacaré | Trincas interconectadas em bloco (fadiga do asfalto) |
| **D40** | Buraco / Panela | Cavidades e desagregação do pavimento |

---

## 🚀 Como Treinar os Modelos

Execute os comandos a partir da raiz do projeto (`teste_modelos_rdd`). O padrão do treinador é de **10 épocas**.

### 1. YOLOv8
```bash
# Treino na GPU com proteção térmica (padrão: 10 épocas, resolução 480):
python scripts/treinar.py yolo 10 8 --device cuda --temp-limit 78 --imgsz 480

# Treinar um NOVO modelo do zero (ignora checkpoints anteriores):
python scripts/treinar.py yolo 10 8 --device cuda --novo

# Treino na CPU (sem acionar a GPU, zero calor):
python scripts/treinar.py yolo 10 8 --device cpu --novo
```

- **Retomada inteligente de checkpoints**:
  - Se interromper o treino (`Ctrl + C` ou desligar), o checkpoint da última época concluída é preservado em `modelos/weights/yolo/last.pt`.
  - Ao rodar novamente sem `--novo`, o script retoma exatamente da época onde parou até atingir a meta.
- **Proteção Térmica Ativa**:
  - A temperatura da GPU é verificada a cada 50 batches. Se atingir o `--temp-limit`, o script pausa a execução até a GPU esfriar, evitando superaquecimento.

### 2. Faster R-CNN
```bash
# Treino na GPU ou CPU:
python scripts/treinar.py faster 10 2 --device cpu --novo
```
- Os checkpoints são gravados a cada época em `modelos/weights/faster/modelo_ultimo.pth`.

---

## 🔍 Como Testar os Modelos

Para executar a comparação visual sobre as imagens da pasta `imagens_teste/`:

```bash
python scripts/testar_modelos.py
```

### O que o testador realiza automaticamente:
1. **Arquivamento Seguro**: Qualquer resultado existente em `resultados/` é transferido para `archive/records/` (adicionando timestamp se houver risco de colisão, garantindo que nada seja sobrescrito).
2. **Inferência Lado a Lado**:
   - **Lado Esquerdo (Vermelho)**: Detecções realizadas pelo **YOLOv8** (`best.pt`).
   - **Lado Direito (Azul)**: Detecções realizadas pelo **Faster R-CNN** (`modelo_ultimo.pth`).
3. **Novo Manifesto**: Salva as imagens comparativas e o relatório de execução em `resultados/manifest.txt`.

---

## 📊 Modelos Disponíveis no Projeto

| Modelo | Arquivo de Pesos | Épocas | Status / Observações |
| :--- | :--- | :---: | :--- |
| **YOLOv8 (Novo)** | `modelos/weights/yolo/best.pt` | 10 | Treinado em 17/11/2026. mAP50: D00=0.404, D10=0.403, D20=0.541, D40=0.272. |
| **YOLOv8 (Base)** | `modelos/weights/yolo/yolov8s.pt` | - | Pesos pré-treinados oficiais COCO da Ultralytics. |
| **Faster R-CNN** | `modelos/weights/faster/modelo_ultimo.pth` | 10 | ResNet-50 FPN treinado no RDD2022 (~158 MB). |

> **Nota sobre o GitHub**: Arquivos `.pth` maiores que 100 MB (como `modelo_ultimo.pth`) são ignorados pelo `.gitignore` para cumprir as políticas de tamanho do GitHub. Eles podem ser hospedados nas *Releases* do repositório ou em serviços de nuvem.

---

## 📋 Atualizar Manifestos

Para auditar e atualizar os manifestos descritivos de todos os modelos em disco:

```bash
python scripts/gerar_manifests.py
```
