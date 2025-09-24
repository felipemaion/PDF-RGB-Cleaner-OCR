# PDF RGB Cleaner + OCR

Remove (pinta de branco) áreas de **PDFs** cujos pixels estão dentro de uma **faixa RGB** configurável.
Opcionalmente gera **previews** (apenas removido, máscara BW, overlay) e extrai **texto (OCR)** para `.txt`.

- **Entrada:** PDFs em uma pasta.
- **Saídas:** PDF processado, PDFs de inspeção (opcionais) e TXT extraído (opcional).
- **Execução:** Python local **ou** Docker (recomendado para portabilidade).
- **Paralelismo:** processamento em múltiplos processos.
- **Autor:** Felipe Maion

---

## Funcionalidades

- **Remoção por faixa RGB** (globais ou por canal; `>`/`<` ou `>=`/`<=`).
- **Região opcional** via função `judge()` (geometria específica).
- **Previews**:

  - **Only-Removed:** mostra **só** as áreas removidas com cores originais; fundo preto/branco.
  - **Mask (BW):** branco = removido; preto = mantido.
  - **Overlay:** destaca removidos sobre a página original (cor/alpha configuráveis).

- **OCR/TXT:** usa Tesseract (`pytesseract`) e salva um `.txt` por PDF.

---

## Estrutura de Pastas (padrão com Docker)

```
data/
 ├─ Input/         # coloque aqui seus PDFs
 ├─ Output/        # PDFs processados (pintados de branco)
 ├─ OutputTxt/     # TXT (OCR) dos PDFs processados
 ├─ OutputMask/    # (opcional) PDFs com máscara BW
 ├─ OutputOverlay/ # (opcional) PDFs com overlay
 └─ OutputRemoved/ # (opcional) PDFs exibindo apenas removidos
```

---

## Uso Rápido (Docker + Docker Compose)

> Um comando, zero dores: `docker compose up --build`

1. **Pré-requisitos**: Docker Desktop (ou Engine) com `docker compose v2`.
2. **Preparar pastas**:

   ```bash
   mkdir -p data/Input data/Output data/OutputTxt
   ```

3. **Coloque seus PDFs** em `data/Input/`.
4. **Execute**:

   ```bash
   docker compose up --build
   ```

5. **Resultados**:

   - `data/Output/`: PDFs com remoção aplicada
   - `data/OutputTxt/`: arquivos `.txt` do OCR
   - (se habilitado no compose) `data/OutputMask/` e `data/OutputOverlay/`

### Scripts de 1 clique

- **macOS/Linux:** `./run.sh`
- **Windows:** `run.bat`

Ambos criam as pastas, constroem a imagem e rodam o compose.

---

## Uso (Docker direto)

### Construir a imagem

```bash
docker build -t pdf-rgb-cleaner:latest .
```

### Rodar com defaults (lê `/data/Input`, grava `/data/Output`, TXT em `/data/OutputTxt`)

```bash
docker run --rm -it -v "$PWD/data:/data" pdf-rgb-cleaner:latest
```

### Rodar com opções extras (ex.: previews + OCR)

```bash
docker run --rm -it -v "$PWD/data:/data" pdf-rgb-cleaner:latest \
  --extract-text --ocr-lang por+eng \
  --rgb-min 170 --rgb-max 250 --preview-mask --overlay
```

> **Importante:** a imagem já contém Poppler e Tesseract (`por` + `eng`).
> O entrypoint garante `-i /data/Input -o /data/Output --dpi 300 --text-dir /data/OutputTxt`.

---

## Uso (Python local)

### Requisitos

- Python 3.10+
- Bibliotecas: `numpy`, `Pillow`, `pdf2image`, `pytesseract`
- **Sistemas:**

  - **Poppler** (p/ `pdf2image`)
  - **Tesseract** + idiomas (`por`, `eng`) p/ OCR

`requirements.txt`:

```txt
numpy>=1.23
Pillow>=9.5
pdf2image>=1.17
pytesseract>=0.3.10
```

### Execução

```bash
python pdf_batch_remove_with_previews_and_ocr.py \
  -i Input -o Output --dpi 300 --text-dir OutputTxt \
  --rgb-min 170 --rgb-max 250 --preview-mask --overlay \
  --extract-text --ocr-lang por+eng
```

---

## Parâmetros (principais)

- **Entrada/Saída**

  - `-i, --input_dir` (default: `Input`)
  - `-o, --output_dir` (default: `Output`)
  - `--text-dir` (default: `OutputTxt`)
  - `--dpi` (default: `300`)
  - `--workers` (default: nº de CPUs)

- **Critérios RGB**

  - `--rgb-min` / `--rgb-max` (default: `170–250`)
  - `--rmin --rmax --gmin --gmax --bmin --bmax` (sobrescrevem o global)
  - `--inclusive` (usa `>=` e `<=`)

- **Região**

  - `--apply-judge` (ativa a função `judge()`)

- **Previews**

  - `--preview-removed` + `--removed-dir` (default: `OutputRemoved`)
    `--bg {black|white}` (fundo do preview), `--skip-empty`
  - `--preview-mask` + `--mask-dir` (default: `OutputMask`)
  - `--overlay` + `--overlay-dir` (default: `OutputOverlay`)
    `--overlay-color "#ff0000"` `--overlay-alpha 0.6` `--overlay-skip-empty`

- **OCR**

  - `--extract-text` (salva TXT)
  - `--ocr-lang por` (ex.: `por`, `eng`, `por+eng`)
  - `--tesseract-cmd` (caminho do binário, se necessário)
  - `--tessdata-dir` (caminho da pasta `tessdata`, se necessário)

---

## Dicas e Troubleshooting

- **Nada acontece?** Verifique se há PDFs em `Input/` (ou `data/Input/` no Docker).
- **OCR falhou / idioma ausente:**

  - Local: instale Tesseract + idioma (`por`, `eng`) e, se preciso, ajuste `--tessdata-dir` / `TESSDATA_PREFIX`.
  - Docker: já vem pronto (Debian/Ubuntu). Dentro do container, o `TESSDATA_PREFIX` padrão é `/usr/share/tesseract-ocr/4.00/tessdata`.

- **Performance:**

  - Aumente `--workers`.
  - `--dpi 300` é bom para OCR; mais que isso aumenta tempo/tamanho.

- **Faixa RGB diferente?** Ajuste `--rgb-min/--rgb-max` ou por canal.

---

## Desenvolvimento

- Código principal: `pdf_batch_remove_with_previews_and_ocr.py`
- Docker:

  - `Dockerfile` com Poppler + Tesseract (`por`/`eng`)
  - `entrypoint.sh` garante defaults e agrega flags do `docker run`
  - `docker-compose.yml` automatiza a execução

- Scripts:

  - `run.sh` (macOS/Linux), `run.bat` (Windows)

---

## Autor

**Felipe Maion**
Contribuições e issues são bem-vindas!
