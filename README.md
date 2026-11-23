# Operações Morfológicas Compostas com OpenCV

**Atividade Final – Processamento de Imagens | Grupo 10 | Tema 1**
Abertura, Fechamento e Gradiente Morfológico (Python + OpenCV + NumPy)

## O que o projeto demonstra

- Elemento estruturante (`cv2.getStructuringElement`): retângulo, elipse e cruz
- Erosão e dilatação implementadas **manualmente com NumPy** e conferidas contra `cv2.erode` / `cv2.dilate`
- Abertura, fechamento e gradiente com `cv2.morphologyEx`, verificando as definições:
  - Abertura = dilatação(erosão(A))
  - Fechamento = erosão(dilatação(A))
  - Gradiente = dilatação(A) − erosão(A)
- Aplicação: remoção de ruído "sal e pimenta" (e comparação da ordem das operações)
- Análise de parâmetros: tamanho do kernel (com métrica de erro), formato do kernel e número de iterações
- Imagens reais em tons de cinza

## Como executar

### Opção 1 – Google Colab
1. Abra o Colab e rode numa célula: `!git clone https://github.com/SEU_USUARIO/morfologia-opencv.git`
2. Depois: `%cd morfologia-opencv`
3. Abra `demo_morfologia.ipynb` e execute todas as células.

### Opção 2 – Local
```bash
git clone https://github.com/SEU_USUARIO/morfologia-opencv.git
cd morfologia-opencv
pip install -r requirements.txt
python demo_morfologia.py          # gera as figuras em resultados/
# ou abra demo_morfologia.ipynb no Jupyter
```

## Estrutura

```
demo_morfologia.py      # script principal (células # %% – roda direto ou vira notebook)
demo_morfologia.ipynb   # mesmo código, já executado
imagens/                # imagens de entrada (cameraman.png, texto.png)
resultados/             # figuras geradas pelo código
requirements.txt
```

## Integrantes
- Nome 1
- Nome 2

> As imagens `cameraman.png` e `texto.png` vêm do pacote de dados do scikit-image.
