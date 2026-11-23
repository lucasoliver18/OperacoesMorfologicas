"""
Testes automatizados de demo_morfologia.py.

Como rodar (na pasta do projeto):
    pip install -r requirements.txt pytest
    python -m pytest -v

O script inteiro é executado uma única vez numa pasta temporária (as figuras de
resultados/ do projeto não são sobrescritas) e os testes inspecionam as funções
e variáveis que ele cria.
"""
import importlib.util
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janelas: plt.show() não bloqueia
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = [
    "01_cena.png",
    "02_erosao_dilatacao.png",
    "03_abertura_fechamento_gradiente.png",
    "04_limpeza_ruido.png",
    "05_tamanho_kernel.png",
    "06_gradiente_tamanho.png",
    "07_formato_kernel.png",
    "08_iteracoes.png",
    "09_imagem_real.png",
    "10_texto.png",
]
FORMAS = {"rect": cv2.MORPH_RECT, "elipse": cv2.MORPH_ELLIPSE, "cruz": cv2.MORPH_CROSS}

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


# --------------------------------------------------------------------------- #
# Fixtures e utilitários
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def execucao(tmp_path_factory):
    """Executa demo_morfologia.py do início ao fim numa pasta temporária."""
    pasta = tmp_path_factory.mktemp("execucao")
    shutil.copytree(RAIZ / "imagens", pasta / "imagens")
    cwd = os.getcwd()
    os.chdir(pasta)
    try:
        spec = importlib.util.spec_from_file_location("demo_morfologia", RAIZ / "demo_morfologia.py")
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
    finally:
        os.chdir(cwd)
        plt.close("all")
    return modulo, pasta


@pytest.fixture(scope="session")
def dm(execucao):
    return execucao[0]


def kernel(forma, t):
    return cv2.getStructuringElement(FORMAS[forma], (t, t))


def imagem_teste(tipo, semente=0, shape=(60, 80)):
    rng = np.random.default_rng(semente)
    if tipo == "binaria":
        return (rng.random(shape) < 0.5).astype(np.uint8) * 255
    return rng.integers(0, 256, shape, dtype=np.uint8)


def abrir(img, k, **kw):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, k, **kw)


def fechar(img, k, **kw):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, **kw)


PARAMS_KERNEL = pytest.mark.parametrize("forma", list(FORMAS))
PARAMS_TAMANHO = pytest.mark.parametrize("t", [3, 5, 7, 9])
PARAMS_TIPO = pytest.mark.parametrize("tipo", ["binaria", "cinza"])


# --------------------------------------------------------------------------- #
# 1. Execução completa (smoke test)
# --------------------------------------------------------------------------- #
def test_script_gera_todas_as_figuras(execucao):
    _, pasta = execucao
    for nome in FIGURAS:
        arq = pasta / "resultados" / nome
        assert arq.exists(), f"figura não gerada: {nome}"
        assert arq.stat().st_size > 0, f"figura vazia: {nome}"


def test_imagens_reais_carregadas(dm):
    # cv2.imread devolve None (sem erro) se o caminho estiver errado
    for img in (dm.foto, dm.texto):
        assert img is not None
        assert img.ndim == 2 and img.dtype == np.uint8


# --------------------------------------------------------------------------- #
# 2. Cena sintética e ruído
# --------------------------------------------------------------------------- #
def test_cena_limpa_binaria(dm):
    cena = dm.criar_cena_limpa()
    assert cena.shape == (360, 540) and cena.dtype == np.uint8
    assert set(np.unique(cena)) == {0, 255}


def test_ruido_reprodutivel(dm):
    a = dm.adicionar_ruido(dm.limpa, semente=7)
    b = dm.adicionar_ruido(dm.limpa, semente=7)
    c = dm.adicionar_ruido(dm.limpa, semente=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_ruido_nao_altera_a_entrada(dm):
    copia = dm.limpa.copy()
    dm.adicionar_ruido(dm.limpa, p_sal=0.2, p_pimenta=0.2)
    assert np.array_equal(dm.limpa, copia)


def test_sal_so_no_fundo(dm):
    ruidosa = dm.adicionar_ruido(dm.limpa, p_sal=0.05, p_pimenta=0)
    fundo, objeto = dm.limpa == 0, dm.limpa == 255
    assert np.array_equal(ruidosa[objeto], dm.limpa[objeto])        # objetos intactos
    frac = np.count_nonzero(ruidosa[fundo]) / np.count_nonzero(fundo)
    assert frac == pytest.approx(0.05, abs=0.005)


def test_pimenta_so_nos_objetos(dm):
    ruidosa = dm.adicionar_ruido(dm.limpa, p_sal=0, p_pimenta=0.05)
    fundo, objeto = dm.limpa == 0, dm.limpa == 255
    assert np.array_equal(ruidosa[fundo], dm.limpa[fundo])          # fundo intacto
    frac = np.count_nonzero(ruidosa[objeto] == 0) / np.count_nonzero(objeto)
    assert frac == pytest.approx(0.05, abs=0.01)


# --------------------------------------------------------------------------- #
# 3. Erosão / dilatação manuais == OpenCV
# --------------------------------------------------------------------------- #
@PARAMS_KERNEL
@PARAMS_TAMANHO
@PARAMS_TIPO
def test_erosao_manual_igual_opencv(dm, forma, t, tipo):
    img, k = imagem_teste(tipo), kernel(forma, t)
    assert np.array_equal(dm.erosao_manual(img, k), cv2.erode(img, k))


@PARAMS_KERNEL
@PARAMS_TAMANHO
@PARAMS_TIPO
def test_dilatacao_manual_igual_opencv(dm, forma, t, tipo):
    img, k = imagem_teste(tipo), kernel(forma, t)
    assert np.array_equal(dm.dilatacao_manual(img, k), cv2.dilate(img, k))


@pytest.mark.parametrize(
    "k",
    [
        np.ones((3, 5), np.uint8),                                        # retangular
        np.ones((4, 4), np.uint8),                                        # tamanho par
        (np.random.default_rng(1).random((5, 5)) < 0.5).astype(np.uint8),  # assimétrico
    ],
    ids=["3x5", "4x4", "assimetrico"],
)
def test_manual_com_kernels_irregulares(dm, k):
    img = imagem_teste("cinza")
    assert np.array_equal(dm.erosao_manual(img, k), cv2.erode(img, k))
    assert np.array_equal(dm.dilatacao_manual(img, k), cv2.dilate(img, k))


def test_manual_preserva_shape_e_dtype(dm):
    img = imagem_teste("cinza", shape=(17, 23))
    for f in (dm.erosao_manual, dm.dilatacao_manual):
        out = f(img, kernel("rect", 5))
        assert out.shape == img.shape and out.dtype == img.dtype


# --------------------------------------------------------------------------- #
# 4. Definições das operações compostas
# --------------------------------------------------------------------------- #
@PARAMS_KERNEL
@pytest.mark.parametrize("t", [3, 5, 9])
def test_definicoes_compostas(dm, forma, t):
    A, k = dm.ruidosa, kernel(forma, t)
    ero, dil = dm.erosao_manual(A, k), dm.dilatacao_manual(A, k)
    assert np.array_equal(abrir(A, k), dm.dilatacao_manual(ero, k))     # (A ⊖ B) ⊕ B
    assert np.array_equal(fechar(A, k), dm.erosao_manual(dil, k))       # (A ⊕ B) ⊖ B
    assert np.array_equal(cv2.morphologyEx(A, cv2.MORPH_GRADIENT, k), dil - ero)


def test_variaveis_do_script_consistentes(dm):
    kp = dm.kp
    assert np.array_equal(dm.abertura, cv2.dilate(cv2.erode(dm.ruidosa, kp), kp))
    assert np.array_equal(dm.fechamento, cv2.erode(cv2.dilate(dm.ruidosa, kp), kp))
    assert np.array_equal(dm.gradiente, cv2.subtract(cv2.dilate(dm.limpa, kp), cv2.erode(dm.limpa, kp)))


# --------------------------------------------------------------------------- #
# 5. Propriedades matemáticas (valem para qualquer imagem)
# --------------------------------------------------------------------------- #
@PARAMS_KERNEL
@pytest.mark.parametrize("t", [3, 5, 7])
@PARAMS_TIPO
def test_propriedades(forma, t, tipo):
    A, k = imagem_teste(tipo, semente=t), kernel(forma, t)
    ab, fe = abrir(A, k), fechar(A, k)

    assert np.all(cv2.erode(A, k) <= A) and np.all(A <= cv2.dilate(A, k))
    assert np.all(ab <= A), "abertura deve ser anti-extensiva"
    assert np.all(fe >= A), "fechamento deve ser extensivo"
    assert np.array_equal(abrir(ab, k), ab), "abertura deve ser idempotente"
    assert np.array_equal(fechar(fe, k), fe), "fechamento deve ser idempotente"
    assert np.array_equal(fe, 255 - abrir(255 - A, k)), "dualidade abertura/fechamento"


def test_gradiente_zero_em_regiao_uniforme():
    A = np.full((30, 30), 120, np.uint8)
    assert not cv2.morphologyEx(A, cv2.MORPH_GRADIENT, kernel("elipse", 5)).any()


def test_abertura_remove_sal_e_fechamento_preenche_pimenta():
    img = np.zeros((21, 21), np.uint8)
    img[5:16, 5:16] = 255   # objeto 11x11
    img[2, 2] = 255         # ponto de sal isolado
    img[10, 10] = 0         # ponto de pimenta isolado
    k = kernel("rect", 3)

    ab, fe = abrir(img, k), fechar(img, k)
    assert ab[2, 2] == 0                                   # abertura remove o sal...
    assert np.count_nonzero(ab[5:16, 5:16]) == 11 * 11 - 1  # ...sem encolher o objeto
    assert fe[10, 10] == 255                               # fechamento preenche a pimenta
    assert fe[2, 2] == 255                                 # ...mas não remove o sal


# --------------------------------------------------------------------------- #
# 6. Resultados apresentados no relatório
# --------------------------------------------------------------------------- #
def test_limpeza_reduz_mais_de_90_por_cento_do_erro(dm):
    assert dm.erro_fa < 0.10 * dm.erro_antes


def test_ordem_das_operacoes_importa(dm):
    assert not np.array_equal(dm.limpeza, dm.limpeza_inv)
    assert dm.erro_fa < dm.erro_af  # fechamento -> abertura é melhor nesta cena


def test_kernel_grande_aumenta_o_erro(dm):
    def erro(t):
        k = kernel("elipse", t)
        return np.count_nonzero(abrir(fechar(dm.ruidosa, k), k) != dm.limpa)

    assert erro(3) < erro(9) < erro(15)


def test_kernel_grande_apaga_linha_fina(dm):
    linha = (slice(35, 185), slice(396, 405))  # linha vertical de 3 px em x=400
    total = np.count_nonzero(dm.limpa[linha])
    assert np.count_nonzero(abrir(dm.limpa, kernel("elipse", 3))[linha]) == total
    assert np.count_nonzero(abrir(dm.limpa, kernel("elipse", 9))[linha]) == 0


def test_gradiente_engrossa_com_o_kernel(dm):
    pixels = [np.count_nonzero(g) for g in dm.limpa_grad]  # kernels 3, 7, 15
    assert pixels == sorted(pixels) and len(set(pixels)) == len(pixels)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_iteracoes_equivalem_a_kernel_maior(dm, n):
    # Com kernel retangular a equivalência é exata: n iterações de 3x3 == (2n+1)x(2n+1)
    assert np.array_equal(
        abrir(dm.ruidosa, kernel("rect", 3), iterations=n),
        abrir(dm.ruidosa, kernel("rect", 2 * n + 1)),
    )
