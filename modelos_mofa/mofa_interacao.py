"""MOFA não linear aditivo com interações entre fatores (v2 do mofa_aditivo.py).

Modelo:

    y_nd = sum_k f_dk(z_nk) + sum_{(k,l) em P} g_dkl(z_nk, z_nl) + ruído

    f_dk(z)     = psi(z)^T u_dk                      (igual ao mofa_aditivo: w z + b2 h2(z) + b3 h3(z), 3 estados)
    g_dkl(a, b) = phi(a, b)^T v_dkl                  (termo de interação do par de fatores (k, l))

Base da interação (produtos dos mesmos Hermite normalizados, então são ortonormais sob z ~ N(0, 1)):

    "bilinear"   : phi = (z_k z_l)                               -> a inclinação em F_l muda linearmente com F_k
    "quadratica" : phi = (z_k z_l, h2(z_k) z_l, z_k h2(z_l))      -> a inclinação também pode mudar de forma curva

Priors (esparsas, para manter a interpretação):

    s_dkl ~ Bernoulli(eta_kl),  eta_kl ~ Beta(1, 99)              (cética: poucas features por par)
    v_dkl | s = 1 ~ N(0, diag(1 / (kappa_kl lambda_q))),  kappa_kl ~ Gama   (ARD por par: um par inteiro pode sumir)
    v_dkl | s = 0 = 0

O resto (u, alfa, rho, theta, pi, tau, q(z) na grade) é o mesmo do mofa_aditivo. A inferência é variacional
(campo médio por blocos: cada f_dk e cada g_dkl é um bloco) com os momentos EXATOS de E[phi phi^T], que agora
misturam fatores diferentes (z_k aparece em f_dk e em g_dkl). Com P vazio o algoritmo é idêntico ao mofa_aditivo.

Uso:

    import mofa_interacao as MI
    res = MI.mofa_interacao(visoes, K=10)                  # 1) MOFA linear  2) aditivo (v1)  3) liga as interações
    res["r2_interacao"]["rna"]                              # R^2 de cada par de fatores (ordem em res["pares"])
    MI.tabela_interacoes(res, visoes)                       # features com interação, coeficientes e leitura
    MI.figura_resumo_interacoes(res)
    MI.figura_interacao(res, visoes, "rna", "GENE", par=(0, 1))

Também aceita um ajuste do aditivo já feito (não refaz a v1):  MI.mofa_interacao(visoes, res_aditivo=res_v1)
Precisa do mofa.py e do mofa_aditivo.py na mesma pasta. O mofa_aditivo.py não é alterado.
"""
import time
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp, expit

import mofa_aditivo as MA
from mofa_aditivo import psi, Eg, Elog, LAMBDA, P, LOG2PI, _kl_gama, _kl_beta

BASES = {"bilinear": [(1, 1)], "quadratica": [(1, 1), (2, 1), (1, 2)]}
LAMBDA_INT = {(1, 1): 1.0, (2, 1): 2.0, (1, 2): 2.0}       # termos com h2 são encolhidos mais (como no aditivo)
NOMES_BASE = {(1, 1): "zk*zl", (2, 1): "h2(zk)*zl", (1, 2): "zk*h2(zl)"}


# ================================================================== estrutura dos termos
def _monta_termos(st):
    """Cada termo t é um produto de componentes da base estendida (1, z, h2, h3) de no máximo dois fatores.

    idx[t, k] = componente do fator k no termo (0 = não entra). Os 3K primeiros termos são os efeitos principais
    (bloco k = termos 3k, 3k+1, 3k+2, na mesma ordem do mofa_aditivo); depois Q termos por par de fatores.
    F, E = (fator, componente) dos dois "slots" de cada termo; o fator fictício K tem momento 1.
    """
    K, pares, base = st["K"], st["pares"], st["base_int"]
    Q = len(base)
    T = P * K + Q * len(pares)
    idx = np.zeros((T, K + 1), int)
    F = np.full((T, 2), K)
    E = np.zeros((T, 2), int)
    for k in range(K):
        for e in range(P):
            t = P * k + e
            idx[t, k] = e + 1; F[t, 0] = k; E[t, 0] = e + 1
    for p, (k, l) in enumerate(pares):
        for q, (a, b) in enumerate(base):
            t = P * K + Q * p + q
            idx[t, k], idx[t, l] = a, b
            F[t] = (k, l); E[t] = (a, b)
    st["termos"] = dict(T=T, Q=Q, idx=idx, F=F, E=E)


def _E4(st):
    """E4[n, k] = E[psi4 psi4^T] com psi4 = (1, z, h2, h3) sob q(z_nk); o fator fictício K vale 1."""
    N, K = st["N"], st["K"]
    E4 = np.ones((N, K + 1, 4, 4))
    E4[:, :K, 0, 1:] = st["m1"]
    E4[:, :K, 1:, 0] = st["m1"]
    E4[:, :K, 1:, 1:] = st["M2"]
    return E4


def _Fbar(E4, tr, excluir=None, linhas=None):
    """E[Phi_t(z_n)] (N x T), o produto das médias dos fatores do termo. excluir=k: omite o fator k do produto."""
    F, E = tr["F"], tr["E"]
    if linhas is not None:
        F, E = F[linhas], E[linhas]
    m1ext = E4[:, :, 0, :]
    out = np.ones((E4.shape[0], len(F)))
    for s in (0, 1):
        f, e = F[:, s].copy(), E[:, s].copy()
        if excluir is not None:
            e[f == excluir] = 0
        out *= m1ext[:, f, e]
    return out


def _momento2(E4, tr, A, B, excluir=None):
    """E[Phi_a Phi_b] (N x |A| x |B|) exato sob q(z) fatorada; excluir=k omite o fator k (usado na atualização de z_k)."""
    F, E, idx = tr["F"], tr["E"], tr["idx"]
    K = E4.shape[1] - 1
    Fa, Ea = F[A].copy(), E[A].copy()
    if excluir is not None:
        Fa[Fa == excluir] = K; Ea[F[A] == excluir] = 0
    R = np.ones((E4.shape[0], len(A), len(B)))
    idxB = idx[B]                                                        # |B| x (K+1)
    for s in (0, 1):                                                     # fatores dos termos-linha: momento conjunto
        f, e = Fa[:, s], Ea[:, s]
        R *= E4[:, f[:, None], e[:, None], idxB[:, f].T]
    m1ext = E4[:, :, 0, :]
    for s in (0, 1):                                                     # fatores só dos termos-coluna: média
        f, e = F[B, s], E[B, s]
        ja = (Fa[:, 0][:, None] == f[None]) | (Fa[:, 1][:, None] == f[None])
        if excluir is not None:
            ja = ja | (f == excluir)[None]
        R *= np.where(ja[None], 1.0, m1ext[:, f, e][:, None, :])
    return R


def _S_soma(E4, tr, bloco=256):
    """sum_n E[Phi_n Phi_n^T] (T x T), em pedaços de amostras para não estourar a memória."""
    T = tr["T"]
    todos = np.arange(T)
    S = np.zeros((T, T))
    for i in range(0, E4.shape[0], bloco):
        S += _momento2(E4[i:i + bloco], tr, todos, todos).sum(axis=0)
    return 0.5 * (S + S.T)


# ================================================================== momentos dos coeficientes
def EV(st, m):
    """E[v] (D x Pp x Q), mistura de s = 0 (v = 0) e s = 1."""
    return st["pv"][m][..., None] * st["mv"][m]


def EVV(st, m):
    mv = st["mv"][m]
    return st["pv"][m][..., None, None] * (mv[..., :, None] * mv[..., None, :] + st["Sv"][m])


def Theta(st, m):
    """Vetor de todos os coeficientes por feature, na ordem dos termos (D x T)."""
    D = st["Y"][m].shape[1]
    return np.hstack([MA.EU(st, m).reshape(D, -1), EV(st, m).reshape(D, -1)])


def P_int(st, m):
    """E[precisão a priori] dos coeficientes de interação: E[kappa_p] lambda_q (Pp x Q)."""
    lam = np.array([LAMBDA_INT[b] for b in st["base_int"]])
    return Eg(st["a_kappa"][m], st["b_kappa"][m])[:, None] * lam[None]


def _cov_blocos(st, m):
    """sum_d tau_d Cov_d(theta) (T x T), bloco diagonal (cada f_dk e cada g_dkl é um bloco independente)."""
    tr, K, Q = st["termos"], st["K"], st["termos"]["Q"]
    tau = Eg(st["a_tau"][m], st["b_tau"][m])
    C = np.zeros((tr["T"], tr["T"]))
    U, UU = MA.EU(st, m), MA.EUU(st, m)
    cu = np.einsum("d,dkij->kij", tau, UU - U[..., :, None] * U[..., None, :])
    for k in range(K):
        C[P * k:P * k + P, P * k:P * k + P] = cu[k]
    if st["pares"]:
        V, VV = EV(st, m), EVV(st, m)
        cv = np.einsum("d,dpij->pij", tau, VV - V[..., :, None] * V[..., None, :])
        for p in range(len(st["pares"])):
            a = P * K + Q * p
            C[a:a + Q, a:a + Q] = cv[p]
    return C


# ================================================================== inicialização
def inicializa(visoes, K=10, res_aditivo=None, pares=None, base="bilinear", max_fatores_interacao=8,
               limiar_produto=0.5, a0_eta=1.0, b0_eta=99.0, verbose=True, opcoes_aditivo=None):
    """Começa do ajuste do mofa_aditivo (v1). As interações começam desligadas (q(s) = 0).

    pares: None = todos os pares entre os `max_fatores_interacao` fatores mais fortes; ou lista de (k, l) (k começa em 0).
    limiar_produto: remove antes os fatores que são o produto z_k z_l de dois fatores mais fortes (R^2 > limiar),
                    o equivalente do efeito "harmônico" do aditivo para interações. Use 1.01 para não remover nenhum.
    """
    if res_aditivo is None:
        if verbose:
            print("ajustando o MOFA aditivo (v1) para o início ...")
        res_aditivo = MA.mofa_aditivo(visoes, K=K, verbose=verbose, **(opcoes_aditivo or {}))
    sa = res_aditivo["estado"]
    st = {c: (dict((m, np.array(v, copy=True)) for m, v in sa[c].items()) if isinstance(sa[c], dict)
              else (np.array(sa[c], copy=True) if isinstance(sa[c], np.ndarray) else sa[c])) for c in sa}
    st["Y"] = sa["Y"]
    st["pri"] = dict(sa["pri"], a0_eta=a0_eta, b0_eta=b0_eta)
    st["elbo_aditivo"] = float(res_aditivo["elbo"][-1])
    st["base_int"] = BASES[base] if isinstance(base, str) else list(base)
    st["pares"] = []
    _inicia_pares(st, [])
    _monta_termos(st)

    forca = np.max(np.vstack([res_aditivo["r2"][m] for m in st["visoes"]]), axis=0)
    removidos = fatores_produto(st["m1"][..., 0], forca, limiar_produto)
    if removidos:
        if verbose:
            for j, k, l, r in removidos:
                print(f"fator {j + 1} é o produto dos fatores {k + 1} e {l + 1} (R² = {r:.2f}): removido")
        manter = np.ones(st["K"], bool); manter[[j for j, *_ in removidos]] = False
        mapa = np.cumsum(manter) - 1
        if pares is not None:
            pares = [(int(mapa[k]), int(mapa[l])) for k, l in pares if manter[k] and manter[l]]
        forca = forca[manter]
        _seleciona(st, manter, np.zeros(0, bool))
    if pares is None:
        fortes = np.sort(np.argsort(-forca)[:max_fatores_interacao])
        pares = list(combinations(fortes.tolist(), 2))
    pares = [tuple(sorted((int(k), int(l)))) for k, l in pares if k != l]
    _inicia_pares(st, pares)
    _monta_termos(st)
    return st, res_aditivo


def _inicia_pares(st, pares):
    st["pares"] = list(pares)
    Pp, Q = len(pares), len(st["base_int"])
    for c in ("mv", "Sv", "pv", "a_kappa", "b_kappa", "a_eta", "b_eta"):
        st.setdefault(c, {})
    for m in st["visoes"]:
        D = st["Y"][m].shape[1]
        st["mv"][m] = np.zeros((D, Pp, Q))
        st["Sv"][m] = np.tile(np.eye(Q) * 1e-2, (D, Pp, 1, 1))
        st["pv"][m] = np.zeros((D, Pp))
        st["a_kappa"][m], st["b_kappa"][m] = np.ones(Pp), np.ones(Pp)
        st["a_eta"][m], st["b_eta"][m] = np.full(Pp, st["pri"]["a0_eta"]), np.full(Pp, st["pri"]["b0_eta"])


def fatores_produto(Z, forca, limiar):
    """Lista (j, k, l, R^2) dos fatores j cuja média E[z_j] é explicada por z_k z_l de dois fatores mais fortes."""
    ordem = np.argsort(-forca)
    mantidos, out = [], []
    Zs = (Z - Z.mean(0)) / Z.std(0)
    for j in ordem:
        melhor = (0.0, None, None)
        for k, l in combinations(mantidos, 2):
            x = Zs[:, k] * Zs[:, l]
            r = np.corrcoef(x, Zs[:, j])[0, 1] ** 2
            if r > melhor[0]:
                melhor = (r, k, l)
        if melhor[0] > limiar:
            out.append((int(j), int(melhor[1]), int(melhor[2]), float(melhor[0])))
        else:
            mantidos.append(j)
    return out


def _seleciona(st, manter_f, manter_p):
    """Mantém só os fatores em manter_f e os pares em manter_p (pares com fator removido também saem)."""
    manter_f = np.asarray(manter_f, bool)
    mapa = np.cumsum(manter_f) - 1
    if len(st["pares"]):
        manter_p = np.asarray(manter_p, bool) & np.array([manter_f[k] and manter_f[l] for k, l in st["pares"]])
    else:
        manter_p = np.zeros(0, bool)
    if not manter_f.all():
        st["m1"], st["M2"], st["Hz"] = st["m1"][:, manter_f], st["M2"][:, manter_f], st["Hz"][:, manter_f]
        for m in st["visoes"]:
            for c in ("mu1", "s21", "mu2", "S2", "p"):
                st[c][m] = st[c][m][:, manter_f]
            for c in ("a_alpha", "b_alpha", "a_rho", "b_rho", "a_theta", "b_theta", "a_pi", "b_pi"):
                st[c][m] = st[c][m][manter_f]
        st["K"] = int(manter_f.sum())
    for m in st["visoes"]:
        for c in ("mv", "Sv", "pv"):
            st[c][m] = st[c][m][:, manter_p]
        for c in ("a_kappa", "b_kappa", "a_eta", "b_eta"):
            st[c][m] = st[c][m][manter_p]
    st["pares"] = [(int(mapa[k]), int(mapa[l])) for (k, l), ok in zip(st["pares"], manter_p) if ok]
    _monta_termos(st)
    return int((~manter_f).sum()), int((~manter_p).sum())


# ================================================================== coeficientes e estados
def atualiza_coeficientes(st, m, Fb, S):
    """Coordenadas por bloco: cada f_dk (3 estados, como no aditivo) e cada g_dkl (2 estados), todas as features juntas.

    c_b = (Fbar^T Y)_b - sum_{b' != b} S_bb' E[theta_b'],  com S = sum_n E[Phi Phi^T] exato (inclui z_k z_l).
    """
    Y, K, tr = st["Y"][m], st["K"], st["termos"]
    Q = tr["Q"]
    tau = Eg(st["a_tau"][m], st["b_tau"][m])
    A = Fb.T @ Y                                                         # T x D
    Th = Theta(st, m)                                                    # D x T
    # --- efeitos principais (igual ao mofa_aditivo.atualiza_U, com S no lugar de G)
    Pd = MA.P_diag(st, m)
    elt = digamma(st["a_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
    el1t = digamma(st["b_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
    elp = digamma(st["a_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
    el1p = digamma(st["b_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
    for k in range(K):
        sl = slice(P * k, P * k + P)
        C = A[sl] - S[sl] @ Th.T + S[sl, sl] @ Th[:, sl].T               # 3 x D
        Sk = S[sl, sl]
        prec = np.diag(Pd[k])[None] + tau[:, None, None] * Sk[None]
        S2 = np.linalg.inv(prec)
        mu2 = np.einsum("dij,dj->di", S2, tau[:, None] * C.T)
        _, logdet_prec = np.linalg.slogdet(prec)
        L2 = elt[k] + elp[k] + 0.5 * (np.log(Pd[k]).sum() - logdet_prec) + 0.5 * np.einsum("di,dij,dj->d", mu2, prec, mu2)
        s21 = 1.0 / (Pd[k, 0] + tau * Sk[0, 0])
        mu1 = s21 * tau * C[0]
        L1 = elt[k] + el1p[k] + 0.5 * np.log(Pd[k, 0] * s21) + mu1 ** 2 / (2 * s21)
        L0 = np.full_like(L1, el1t[k])
        Ls = np.stack([L0, L1, L2], axis=-1)
        p = np.exp(Ls - logsumexp(Ls, axis=-1, keepdims=True))
        st["mu1"][m][:, k], st["s21"][m][:, k] = mu1, s21
        st["mu2"][m][:, k], st["S2"][m][:, k], st["p"][m][:, k] = mu2, S2, p
        Th[:, sl] = p[:, 2:3] * mu2
        Th[:, P * k] += p[:, 1] * mu1
    # --- interações: spike-and-slab por (feature, par)
    if not st["pares"]:
        return
    Pi = P_int(st, m)
    ele = digamma(st["a_eta"][m]) - digamma(st["a_eta"][m] + st["b_eta"][m])
    el1e = digamma(st["b_eta"][m]) - digamma(st["a_eta"][m] + st["b_eta"][m])
    for p_ in range(len(st["pares"])):
        a = P * K + Q * p_
        sl = slice(a, a + Q)
        C = A[sl] - S[sl] @ Th.T + S[sl, sl] @ Th[:, sl].T               # Q x D
        prec = np.diag(Pi[p_])[None] + tau[:, None, None] * S[sl, sl][None]
        Sv = np.linalg.inv(prec)
        mv = np.einsum("dij,dj->di", Sv, tau[:, None] * C.T)
        _, logdet_prec = np.linalg.slogdet(prec)
        L1 = ele[p_] + 0.5 * (np.log(Pi[p_]).sum() - logdet_prec) + 0.5 * np.einsum("di,dij,dj->d", mv, prec, mv)
        pv = expit(L1 - el1e[p_])
        st["mv"][m][:, p_], st["Sv"][m][:, p_], st["pv"][m][:, p_] = mv, Sv, pv
        Th[:, sl] = pv[:, None] * mv


def atualiza_hiper(st, m):
    MA.atualiza_hiper(st, m)                                             # alfa, rho, theta, pi: iguais ao aditivo
    if not st["pares"]:
        return
    pr, D, Q = st["pri"], st["Y"][m].shape[1], len(st["base_int"])
    lam = np.array([LAMBDA_INT[b] for b in st["base_int"]])
    pv, mv, Sv = st["pv"][m], st["mv"][m], st["Sv"][m]
    Ek = Eg(st["a_kappa"][m], st["b_kappa"][m])
    diag = mv ** 2 + np.diagonal(Sv, axis1=-2, axis2=-1)
    EvLv = (1 - pv) * Q / Ek + pv * (lam * diag).sum(-1)                # em s = 0, v segue a prior
    st["a_kappa"][m] = np.full(len(st["pares"]), pr["a0"] + D * Q / 2)
    st["b_kappa"][m] = pr["b0"] + 0.5 * EvLv.sum(0)
    st["a_eta"][m] = pr["a0_eta"] + pv.sum(0)
    st["b_eta"][m] = pr["b0_eta"] + (1 - pv).sum(0)


# ================================================================== fatores na grade
def _guarda_z(st, k, logq):
    MA._guarda_z(st, k, logq)


def _hH_fator(E4, tr, k, YT, G):
    """h_n (N x 4) e H_n (N x 4 x 4) de log q(z_nk = z) = -z^2/2 + psi4(z)^T h_n - 1/2 psi4(z)^T H_n psi4(z).

    YT = sum_m Y_m diag(tau) Theta_m (N x T),  G = sum_m sum_d tau_d E[theta_d theta_d^T] (T x T).
    """
    idx = tr["idx"]
    Ik = np.where(idx[:, k] > 0)[0]
    todos = np.arange(tr["T"])
    R1 = _Fbar(E4, tr, excluir=k)                                        # N x T, sem o fator k
    Oh = np.eye(4)[idx[:, k]]                                            # T x 4 (componente de z_k em cada termo)
    h = (YT * R1) @ Oh                                                   # N x 4
    R2 = _momento2(E4, tr, todos, Ik, excluir=k)                        # N x T x |Ik|
    Hs = np.einsum("nab,ab,ae,bf->nef", R2, G[:, Ik], Oh, Oh[Ik], optimize=True)
    H = np.zeros_like(Hs)
    H[:, :, 1:] = Hs[:, :, 1:]
    H[:, 1:, 0] = Hs[:, 0, 1:]
    return h, H


def atualiza_Z(st):
    tr, K = st["termos"], st["K"]
    YT, G = 0.0, 0.0
    for m in st["visoes"]:
        tau = Eg(st["a_tau"][m], st["b_tau"][m])
        Th = Theta(st, m)
        YT = YT + st["Y"][m] @ (tau[:, None] * Th)
        G = G + Th.T @ (tau[:, None] * Th) + _cov_blocos(st, m)
    zg = st["zg"]
    psig4 = np.column_stack([np.ones_like(zg), st["psig"]])
    E4 = _E4(st)
    for k in range(K):
        h, H = _hH_fator(E4, tr, k, YT, G)
        quad = np.einsum("ge,nef,gf->ng", psig4, H, psig4, optimize=True)
        logq = -0.5 * zg[None] ** 2 + h @ psig4.T - 0.5 * quad
        _guarda_z(st, k, logq)
        E4[:, k, 0, 1:] = st["m1"][:, k]; E4[:, k, 1:, 0] = st["m1"][:, k]; E4[:, k, 1:, 1:] = st["M2"][:, k]


# ================================================================== ruído e ELBO
def erro_quadratico_esperado(st, m, Fb, S):
    """sum_n E[(y_nd - mu_nd)^2] = ||Y - Fbar Theta^T||^2 + Theta (S - Fbar^T Fbar) Theta + tr(Cov_d S)."""
    Y = st["Y"][m]
    Th = Theta(st, m)
    R = Y - Fb @ Th.T
    extra = np.einsum("dt,ts,ds->d", Th, S - Fb.T @ Fb, Th, optimize=True)
    tr, K, Q = st["termos"], st["K"], st["termos"]["Q"]
    U, UU = MA.EU(st, m), MA.EUU(st, m)
    cov = 0.0
    for k in range(K):
        sl = slice(P * k, P * k + P)
        cov = cov + np.einsum("dij,ij->d", UU[:, k] - U[:, k, :, None] * U[:, k, None, :], S[sl, sl])
    if st["pares"]:
        V, VV = EV(st, m), EVV(st, m)
        for p_ in range(len(st["pares"])):
            a = P * K + Q * p_
            cov = cov + np.einsum("dij,ij->d", VV[:, p_] - V[:, p_, :, None] * V[:, p_, None, :], S[a:a + Q, a:a + Q])
    return (R ** 2).sum(0) + extra + cov


def atualiza_tau(st, m, Fb, S):
    sqe = erro_quadratico_esperado(st, m, Fb, S)
    st["a_tau"][m] = np.full_like(sqe, st["pri"]["a0"] + st["N"] / 2)
    st["b_tau"][m] = st["pri"]["b0"] + 0.5 * sqe


def elbo(st, Fb=None, S=None):
    if Fb is None:
        E4 = _E4(st); Fb, S = _Fbar(E4, st["termos"]), _S_soma(E4, st["termos"])
    pr = st["pri"]
    L = (-0.5 * LOG2PI - 0.5 * st["M2"][..., 0, 0] + st["Hz"]).sum()
    for m in st["visoes"]:
        a_t, b_t = st["a_tau"][m], st["b_tau"][m]
        L += (st["N"] / 2 * (Elog(a_t, b_t) - LOG2PI) - 0.5 * Eg(a_t, b_t) * erro_quadratico_esperado(st, m, Fb, S)).sum()
        # efeitos principais: mesmos termos do mofa_aditivo.elbo
        Pd = MA.P_diag(st, m)
        ElogP = np.column_stack([Elog(st["a_alpha"][m], st["b_alpha"][m]),
                                 Elog(st["a_rho"][m], st["b_rho"][m]) + np.log(LAMBDA[0]),
                                 Elog(st["a_rho"][m], st["b_rho"][m]) + np.log(LAMBDA[1])])
        comum = 0.5 * ElogP.sum(axis=1)[None]
        logPd = np.log(Pd)[None]
        mu2, S2 = st["mu2"][m], st["S2"][m]
        v0 = comum - 0.5 * logPd.sum(axis=-1)
        v1 = comum - 0.5 * Pd[None, :, 0] * (st["mu1"][m] ** 2 + st["s21"][m]) + 0.5 * np.log(st["s21"][m]) + 0.5 \
            - 0.5 * logPd[..., 1:].sum(axis=-1)
        diag2 = mu2 ** 2 + np.diagonal(S2, axis1=-2, axis2=-1)
        _, logdetS2 = np.linalg.slogdet(S2)
        v2 = comum - 0.5 * (Pd[None] * diag2).sum(axis=-1) + 0.5 * logdetS2 + 1.5
        p = st["p"][m]
        L += (p[..., 0] * v0 + p[..., 1] * v1 + p[..., 2] * v2).sum()
        elt = digamma(st["a_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        el1t = digamma(st["b_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        elp = digamma(st["a_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
        el1p = digamma(st["b_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
        pc = np.clip(p, 1e-300, 1)
        L += (p[..., 0] * el1t + p[..., 1] * (elt + el1p) + p[..., 2] * (elt + elp) - (p * np.log(pc)).sum(-1)).sum()
        L -= _kl_gama(st["a_alpha"][m], st["b_alpha"][m], pr["a0"], pr["b0"])
        L -= _kl_gama(st["a_rho"][m], st["b_rho"][m], pr["a0"], pr["b0"])
        L -= _kl_gama(a_t, b_t, pr["a0"], pr["b0"])
        L -= _kl_beta(st["a_theta"][m], st["b_theta"][m], pr["a0_theta"], pr["b0_theta"])
        L -= _kl_beta(st["a_pi"][m], st["b_pi"][m], pr["a0_pi"], pr["b0_pi"])
        # interações
        if st["pares"]:
            Q = len(st["base_int"])
            lam = np.array([LAMBDA_INT[b] for b in st["base_int"]])
            Pi = P_int(st, m)                                            # Pp x Q
            comum = 0.5 * (Elog(st["a_kappa"][m], st["b_kappa"][m])[:, None] + np.log(lam)[None]).sum(1)[None]
            w0 = comum - 0.5 * np.log(Pi).sum(1)[None]
            mv, Sv, pv = st["mv"][m], st["Sv"][m], st["pv"][m]
            diag = mv ** 2 + np.diagonal(Sv, axis1=-2, axis2=-1)
            _, logdetSv = np.linalg.slogdet(Sv)
            w1 = comum - 0.5 * (Pi[None] * diag).sum(-1) + 0.5 * logdetSv + Q / 2
            L += ((1 - pv) * w0 + pv * w1).sum()
            ele = digamma(st["a_eta"][m]) - digamma(st["a_eta"][m] + st["b_eta"][m])
            el1e = digamma(st["b_eta"][m]) - digamma(st["a_eta"][m] + st["b_eta"][m])
            pvc = np.clip(pv, 1e-300, 1 - 1e-16)
            L += (pv * ele + (1 - pv) * el1e - pv * np.log(pvc) - (1 - pv) * np.log1p(-pvc)).sum()
            L -= _kl_gama(st["a_kappa"][m], st["b_kappa"][m], pr["a0"], pr["b0"])
            L -= _kl_beta(st["a_eta"][m], st["b_eta"][m], pr["a0_eta"], pr["b0_eta"])
    return L


# ================================================================== variância explicada e poda
def r2_termos(st, termos, Fb=None):
    """R^2 de cada vista usando só os termos da lista (com as médias, como o r2 do mofa_aditivo)."""
    if Fb is None:
        Fb = _Fbar(_E4(st), st["termos"])
    termos = np.asarray(termos, int)
    out = {}
    for m in st["visoes"]:
        Y = st["Y"][m]
        pred = Fb[:, termos] @ Theta(st, m)[:, termos].T
        out[m] = 1 - ((Y - pred) ** 2).sum() / (Y ** 2).sum()
    return out


def r2_fatores(st, partes=(0, 1, 2), Fb=None):
    """R^2 de cada fator (só os efeitos principais) por vista. partes: 0 = linear; 1, 2 = curvos."""
    if Fb is None:
        Fb = _Fbar(_E4(st), st["termos"])
    out = {m: np.empty(st["K"]) for m in st["visoes"]}
    for k in range(st["K"]):
        r = r2_termos(st, [P * k + e for e in partes], Fb)
        for m in r:
            out[m][k] = r[m]
    return out


def r2_pares(st, Fb=None):
    """R^2 de cada termo de interação g_kl por vista (vetor com um valor por par)."""
    if Fb is None:
        Fb = _Fbar(_E4(st), st["termos"])
    K, Q = st["K"], st["termos"]["Q"]
    out = {m: np.empty(len(st["pares"])) for m in st["visoes"]}
    for p_ in range(len(st["pares"])):
        r = r2_termos(st, range(P * K + Q * p_, P * K + Q * (p_ + 1)), Fb)
        for m in r:
            out[m][p_] = r[m]
    return out


def poda(st, limiar_fator, limiar_par, podar_pares, Fb):
    Rf = r2_fatores(st, Fb=Fb)
    manter_f = np.max(np.vstack([Rf[m] for m in st["visoes"]]), axis=0) >= limiar_fator
    if podar_pares and st["pares"]:
        Rp = r2_pares(st, Fb)
        manter_p = np.max(np.vstack([Rp[m] for m in st["visoes"]]), axis=0) >= limiar_par
    else:
        manter_p = np.ones(len(st["pares"]), bool)
    if manter_f.all() and manter_p.all():
        return 0, 0
    return _seleciona(st, manter_f, manter_p)


# ================================================================== o algoritmo
def iteracao(st):
    tr = st["termos"]
    E4 = _E4(st)
    Fb, S = _Fbar(E4, tr), _S_soma(E4, tr)
    for m in st["visoes"]:
        atualiza_coeficientes(st, m, Fb, S)
        atualiza_hiper(st, m)
    atualiza_Z(st)
    E4 = _E4(st)
    Fb, S = _Fbar(E4, tr), _S_soma(E4, tr)
    for m in st["visoes"]:
        atualiza_tau(st, m, Fb, S)
    return Fb, S


def mofa_interacao(visoes, K=10, res_aditivo=None, pares=None, base="bilinear", max_fatores_interacao=8,
                   max_iter=1000, tol=1e-6, limiar_poda=0.005, limiar_poda_par=0.002, inicio_poda_pares=20,
                   verbose=True, mostrar_a_cada=25, opcoes_aditivo=None, **opcoes):
    """Ajusta a v2. limiar_poda_par: um par sai quando o R^2 da interação fica abaixo disso em todas as vistas
    (só depois de `inicio_poda_pares` iterações, porque as interações começam desligadas)."""
    t0 = time.time()
    st, res_ad = inicializa(visoes, K, res_aditivo=res_aditivo, pares=pares, base=base,
                            max_fatores_interacao=max_fatores_interacao, verbose=verbose,
                            opcoes_aditivo=opcoes_aditivo, **opcoes)
    if verbose:
        print(f"início aditivo pronto (K = {st['K']}, {len(st['pares'])} pares candidatos); ligando as interações ...")
    hist = []
    for it in range(max_iter):
        Fb, S = iteracao(st)
        fr, pr_ = poda(st, limiar_poda, limiar_poda_par, it + 1 >= inicio_poda_pares, Fb)
        if (fr or pr_) and verbose:
            print(f"iter {it + 1}: {fr} fator(es) e {pr_} par(es) podados, K = {st['K']}, {len(st['pares'])} pares")
        hist.append(elbo(st))
        if verbose and (it + 1) % mostrar_a_cada == 0:
            print(f"iter {it + 1}: ELBO = {hist[-1]:.1f}, K = {st['K']}, pares = {len(st['pares'])}, {time.time() - t0:.0f} s")
        if it > max(inicio_poda_pares, 1) and abs(hist[-1] - hist[-2]) / abs(hist[-2]) < tol:
            break
    if verbose:
        print(f"convergiu em {it + 1} iterações ({time.time() - t0:.0f} s), ELBO = {hist[-1]:.1f} "
              f"(aditivo: {st['elbo_aditivo']:.1f}), K = {st['K']}, pares = {len(st['pares'])}")
    out = resultado(st, np.array(hist))
    out["aditivo"] = res_ad
    return out


def resultado(st, hist):
    Fb = _Fbar(_E4(st), st["termos"])
    r_tot, r_lin = r2_fatores(st, Fb=Fb), r2_fatores(st, partes=(0,), Fb=Fb)
    vs = st["visoes"]
    return {
        "Z": st["m1"][..., 0],
        "W": {m: MA.EU(st, m)[..., 0] for m in vs},
        "U": {m: MA.EU(st, m) for m in vs},
        "estados": {m: st["p"][m] for m in vs},
        "gamma": {m: 1 - st["p"][m][..., 0] for m in vs},
        "p_nao_linear": {m: st["p"][m][..., 2] for m in vs},
        "pares": list(st["pares"]),
        "base_interacao": [NOMES_BASE[b] for b in st["base_int"]],
        "V": {m: EV(st, m) for m in vs},                                    # D x Pp x Q
        "p_interacao": {m: st["pv"][m] for m in vs},                       # D x Pp, q(s = 1)
        "eta": {m: st["a_eta"][m] / (st["a_eta"][m] + st["b_eta"][m]) for m in vs},
        "kappa": {m: Eg(st["a_kappa"][m], st["b_kappa"][m]) for m in vs},
        "alpha": {m: Eg(st["a_alpha"][m], st["b_alpha"][m]) for m in vs},
        "rho": {m: Eg(st["a_rho"][m], st["b_rho"][m]) for m in vs},
        "pi": {m: st["a_pi"][m] / (st["a_pi"][m] + st["b_pi"][m]) for m in vs},
        "tau": {m: Eg(st["a_tau"][m], st["b_tau"][m]) for m in vs},
        "r2": r_tot, "r2_linear": r_lin, "r2_nao_linear": {m: r_tot[m] - r_lin[m] for m in vs},
        "r2_interacao": r2_pares(st, Fb),
        "r2_modelo": r2_termos(st, np.arange(st["termos"]["T"]), Fb),
        "r2_modelo_sem_interacao": r2_termos(st, np.arange(P * st["K"]), Fb),
        "elbo": hist, "elbo_aditivo": st["elbo_aditivo"], "estado": st,
    }


# ================================================================== interpretação
def nome_par(res, p_):
    k, l = res["pares"][p_]
    return f"F{k + 1}xF{l + 1}"


def _acha_par(res, par):
    if isinstance(par, (int, np.integer)):
        return int(par)
    alvo = tuple(sorted(par))
    return res["pares"].index(alvo)


def tabela_interacoes(res, visoes=None, limiar=0.5):
    """Features com q(interação) > limiar, com a leitura da interação.

    Para a base bilinear, g = v z_k z_l: a inclinação da feature em F_l é w_l + v z_k. As colunas
    incl_Fl_com_Fk_baixo / alto mostram essa inclinação com z_k = -1 e +1 (um desvio-padrão), e "leitura" diz se o
    efeito de F_l fica mais forte (reforça), mais fraco (atenua) ou troca de sinal (inverte) quando F_k sobe.
    r2_ganho = R^2 da feature com a interação menos R^2 só com os efeitos principais.
    """
    st, linhas = res["estado"], []
    E4 = _E4(st); Fb = _Fbar(E4, st["termos"])
    K, Q = st["K"], st["termos"]["Q"]
    for m in st["visoes"]:
        Y = st["Y"][m]
        Th = Theta(st, m)
        nomes = MA._nomes(visoes or {}, m, Y.shape[1])
        U = MA.EU(st, m)
        pred_princ = Fb[:, :P * K] @ Th[:, :P * K].T
        sqd = (Y ** 2).sum(0)
        for p_, (k, l) in enumerate(st["pares"]):
            sl = slice(P * K + Q * p_, P * K + Q * (p_ + 1))
            for d in np.where(st["pv"][m][:, p_] > limiar)[0]:
                v = EV(st, m)[d, p_]
                pred = pred_princ[:, d] + Fb[:, sl] @ v
                r2_p = 1 - ((Y[:, d] - pred_princ[:, d]) ** 2).sum() / sqd[d]
                r2_t = 1 - ((Y[:, d] - pred) ** 2).sum() / sqd[d]
                wl, wk = U[d, l, 0], U[d, k, 0]
                baixo, alto = wl - v[0], wl + v[0]
                if np.sign(baixo) != np.sign(alto):
                    leitura = f"efeito de F{l + 1} inverte de sinal com F{k + 1}"
                elif abs(alto) > abs(baixo):
                    leitura = f"efeito de F{l + 1} reforça quando F{k + 1} sobe"
                else:
                    leitura = f"efeito de F{l + 1} atenua quando F{k + 1} sobe"
                linha = dict(omica=m, par=f"F{k + 1}xF{l + 1}", feature=nomes[d], p_interacao=st["pv"][m][d, p_],
                             w_Fk=wk, w_Fl=wl)
                for q, b in enumerate(st["base_int"]):
                    linha[f"v_{NOMES_BASE[b]}"] = v[q]
                linha.update(incl_Fl_com_Fk_baixo=baixo, incl_Fl_com_Fk_alto=alto, r2_principais=r2_p,
                             r2_com_interacao=r2_t, r2_ganho=r2_t - r2_p, leitura=leitura)
                linhas.append(linha)
    df = pd.DataFrame(linhas)
    return df.sort_values(["omica", "par", "r2_ganho"], ascending=[True, True, False]).reset_index(drop=True) if len(df) else df


def figura_resumo_interacoes(res, arquivo=None):
    """Por vista, uma matriz K x K: diagonal = R^2 dos efeitos principais do fator; acima da diagonal = R^2 da
    interação do par; abaixo = E[eta] (fração de features com interação no par). Mais a ELBO."""
    import matplotlib.pyplot as plt
    st = res["estado"]
    vs, K = st["visoes"], st["K"]
    fig, ax = plt.subplots(1, len(vs) + 1, figsize=(4.6 * (len(vs) + 1), 4.2))
    ax[0].plot(np.arange(1, len(res["elbo"]) + 1), res["elbo"], color="#2a6fdb", lw=2)
    ax[0].axhline(res["elbo_aditivo"], color="#9aa0a6", ls="--", lw=1, label="aditivo (v1)")
    ax[0].set(title="ELBO (fase com interações)", xlabel="iteração")
    ax[0].legend(frameon=False, fontsize=8)
    for a, m in zip(ax[1:], vs):
        M = np.full((K, K), np.nan)
        M[np.arange(K), np.arange(K)] = res["r2"][m]
        for p_, (k, l) in enumerate(res["pares"]):
            M[k, l] = res["r2_interacao"][m][p_]
        vmax = max(np.nanmax(M), 1e-3)
        im = a.imshow(M, cmap="Oranges", vmin=0, vmax=vmax)
        for p_, (k, l) in enumerate(res["pares"]):
            a.text(k, l, f"{res['eta'][m][p_]:.2f}", ha="center", va="center", fontsize=7, color="#1a73e8")
        for i in range(K):
            for j in range(K):
                if not np.isnan(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                           color="white" if M[i, j] > 0.6 * vmax else "black")
        a.set(title=f"{m}: R² (diag = fator, acima = par)\nazul abaixo = E[η]", xticks=range(K), yticks=range(K),
              xticklabels=[f"F{k + 1}" for k in range(K)], yticklabels=[f"F{k + 1}" for k in range(K)])
        a.title.set_fontsize(9)
        fig.colorbar(im, ax=a, fraction=0.04)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


def figura_interacao(res, visoes, m, feature, par, arquivo=None):
    """Uma feature num par (k, l): (1) superfície E[f_k(z_k) + f_l(z_l) + g_kl(z_k, z_l)];
    (2) resíduo parcial contra F_l com as amostras coloridas pelo tercil de F_k e a reta/curva prevista em cada tercil
    (se as linhas têm inclinações diferentes, é interação); (3) o mesmo trocando os papéis de F_k e F_l."""
    import matplotlib.pyplot as plt
    st = res["estado"]
    p_ = _acha_par(res, par)
    k, l = st["pares"][p_]
    nomes = MA._nomes(visoes or {}, m, st["Y"][m].shape[1])
    d = int(np.where(nomes == feature)[0][0]) if isinstance(feature, str) else int(feature)
    K, Q, tr = st["K"], st["termos"]["Q"], st["termos"]
    E4 = _E4(st); Fb = _Fbar(E4, tr)
    th = Theta(st, m)[d]
    sl = list(range(P * k, P * k + P)) + list(range(P * l, P * l + P)) + list(range(P * K + Q * p_, P * K + Q * (p_ + 1)))
    resto = np.setdiff1d(np.arange(tr["T"]), sl)
    y = st["Y"][m][:, d] - Fb[:, resto] @ th[resto]                    # resíduo parcial: tira os outros termos

    def g(zk, zl):
        out = psi(zk) @ th[P * k:P * k + P] + psi(zl) @ th[P * l:P * l + P]
        pk, pl = np.column_stack([np.ones_like(zk), psi(zk)]), np.column_stack([np.ones_like(zl), psi(zl)])
        for q, (a, b) in enumerate(st["base_int"]):
            out = out + th[P * K + Q * p_ + q] * pk[:, a] * pl[:, b]
        return out

    zk, zl = st["m1"][:, k, 0], st["m1"][:, l, 0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    gk = np.linspace(np.percentile(zk, 2), np.percentile(zk, 98), 60)
    gl = np.linspace(np.percentile(zl, 2), np.percentile(zl, 98), 60)
    A, B = np.meshgrid(gk, gl)
    Zs = g(A.ravel(), B.ravel()).reshape(A.shape)
    im = ax[0].contourf(A, B, Zs, levels=20, cmap="RdBu_r")
    ax[0].scatter(zk, zl, s=4, color="black", alpha=0.3)
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    ax[0].set(xlabel=f"F{k + 1}", ylabel=f"F{l + 1}", title=f"{nomes[d]}: efeito conjunto previsto")
    cores = ["#1a73e8", "#9aa0a6", "#d93025"]
    for a, (zx, zc, fx, fc, inv) in zip(ax[1:], [(zl, zk, l, k, False), (zk, zl, k, l, True)]):
        cortes = np.percentile(zc, [33.3, 66.7])
        grupo = np.digitize(zc, cortes)
        gx = np.linspace(np.percentile(zx, 2), np.percentile(zx, 98), 80)
        for i, (cor, rot) in enumerate(zip(cores, ["baixo", "médio", "alto"])):
            sel = grupo == i
            a.scatter(zx[sel], y[sel], s=7, color=cor, alpha=0.5)
            c = np.full_like(gx, zc[sel].mean())
            curva = g(c, gx) if not inv else g(gx, c)
            a.plot(gx, curva, color=cor, lw=2, label=f"F{fc + 1} {rot} (z = {zc[sel].mean():.1f})")
        a.set(xlabel=f"F{fx + 1} (E[z])", ylabel="resíduo parcial", title=f"efeito de F{fx + 1} por tercil de F{fc + 1}")
        a.legend(frameon=False, fontsize=7)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{m}, {nomes[d]}, par F{k + 1}xF{l + 1}: q(interação) = {st['pv'][m][d, p_]:.2f}", fontsize=10, color="#5f6368")
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


# ================================================================== validação cruzada: linear x aditivo x interações
def _infere_Z(st, Ynovo, varreduras=30):
    """q(z) de amostras novas usando só as vistas em Ynovo, com coeficientes e tau do ajuste fixos. Devolve E4."""
    tr, K = st["termos"], st["K"]
    N = next(iter(Ynovo.values())).shape[0]
    zg = st["zg"]; psig = st["psig"]
    psig4 = np.column_stack([np.ones_like(zg), psig])
    YT, G = 0.0, 0.0
    for m, y in Ynovo.items():
        tau = Eg(st["a_tau"][m], st["b_tau"][m])
        Th = Theta(st, m)
        YT = YT + y @ (tau[:, None] * Th)
        G = G + Th.T @ (tau[:, None] * Th) + _cov_blocos(st, m)
    E4 = np.ones((N, K + 1, 4, 4))
    w0 = np.exp(-0.5 * zg ** 2); w0 /= w0.sum()
    E4[:, :K, 0, 1:] = w0 @ psig
    E4[:, :K, 1:, 0] = w0 @ psig
    E4[:, :K, 1:, 1:] = np.einsum("g,gi,gj->ij", w0, psig, psig)
    for _ in range(varreduras):
        for k in range(K):
            h, H = _hH_fator(E4, tr, k, YT, G)
            quad = np.einsum("ge,nef,gf->ng", psig4, H, psig4, optimize=True)
            logq = -0.5 * zg[None] ** 2 + h @ psig4.T - 0.5 * quad
            w = np.exp(logq - logsumexp(logq, axis=1, keepdims=True))
            m1 = w @ psig
            E4[:, k, 0, 1:] = m1; E4[:, k, 1:, 0] = m1
            E4[:, k, 1:, 1:] = np.einsum("ng,gi,gj->nij", w, psig, psig)
    return E4


def preve(res, Ynovo, alvo, varreduras=30):
    """Prevê a vista `alvo` de amostras novas a partir das vistas em Ynovo (dicionário vista -> N x D centrado)."""
    st = res["estado"]
    E4 = _infere_Z(st, Ynovo, varreduras)
    return _Fbar(E4, st["termos"]) @ Theta(st, alvo).T


def validacao_cruzada(visoes, K=10, folds=5, semente=0, max_iter=3000, opcoes_interacao=None, **kw):
    """Compara MOFA linear, aditivo (v1) e com interações (v2) fora da amostra.

    Em cada fold: ajusta os três no treino (a v2 começa do aditivo do mesmo fold); nas amostras de teste esconde uma
    vista inteira, infere Z com as outras e prevê a vista escondida. DataFrame com o R^2 de teste por fold, modelo e vista.
    """
    import mofa as M
    V = {m: (v.values if isinstance(v, pd.DataFrame) else np.asarray(v)).astype(float)
         for m, v in (visoes.items() if isinstance(visoes, dict) else enumerate(visoes))}
    N = next(iter(V.values())).shape[0]
    idx = np.random.default_rng(semente).permutation(N)
    linhas = []
    for f, te in enumerate(np.array_split(idx, folds)):
        tr = np.setdiff1d(idx, te)
        mu = {m: v[tr].mean(0) for m, v in V.items()}
        Vtr = {m: v[tr] - mu[m] for m, v in V.items()}
        Vte = {m: v[te] - mu[m] for m, v in V.items()}
        lin = M.mofa(Vtr, K=K, max_iter=max_iter, verbose=False)
        ad = MA.mofa_aditivo(Vtr, K=K, max_iter=max_iter, verbose=False, **kw)
        it = mofa_interacao(Vtr, K=K, res_aditivo=ad, max_iter=max_iter, verbose=False, **(opcoes_interacao or {}))
        for alvo in V:
            obs = {m: Vte[m] for m in V if m != alvo}
            y = Vte[alvo]
            for nome, par in [("linear", MA._params_linear(lin)), ("aditivo", MA._params_aditivo(ad))]:
                m1 = MA._infere_Z(par, obs)
                pred = np.einsum("nki,dki->nd", m1, par[alvo][0])
                linhas.append(dict(fold=f + 1, modelo=nome, vista=alvo, r2=1 - ((y - pred) ** 2).sum() / (y ** 2).sum()))
            pred = preve(it, obs, alvo)
            linhas.append(dict(fold=f + 1, modelo="interacao", vista=alvo, r2=1 - ((y - pred) ** 2).sum() / (y ** 2).sum()))
        print(f"fold {f + 1}/{folds} pronto", flush=True)
    return pd.DataFrame(linhas)
