"""MOFA aditivo: fatoração bayesiana não linear com curvas por feature (base de Hermite He_2, He_3; B = 2).

Modelo (documento "MOFA não linear aditivo", seções 2 a 4):

    y_nd = sum_k f_dk(z_nk) + ruído,      f_dk(z) = a_dk [ w_dk z + c_dk (b2_dk h2(z) + b3_dk h3(z)) ]
    h2(z) = (z^2 - 1)/sqrt(2),   h3(z) = (z^3 - 3z)/sqrt(6)        (Hermite normalizados, ortogonais sob N(0, 1))

Cada (feature d, fator k) está em um de três estados: 0 = não associado, 1 = linear, 2 = não linear.
Priors: w ~ N(0, 1/alfa_k), b_b ~ N(0, 1/(rho_k lambda_b)) com lambda = (2, 6) = curvatura média b(b-1),
P(associado) = theta_k ~ Beta, P(não linear | associado) = pi_k ~ Beta (cética: Beta(1, 9)),
alfa, rho, tau ~ Gama. z_nk ~ N(0, 1), com q(z_nk) calculada numa grade (não é suposta gaussiana).

Uso (mesmo formato do mofa.py: pacientes x features, mesmos pacientes na mesma ordem):

    import mofa_aditivo as MA
    visoes = {"rna": dados_rna_filtrado.T, "met": dados_metilacao_filtrado.T, "cnv": dados_cnv_filtrado.T}
    res = MA.mofa_aditivo(visoes, K=15)
    res["estados"]["cnv"]        # D x K x 3: q(estado) = (não associado, linear, não linear)
    res["r2_linear"], res["r2_nao_linear"]    # vistas x fatores
    MA.tabela_formas(res, visoes)             # features não lineares, com forma da curva
    MA.figura_curvas(res, visoes, "cnv", fator=0)

Precisa do mofa.py na mesma pasta (é usado para o início linear).
"""
import time
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, betaln, logsumexp

import mofa as M

LOG2PI = np.log(2 * np.pi)
LAMBDA = np.array([2.0, 6.0])          # suavidade de h2 e h3: E[f''(z)^2] = sum_b b(b-1) beta_b^2
P = 3                                  # número de coeficientes por (d, k): w, beta2, beta3


def psi(z):
    """Base psi(z) = (z, h2(z), h3(z)), com os Hermite normalizados (média 0, variância 1 e ortogonais sob N(0, 1))."""
    z = np.asarray(z, dtype=float)
    return np.stack([z, (z ** 2 - 1) / np.sqrt(2), (z ** 3 - 3 * z) / np.sqrt(6)], axis=-1)


# ================================================================== estado variacional
def inicializa(visoes, K, semente=0, iter_linear=200, grade=(-6, 6, 241), a0=1e-14, b0=1e-14,
               a0_theta=1.0, b0_theta=1.0, a0_pi=1.0, b0_pi=9.0, limiar_poda=0.005, limiar_derivado=0.5,
               verbose=True):
    """Começa da solução do MOFA linear (mofa.py): Z, w e gamma vêm de lá, e nenhuma curva está ligada ainda.

    Antes, remove os fatores lineares que são função não linear (h2, h3) de outro fator, com R^2 > limiar_derivado.
    Use limiar_derivado=1.01 para não remover nenhum.
    """
    if not isinstance(visoes, dict):
        visoes = {f"visao{i}": v for i, v in enumerate(visoes)}
    visoes = {m: np.asarray(v, dtype=float) for m, v in visoes.items()}
    lin = M.mofa(visoes, K=K, max_iter=iter_linear, tol=1e-6, limiar_poda=limiar_poda, semente=semente,
                 verbose=verbose, a0_alpha=a0, b0_alpha=b0, a0_tau=a0, b0_tau=b0)
    sl = lin["estado"]
    # fatores "harmônicos": no MOFA linear, uma resposta curva vira um fator extra que é função de outro fator
    # (z_j ~ h2(z_k), o efeito ferradura do PCA). Esses fatores são removidos para as curvas assumirem o papel deles.
    derivados = fatores_derivados(sl["Zmu"], {m: lin["r2"][m] for m in sl["visoes"]}, limiar_derivado)
    if derivados:
        if verbose:
            for j, k, r in derivados:
                print(f"fator linear {j + 1} é função não linear do fator {k + 1} (R² = {r:.2f}): removido")
        manter = np.ones(sl["K"], bool); manter[[j for j, _, _ in derivados]] = False
        sl["Zmu"], sl["Zs2"] = sl["Zmu"][:, manter], sl["Zs2"][manter]
        for m in sl["visoes"]:
            for c in ("Wmu", "Ws2", "gamma"):
                sl[c][m] = sl[c][m][:, manter]
            for c in ("a_alpha", "b_alpha", "a_theta", "b_theta"):
                sl[c][m] = sl[c][m][manter]
        sl["K"] = int(manter.sum())
    K = sl["K"]
    zg = np.linspace(*grade)
    st = {
        "Y": sl["Y"], "N": sl["N"], "K": K, "visoes": sl["visoes"], "zg": zg, "psig": psi(zg),
        "dz": zg[1] - zg[0],
        "pri": dict(a0=a0, b0=b0, a0_theta=a0_theta, b0_theta=b0_theta, a0_pi=a0_pi, b0_pi=b0_pi),
        "mu1": {}, "s21": {}, "mu2": {}, "S2": {}, "p": {},
        "a_alpha": {}, "b_alpha": {}, "a_rho": {}, "b_rho": {}, "a_theta": {}, "b_theta": {},
        "a_pi": {}, "b_pi": {}, "a_tau": {}, "b_tau": {}, "elbo_linear": lin["elbo"],
    }
    # q(z) na grade, a partir da gaussiana do MOFA linear
    st["m1"], st["M2"], st["Hz"] = np.zeros((st["N"], K, P)), np.zeros((st["N"], K, P, P)), np.zeros((st["N"], K))
    for k in range(K):
        logq = -0.5 * (zg[None, :] - sl["Zmu"][:, k:k + 1]) ** 2 / sl["Zs2"][k]
        _guarda_z(st, k, logq)
    for m in st["visoes"]:
        D = st["Y"][m].shape[1]
        g = sl["gamma"][m]
        st["mu1"][m], st["s21"][m] = sl["Wmu"][m].copy(), sl["Ws2"][m].copy()
        st["mu2"][m] = np.zeros((D, K, P)); st["mu2"][m][..., 0] = sl["Wmu"][m]
        st["S2"][m] = np.tile(np.eye(P) * 1e-2, (D, K, 1, 1)); st["S2"][m][..., 0, 0] = sl["Ws2"][m]
        st["p"][m] = np.stack([1 - g, g, np.zeros_like(g)], axis=-1)            # estado 2 começa desligado
        st["a_alpha"][m], st["b_alpha"][m] = sl["a_alpha"][m].copy(), sl["b_alpha"][m].copy()
        st["a_rho"][m], st["b_rho"][m] = np.full(K, 1.0), np.full(K, 1.0)
        st["a_theta"][m], st["b_theta"][m] = sl["a_theta"][m].copy(), sl["b_theta"][m].copy()
        st["a_pi"][m], st["b_pi"][m] = np.full(K, a0_pi), np.full(K, b0_pi)
        st["a_tau"][m], st["b_tau"][m] = sl["a_tau"][m].copy(), sl["b_tau"][m].copy()
    return st


def fatores_derivados(Z, r2_lin, limiar):
    """Lista (j, k, R^2) dos fatores j cuja média E[z_j] é explicada por h2(z_k), h3(z_k) de um fator k mais forte."""
    K = Z.shape[1]
    forca = np.max(np.vstack(list(r2_lin.values())), axis=0)           # R^2 máximo entre as vistas
    ordem = np.argsort(-forca)
    mantidos, derivados = [], []
    for j in ordem:
        melhor = (0.0, None)
        for k in mantidos:
            X = np.column_stack([np.ones(len(Z)), psi(Z[:, k] / Z[:, k].std())[:, 1:]])
            b = np.linalg.lstsq(X, Z[:, j], rcond=None)[0]
            r = 1 - ((Z[:, j] - X @ b) ** 2).sum() / ((Z[:, j] - Z[:, j].mean()) ** 2).sum()
            if r > melhor[0]:
                melhor = (r, k)
        if melhor[0] > limiar:
            derivados.append((int(j), int(melhor[1]), float(melhor[0])))
        else:
            mantidos.append(j)
    return derivados


def _guarda_z(st, k, logq):
    """Normaliza q(z_nk) na grade e guarda os momentos E[psi], E[psi psi^T] e a entropia."""
    logw = logq - logsumexp(logq, axis=1, keepdims=True)
    w = np.exp(logw)
    psig = st["psig"]
    st["m1"][:, k] = w @ psig
    st["M2"][:, k] = np.einsum("ng,gi,gj->nij", w, psig, psig)
    st["Hz"][:, k] = -(w * logw).sum(axis=1) + np.log(st["dz"])          # entropia da densidade na grade


# ================================================================== momentos
def Eg(a, b): return a / b                                              # E[x] de uma Gama(a, b)
def Elog(a, b): return digamma(a) - np.log(b)                           # E[log x]


def P_diag(st, m):
    """E[precisão a priori] de (w, beta2, beta3) por fator: (E[alfa], E[rho] 2, E[rho] 6). Matriz K x 3."""
    ea, er = Eg(st["a_alpha"][m], st["b_alpha"][m]), Eg(st["a_rho"][m], st["b_rho"][m])
    return np.column_stack([ea, er * LAMBDA[0], er * LAMBDA[1]])


def EU(st, m):
    """E[u_til] (D x K x 3): u_til = a (w, c beta). Estado 1 só tem w; estado 2 tem os três coeficientes."""
    p = st["p"][m]
    out = p[..., 2:3] * st["mu2"][m]
    out[..., 0] += p[..., 1] * st["mu1"][m]
    return out


def EUU(st, m):
    """E[u_til u_til^T] (D x K x 3 x 3)."""
    p, mu2 = st["p"][m], st["mu2"][m]
    out = p[..., 2, None, None] * (mu2[..., :, None] * mu2[..., None, :] + st["S2"][m])
    out[..., 0, 0] += p[..., 1] * (st["mu1"][m] ** 2 + st["s21"][m])
    return out


# ================================================================== passo 4.1: coeficientes e estados
def atualiza_U(st, m):
    """Para cada fator k (em sequência), atualiza q(u | estado) e q(estado) de todas as features da vista m.

    c_dk = sum_n E[psi(z_nk)] r^(k)_nd é calculado sem a matriz de resíduos, como no mofa.py:
        c_k = (Psi^T Y)_k - sum_{j != k} (Psi^T Psi)_kj E[u_j]
    """
    Y, N, K = st["Y"][m], st["N"], st["K"]
    tau = Eg(st["a_tau"][m], st["b_tau"][m])
    Pd = P_diag(st, m)
    elt = digamma(st["a_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])     # E[log theta]
    el1t = digamma(st["b_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])    # E[log(1 - theta)]
    elp = digamma(st["a_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])             # E[log pi]
    el1p = digamma(st["b_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])            # E[log(1 - pi)]
    Psi = st["m1"].reshape(N, K * P)
    A = Psi.T @ Y                                                       # 3K x D
    G = Psi.T @ Psi                                                     # 3K x 3K (produto das médias)
    U = EU(st, m)                                                       # D x K x 3, atualizado fator a fator
    for k in range(K):
        sl = slice(P * k, P * k + P)
        Uall = U.reshape(-1, K * P)
        C = A[sl] - G[sl] @ Uall.T + G[sl, sl] @ U[:, k].T             # 3 x D
        S = st["M2"][:, k].sum(axis=0)                                  # sum_n E[psi psi^T], 3 x 3
        # estado 2 (não linear): normal 3D
        prec = np.diag(Pd[k])[None] + tau[:, None, None] * S[None]
        S2 = np.linalg.inv(prec)
        mu2 = np.einsum("dij,dj->di", S2, tau[:, None] * C.T)
        _, logdet_prec = np.linalg.slogdet(prec)
        L2 = elt[k] + elp[k] + 0.5 * (np.log(Pd[k]).sum() - logdet_prec) \
            + 0.5 * np.einsum("di,dij,dj->d", mu2, prec, mu2)
        # estado 1 (linear): igual ao MOFA
        s21 = 1.0 / (Pd[k, 0] + tau * S[0, 0])
        mu1 = s21 * tau * C[0]
        L1 = elt[k] + el1p[k] + 0.5 * np.log(Pd[k, 0] * s21) + mu1 ** 2 / (2 * s21)
        # estado 0
        L0 = np.full_like(L1, el1t[k])
        Ls = np.stack([L0, L1, L2], axis=-1)
        p = np.exp(Ls - logsumexp(Ls, axis=-1, keepdims=True))
        st["mu1"][m][:, k], st["s21"][m][:, k] = mu1, s21
        st["mu2"][m][:, k], st["S2"][m][:, k], st["p"][m][:, k] = mu2, S2, p
        U[:, k] = p[:, 2:3] * mu2
        U[:, k, 0] += p[:, 1] * mu1


# ================================================================== passo 4.2: alfa, rho, theta, pi
def atualiza_hiper(st, m):
    D, p0 = st["Y"][m].shape[1], st["p"][m][..., 0]
    p1, p2 = st["p"][m][..., 1], st["p"][m][..., 2]
    pr = st["pri"]
    Ea, Er = Eg(st["a_alpha"][m], st["b_alpha"][m]), Eg(st["a_rho"][m], st["b_rho"][m])
    # alfa: E[w^2] somando os três estados (no estado 0, w segue a prior)
    Ew2 = p0 / Ea + p1 * (st["mu1"][m] ** 2 + st["s21"][m]) + p2 * (st["mu2"][m][..., 0] ** 2 + st["S2"][m][..., 0, 0])
    st["a_alpha"][m] = np.full(st["K"], pr["a0"] + D / 2)
    st["b_alpha"][m] = pr["b0"] + 0.5 * Ew2.sum(axis=0)
    # rho: E[beta^T Lambda beta] (nos estados 0 e 1, beta segue a prior: valor esperado 2/E[rho])
    mu, S = st["mu2"][m][..., 1:], st["S2"][m]
    Eb = (LAMBDA * (mu ** 2 + np.stack([S[..., 1, 1], S[..., 2, 2]], axis=-1))).sum(axis=-1)
    EbLb = (p0 + p1) * 2 / Er + p2 * Eb
    st["a_rho"][m] = np.full(st["K"], pr["a0"] + D * 2 / 2)
    st["b_rho"][m] = pr["b0"] + 0.5 * EbLb.sum(axis=0)
    # theta (associado) e pi (não linear entre os associados)
    st["a_theta"][m] = pr["a0_theta"] + (p1 + p2).sum(axis=0)
    st["b_theta"][m] = pr["b0_theta"] + p0.sum(axis=0)
    st["a_pi"][m] = pr["a0_pi"] + p2.sum(axis=0)
    st["b_pi"][m] = pr["b0_pi"] + p1.sum(axis=0)


# ================================================================== passo 4.3: Z na grade
def atualiza_Z(st):
    """log q(z_nk = z) = -z^2/2 + psi(z)^T h_nk - 1/2 psi(z)^T H_k psi(z), avaliado na grade."""
    N, K = st["N"], st["K"]
    YTU, GU, H = 0.0, 0.0, np.zeros((K, P, P))
    for m in st["visoes"]:
        tau = Eg(st["a_tau"][m], st["b_tau"][m])
        Uall = EU(st, m).reshape(-1, K * P)
        TU = tau[:, None] * Uall
        YTU = YTU + st["Y"][m] @ TU                                     # N x 3K
        GU = GU + Uall.T @ TU                                           # 3K x 3K
        H += np.einsum("d,dkij->kij", tau, EUU(st, m))                  # sum_d tau E[u u^T], por fator
    zg, psig = st["zg"], st["psig"]
    for k in range(K):
        sl = slice(P * k, P * k + P)
        Psi = st["m1"].reshape(N, K * P)
        h = YTU[:, sl] - Psi @ GU[:, sl] + st["m1"][:, k] @ GU[sl, sl]  # N x 3
        quad = np.einsum("gi,ij,gj->g", psig, H[k], psig)
        logq = -0.5 * zg[None] ** 2 + h @ psig.T - 0.5 * quad[None]
        _guarda_z(st, k, logq)


# ================================================================== passo 4.2: ruído tau
def erro_quadratico_esperado(st, m):
    """sum_n E[(y_nd - sum_k f_dk(z_nk))^2]: resíduo das médias + soma das variâncias das curvas."""
    Y, N, K = st["Y"][m], st["N"], st["K"]
    U, UU = EU(st, m), EUU(st, m)
    R = Y - st["m1"].reshape(N, K * P) @ U.reshape(-1, K * P).T
    S = st["M2"].sum(axis=0)                                            # K x 3 x 3
    Q = np.einsum("nki,nkj->kij", st["m1"], st["m1"])                   # K x 3 x 3
    var = np.einsum("dkij,kij->d", UU, S) - np.einsum("dki,kij,dkj->d", U, Q, U)
    return (R ** 2).sum(axis=0) + var


def atualiza_tau(st, m):
    sqe = erro_quadratico_esperado(st, m)
    st["a_tau"][m] = np.full_like(sqe, st["pri"]["a0"] + st["N"] / 2)
    st["b_tau"][m] = st["pri"]["b0"] + 0.5 * sqe


# ================================================================== ELBO
def _kl_gama(a, b, a0, b0):
    return ((a - a0) * digamma(a) - gammaln(a) + gammaln(a0) + a0 * (np.log(b) - np.log(b0)) + a * (b0 - b) / b).sum()


def _kl_beta(a, b, a0, b0):
    ea, eb = digamma(a) - digamma(a + b), digamma(b) - digamma(a + b)
    return ((a - a0) * ea + (b - b0) * eb - betaln(a, b) + betaln(a0, b0)).sum()


def elbo(st):
    pr = st["pri"]
    # fatores: E[log p(z)] + H[q(z)]
    L = (-0.5 * LOG2PI - 0.5 * st["M2"][..., 0, 0] + st["Hz"]).sum()
    for m in st["visoes"]:
        a_t, b_t = st["a_tau"][m], st["b_tau"][m]
        L += (st["N"] / 2 * (Elog(a_t, b_t) - LOG2PI) - 0.5 * Eg(a_t, b_t) * erro_quadratico_esperado(st, m)).sum()
        # coeficientes: E[log p(u | alfa, rho)] - E[log q(u | estado)], por estado (seção 4 do documento)
        Pd = P_diag(st, m)                                               # K x 3
        ElogP = np.column_stack([Elog(st["a_alpha"][m], st["b_alpha"][m]),
                                 Elog(st["a_rho"][m], st["b_rho"][m]) + np.log(LAMBDA[0]),
                                 Elog(st["a_rho"][m], st["b_rho"][m]) + np.log(LAMBDA[1])])
        comum = 0.5 * ElogP.sum(axis=1)[None]                            # 1 x K
        logPd = np.log(Pd)[None]                                         # 1 x K x 3
        mu2, S2 = st["mu2"][m], st["S2"][m]
        v0 = comum - 0.5 * logPd.sum(axis=-1)
        v1 = comum - 0.5 * Pd[None, :, 0] * (st["mu1"][m] ** 2 + st["s21"][m]) + 0.5 * np.log(st["s21"][m]) + 0.5 \
            - 0.5 * logPd[..., 1:].sum(axis=-1)
        diag2 = mu2 ** 2 + np.diagonal(S2, axis1=-2, axis2=-1)
        _, logdetS2 = np.linalg.slogdet(S2)
        v2 = comum - 0.5 * (Pd[None] * diag2).sum(axis=-1) + 0.5 * logdetS2 + 1.5
        p = st["p"][m]
        L += (p[..., 0] * v0 + p[..., 1] * v1 + p[..., 2] * v2).sum()
        # estados: E[log p(estado | theta, pi)] - E[log q(estado)]
        elt = digamma(st["a_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        el1t = digamma(st["b_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        elp = digamma(st["a_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
        el1p = digamma(st["b_pi"][m]) - digamma(st["a_pi"][m] + st["b_pi"][m])
        pc = np.clip(p, 1e-300, 1)
        L += (p[..., 0] * el1t + p[..., 1] * (elt + el1p) + p[..., 2] * (elt + elp) - (p * np.log(pc)).sum(-1)).sum()
        # hiperparâmetros
        L -= _kl_gama(st["a_alpha"][m], st["b_alpha"][m], pr["a0"], pr["b0"])
        L -= _kl_gama(st["a_rho"][m], st["b_rho"][m], pr["a0"], pr["b0"])
        L -= _kl_gama(a_t, b_t, pr["a0"], pr["b0"])
        L -= _kl_beta(st["a_theta"][m], st["b_theta"][m], pr["a0_theta"], pr["b0_theta"])
        L -= _kl_beta(st["a_pi"][m], st["b_pi"][m], pr["a0_pi"], pr["b0_pi"])
    return L


# ================================================================== variância explicada e poda
def r2(st, partes=(0, 1, 2)):
    """R^2 de cada fator em cada vista usando só os termos da base em `partes` (0 = linear; 1, 2 = curvos)."""
    N, K, out = st["N"], st["K"], {}
    mask = np.zeros(P); mask[list(partes)] = 1
    for m in st["visoes"]:
        Y = st["Y"][m]
        sq = (Y ** 2).sum()
        U = EU(st, m) * mask
        A = st["m1"].reshape(N, K * P).T @ Y                             # 3K x D
        out[m] = np.empty(K)
        for k in range(K):
            sl = slice(P * k, P * k + P)
            Q = st["m1"][:, k].T @ st["m1"][:, k]
            cruz = (A[sl] * U[:, k].T).sum()
            quad = np.einsum("di,ij,dj->", U[:, k], Q, U[:, k])
            out[m][k] = 1 - (sq - 2 * cruz + quad) / sq
    return out


def poda(st, limiar):
    R = r2(st)
    manter = np.max(np.vstack([R[m] for m in st["visoes"]]), axis=0) >= limiar
    if manter.all():
        return 0
    st["m1"], st["M2"], st["Hz"] = st["m1"][:, manter], st["M2"][:, manter], st["Hz"][:, manter]
    for m in st["visoes"]:
        for c in ("mu1", "s21", "mu2", "S2", "p"):
            st[c][m] = st[c][m][:, manter]
        for c in ("a_alpha", "b_alpha", "a_rho", "b_rho", "a_theta", "b_theta", "a_pi", "b_pi"):
            st[c][m] = st[c][m][manter]
    st["K"] = int(manter.sum())
    return int((~manter).sum())


# ================================================================== o algoritmo
def iteracao(st):
    for m in st["visoes"]:
        atualiza_U(st, m)
        atualiza_hiper(st, m)
    atualiza_Z(st)
    for m in st["visoes"]:
        atualiza_tau(st, m)


def mofa_aditivo(visoes, K=10, max_iter=500, tol=1e-6, limiar_poda=0.005, semente=0, verbose=True,
                 mostrar_a_cada=25, **opcoes):
    t0 = time.time()
    st = inicializa(visoes, K, semente=semente, limiar_poda=limiar_poda, verbose=verbose, **opcoes)
    if verbose:
        print(f"início linear pronto (K = {st['K']}); ligando as curvas ...")
    hist = []
    for it in range(max_iter):
        iteracao(st)
        removidos = poda(st, limiar_poda)
        if removidos and verbose:
            print(f"iter {it + 1}: {removidos} fator(es) podado(s), K = {st['K']}")
        hist.append(elbo(st))
        if verbose and (it + 1) % mostrar_a_cada == 0:
            print(f"iter {it + 1}: ELBO = {hist[-1]:.1f}, K = {st['K']}, {time.time() - t0:.0f} s")
        if it > 0 and abs(hist[-1] - hist[-2]) / abs(hist[-2]) < tol:
            break
    if verbose:
        print(f"convergiu em {it + 1} iterações ({time.time() - t0:.0f} s), ELBO = {hist[-1]:.1f}, K = {st['K']}")
    return resultado(st, np.array(hist))


def resultado(st, hist):
    r_tot, r_lin = r2(st), r2(st, partes=(0,))
    return {
        "Z": st["m1"][..., 0],                                           # E[z_nk]
        "W": {m: EU(st, m)[..., 0] for m in st["visoes"]},               # inclinação média E[a w]
        "U": {m: EU(st, m) for m in st["visoes"]},                       # E[(a w, a c beta2, a c beta3)]
        "estados": {m: st["p"][m] for m in st["visoes"]},                # q(0), q(1), q(2)
        "gamma": {m: 1 - st["p"][m][..., 0] for m in st["visoes"]},      # P(associado), como no MOFA
        "p_nao_linear": {m: st["p"][m][..., 2] for m in st["visoes"]},
        "alpha": {m: Eg(st["a_alpha"][m], st["b_alpha"][m]) for m in st["visoes"]},
        "rho": {m: Eg(st["a_rho"][m], st["b_rho"][m]) for m in st["visoes"]},
        "pi": {m: st["a_pi"][m] / (st["a_pi"][m] + st["b_pi"][m]) for m in st["visoes"]},
        "tau": {m: Eg(st["a_tau"][m], st["b_tau"][m]) for m in st["visoes"]},
        "r2": r_tot, "r2_linear": r_lin,
        "r2_nao_linear": {m: r_tot[m] - r_lin[m] for m in st["visoes"]},
        "elbo": hist, "estado": st,
    }


# ================================================================== interpretação por feature
def curva(res, m, d, k, z=None):
    """Curva E[f_dk(z)] e desvio-padrão a posteriori (mistura dos três estados), numa grade de z."""
    st = res["estado"]
    z = np.linspace(-2.5, 2.5, 101) if z is None else np.asarray(z)
    ps = psi(z)
    u, uu = EU(st, m)[d, k], EUU(st, m)[d, k]
    media = ps @ u
    var = np.einsum("gi,ij,gj->g", ps, uu - np.outer(u, u), ps)
    return z, media, np.sqrt(np.maximum(var, 0))


def _nomes(visoes, m, D):
    v = visoes.get(m) if isinstance(visoes, dict) else None
    return np.asarray(v.columns) if isinstance(v, pd.DataFrame) else np.array([f"{m}_{d}" for d in range(D)])


def classifica_forma(w, b2, b3):
    """Forma dominante da curva w z + b2 h2 + b3 h3.

    As formas são relativas ao sentido da inclinação, então não dependem do sinal arbitrário do fator:
    b2 com sinal oposto a w = a resposta perde força no fim para onde ela cresce (satura); mesmo sinal = acelera.
    """
    if abs(b2) >= abs(b3):
        if abs(w) < abs(b2):
            return "U" if b2 > 0 else "U invertido"
        return "satura" if b2 * w < 0 else "acelera"
    if abs(w) < abs(b3):
        return "cúbica"
    return "sigmoide (achata nas pontas)" if b3 * w < 0 else "acentua nas pontas"


def tabela_formas(res, visoes=None, limiar=0.5, limiar_r2=0.01):
    """Features com q(não linear) > limiar nos pares (vista, fator) ativos, com coeficientes, forma e R^2 da feature."""
    st, linhas = res["estado"], []
    for m in st["visoes"]:
        U, p, Y = EU(st, m), st["p"][m], st["Y"][m]
        nomes = _nomes(visoes or {}, m, U.shape[0])
        sqd = (Y ** 2).sum(axis=0)
        for k in np.where(res["r2"][m] > limiar_r2)[0]:
            for d in np.where(p[:, k, 2] > limiar)[0]:
                w, b2, b3 = U[d, k]
                f_lin = st["m1"][:, k, 0] * w
                f_tot = st["m1"][:, k] @ U[d, k]
                r2_lin = 1 - ((Y[:, d] - f_lin) ** 2).sum() / sqd[d]
                r2_tot = 1 - ((Y[:, d] - f_tot) ** 2).sum() / sqd[d]
                linhas.append(dict(omica=m, fator=f"F{k + 1}", feature=nomes[d], p_nao_linear=p[d, k, 2],
                                   w=w, beta2=b2, beta3=b3, fracao_nao_linear=(b2 ** 2 + b3 ** 2) / (w ** 2 + b2 ** 2 + b3 ** 2),
                                   r2_linear=r2_lin, r2_total=r2_tot, forma=classifica_forma(w, b2, b3)))
    df = pd.DataFrame(linhas)
    return df.sort_values("p_nao_linear", ascending=False).reset_index(drop=True) if len(df) else df


# ================================================================== figuras
def figura_resumo(res, arquivo=None):
    """ELBO, R^2 linear, R^2 não linear (vistas x fatores) e E[pi] (fração de curvas entre as features associadas)."""
    import matplotlib.pyplot as plt
    nomes = list(res["r2"])
    Rl = np.array([res["r2_linear"][m] for m in nomes])
    Rn = np.array([res["r2_nao_linear"][m] for m in nomes])
    Pi = np.array([res["pi"][m] for m in nomes])
    K = Rl.shape[1]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4), gridspec_kw={"width_ratios": [1, 1.3, 1.3, 1.3]})
    ax[0].plot(np.arange(1, len(res["elbo"]) + 1), res["elbo"], color="#2a6fdb", lw=2)
    ax[0].set(title="ELBO (fase não linear)", xlabel="iteração", ylabel="ELBO")
    vmax = max(Rl.max(), Rn.max(), 1e-3)
    for a, mat, t, cmap, vm in [(ax[1], Rl, "R² linear", "Blues", vmax), (ax[2], Rn, "R² não linear", "Oranges", vmax),
                                (ax[3], Pi, "E[π]: fração de curvas", "Greens", 1.0)]:
        im = a.imshow(mat, cmap=cmap, vmin=0, vmax=vm, aspect="auto")
        a.set(title=t, yticks=range(len(nomes)), yticklabels=nomes, xticks=range(K),
              xticklabels=[f"F{k + 1}" for k in range(K)])
        for i in range(mat.shape[0]):
            for j in range(K):
                a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if mat[i, j] > 0.6 * vm else "black")
        fig.colorbar(im, ax=a, fraction=0.04)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


def figura_curvas(res, visoes, m, fator=0, n=8, arquivo=None, residuo_parcial=True, features=None):
    """As n features mais claramente não lineares de uma vista num fator, com a curva estimada e banda de 95%.

    Cada ponto é uma amostra: eixo x = E[z_n] do fator, eixo y = valor da feature.
    residuo_parcial=True tira do y a parte explicada pelos OUTROS fatores, que em dados reais esconde a curva.
    Os losangos pretos são a média do y em cada decil do fator (uma checagem sem modelo da forma).
    features: lista de índices para desenhar features específicas em vez das n mais não lineares.
    """
    import matplotlib.pyplot as plt
    st = res["estado"]
    p = st["p"][m][:, fator, 2] * (res["r2"][m][fator] > 0)
    idx = np.argsort(-p)[:n] if features is None else np.asarray(features)
    nomes = _nomes(visoes or {}, m, st["Y"][m].shape[1])
    N, K = st["N"], st["K"]
    U = EU(st, m)
    Yp = st["Y"][m]
    if residuo_parcial:
        outros = np.ones(K, bool); outros[fator] = False
        Yp = Yp - st["m1"][:, outros].reshape(N, -1) @ U[:, outros].reshape(U.shape[0], -1).T
    cols = 4
    lin = int(np.ceil(len(idx) / cols))
    fig, ax = plt.subplots(lin, cols, figsize=(4 * cols, 3.2 * lin), squeeze=False)
    zbar = st["m1"][:, fator, 0]
    for a, d in zip(ax.ravel(), idx):
        z, f, sd = curva(res, m, d, fator, np.linspace(np.percentile(zbar, 1), np.percentile(zbar, 99), 101))
        a.scatter(zbar, Yp[:, d], s=6, color="#9aa0a6", alpha=0.6)
        dec = np.digitize(zbar, np.percentile(zbar, np.arange(10, 100, 10)))
        a.plot([zbar[dec == i].mean() for i in range(10)], [Yp[dec == i, d].mean() for i in range(10)],
               "D", color="black", ms=3.5)
        a.fill_between(z, f - 1.96 * sd, f + 1.96 * sd, color="#e8710a", alpha=0.25, lw=0)
        a.plot(z, f, color="#e8710a", lw=2)
        w = EU(st, m)[d, fator, 0]
        a.plot(z, w * z, color="#2a6fdb", lw=1, ls="--")
        a.set(title=f"{nomes[d]}  q(não linear) = {st['p'][m][d, fator, 2]:.2f}", xlabel=f"F{fator + 1} (E[z] por amostra)",
              ylabel="resíduo parcial" if residuo_parcial else "y centrado")
        a.title.set_fontsize(9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a in ax.ravel()[len(idx):]:
        a.axis("off")
    fig.suptitle(f"{m}, fator F{fator + 1}: curva estimada (laranja, banda de 95%) e parte linear (azul tracejado)",
                 fontsize=10, color="#5f6368")
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


def importancia(res, m, fator=0):
    """Importância de cada feature da vista m no fator: fração da variância da feature explicada pela curva f_dk(z).

    Funciona para features lineares e curvas (para uma reta é w^2 var(z) / var(y)). Vetor de tamanho D.
    """
    st = res["estado"]
    f = st["m1"][:, fator] @ EU(st, m)[:, fator].T                      # N x D, E[f_dk(z_n)]
    return f.var(axis=0) / st["Y"][m].var(axis=0)


def figura_feature(res, visoes, m, feature, fator=0, n_top=20, arquivo=None):
    """Importância de uma feature (por nome ou índice) num fator: ranking na vista e a curva dela.

    fator começa em 0 (fator=0 é o F1). Painel 1: as n_top features mais importantes com a escolhida destacada.
    Painel 2: posição da feature no ranking de todas. Painel 3: curva estimada com resíduo parcial e médias por decil.
    """
    import matplotlib.pyplot as plt
    st = res["estado"]
    nomes = _nomes(visoes or {}, m, st["Y"][m].shape[1])
    d = int(np.where(nomes == feature)[0][0]) if isinstance(feature, str) else int(feature)
    imp = importancia(res, m, fator)
    ordem = np.argsort(-imp)
    pos = int(np.where(ordem == d)[0][0]) + 1
    p = st["p"][m][d, fator]
    u = EU(st, m)[d, fator]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
    # 1. top features
    top = list(ordem[:n_top])
    if d not in top:
        top = top[:-1] + [d]
    cores = ["#e8710a" if j == d else "#9aa0a6" for j in top]
    ax[0].barh(range(len(top))[::-1], imp[top], color=cores)
    ax[0].set_yticks(range(len(top))[::-1], [nomes[j] for j in top], fontsize=7)
    ax[0].set(xlabel="R² da feature explicado pelo fator", title=f"Top {n_top} em {m}, F{fator + 1}")
    # 2. ranking de todas
    ax[1].plot(np.arange(1, len(imp) + 1), imp[ordem], color="#5f6368", lw=1)
    ax[1].scatter([pos], [imp[d]], color="#e8710a", zorder=3, s=40)
    ax[1].annotate(f"{nomes[d]}\nposição {pos} de {len(imp)}\nR² = {imp[d]:.3f}", (pos, imp[d]),
                   xytext=(10, 10), textcoords="offset points", fontsize=8)
    ax[1].set(xscale="log", xlabel="posição no ranking (log)", ylabel="R² explicado pelo fator",
              title="Ranking de todas as features")
    # 3. curva com resíduo parcial
    N, K = st["N"], st["K"]
    U = EU(st, m)
    outros = np.ones(K, bool); outros[fator] = False
    y = st["Y"][m][:, d] - st["m1"][:, outros].reshape(N, -1) @ U[d, outros].reshape(-1)
    zbar = st["m1"][:, fator, 0]
    z, fz, sd = curva(res, m, d, fator, np.linspace(np.percentile(zbar, 1), np.percentile(zbar, 99), 101))
    ax[2].scatter(zbar, y, s=8, color="#9aa0a6", alpha=0.6)
    dec = np.digitize(zbar, np.percentile(zbar, np.arange(10, 100, 10)))
    ax[2].plot([zbar[dec == i].mean() for i in range(10)], [y[dec == i].mean() for i in range(10)], "D", color="black", ms=4)
    ax[2].fill_between(z, fz - 1.96 * sd, fz + 1.96 * sd, color="#e8710a", alpha=0.25, lw=0)
    ax[2].plot(z, fz, color="#e8710a", lw=2)
    ax[2].plot(z, u[0] * z, color="#2a6fdb", lw=1, ls="--")
    ax[2].set(xlabel=f"F{fator + 1} (E[z] por amostra)", ylabel="resíduo parcial",
              title=f"{nomes[d]}: P(associada) = {p[1] + p[2]:.2f}, P(não linear) = {p[2]:.2f}")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig


# ================================================================== validação cruzada: prever uma vista a partir das outras
def _params_linear(lin):
    st = lin["estado"]; out = {}
    for m in st["visoes"]:
        g, mu, s2 = st["gamma"][m], st["Wmu"][m], st["Ws2"][m]
        D, K = g.shape
        U = np.zeros((D, K, P)); UU = np.zeros((D, K, P, P))
        U[..., 0] = g * mu; UU[..., 0, 0] = g * (mu ** 2 + s2)
        out[m] = (U, UU, st["a_tau"][m] / st["b_tau"][m])
    return out


def _params_aditivo(res):
    st = res["estado"]
    return {m: (EU(st, m), EUU(st, m), Eg(st["a_tau"][m], st["b_tau"][m])) for m in st["visoes"]}


def _infere_Z(par, Y, varreduras=30):
    """q(z) de amostras novas (na grade) usando só as vistas em Y, com pesos e tau fixos."""
    N = next(iter(Y.values())).shape[0]
    K = next(iter(par.values()))[0].shape[1]
    zg = np.linspace(-6, 6, 241); psig = psi(zg)
    m1 = np.zeros((N, K, P))
    for _ in range(varreduras):
        for k in range(K):
            h = np.zeros((N, P)); H = np.zeros((P, P))
            for m, y in Y.items():
                U, UU, tau = par[m]
                outros = np.einsum("nki,dki->nd", np.delete(m1, k, 1), np.delete(U, k, 1))
                h += ((y - outros) * tau) @ U[:, k]
                H += np.einsum("d,dij->ij", tau, UU[:, k])
            logq = -0.5 * zg ** 2 + h @ psig.T - 0.5 * np.einsum("gi,ij,gj->g", psig, H, psig)
            w = np.exp(logq - logsumexp(logq, axis=1, keepdims=True))
            m1[:, k] = w @ psig
    return m1


def validacao_cruzada(visoes, K=10, folds=5, semente=0, max_iter=3000, **kw):
    """Compara MOFA linear e aditivo fora da amostra.

    Em cada fold: ajusta os dois modelos no treino; nas amostras de teste esconde uma vista inteira, infere Z com
    as outras e prevê a vista escondida. Devolve um DataFrame com o R^2 de teste por fold, modelo e vista prevista.
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
        ad = mofa_aditivo(Vtr, K=K, max_iter=max_iter, verbose=False, **kw)
        for nome, par in [("linear", _params_linear(lin)), ("aditivo", _params_aditivo(ad))]:
            for alvo in V:
                m1 = _infere_Z(par, {m: Vte[m] for m in V if m != alvo})
                pred = np.einsum("nki,dki->nd", m1, par[alvo][0])
                y = Vte[alvo]
                linhas.append(dict(fold=f + 1, modelo=nome, vista=alvo,
                                   r2=1 - ((y - pred) ** 2).sum() / (y ** 2).sum()))
        print(f"fold {f + 1}/{folds} pronto", flush=True)
    return pd.DataFrame(linhas)


def concentracao_curvas(res, m, fator=0, n=3, sims=300, semente=0):
    """Quanto da parte curva do fator (soma de quadrados) vem das n amostras com maior contribuição.

    Devolve (observado, referencia, p95). A referência é o mesmo cálculo com as mesmas curvas, mas com z simulado
    de uma N(0,1) com o mesmo N: como h2 e h3 crescem como z^2 e z^3, mesmo uma curva real e bem distribuída
    concentra bastante nas pontas (com N = 121, cerca de 40 a 50% em 3 amostras). Só desconfie quando o observado
    passa do p95 da referência.
    """
    st = res["estado"]
    B = EU(st, m)[:, fator, 1:]                                            # D x 2
    def frac(ps):
        ss = ((ps @ B.T) ** 2).sum(1)
        return np.sort(ss)[-n:].sum() / ss.sum()
    obs = frac(st["m1"][:, fator, 1:])
    rng = np.random.default_rng(semente)
    ref = np.array([frac(psi(rng.standard_normal(st["N"]))[:, 1:]) for _ in range(sims)])
    return float(obs), float(np.median(ref)), float(np.percentile(ref, 95))


# ================================================================== pesos no estilo do MOFA
def peso_efetivo(res, m, fator=0, visoes=None):
    """Análogo do E[w] do MOFA para o modelo aditivo. DataFrame com uma linha por feature.

    peso       = sinal(tendência) x desvio-padrão da curva E[f_dk(z_n)] entre as amostras. Para uma feature linear
                 é E[w] x dp(z), ou seja, proporcional a E[w] com a mesma constante para todas as features; para uma curva mede o tamanho total do efeito, mesmo quando
                 w ~ 0 (um U puro tem w = 0 mas efeito grande). O sinal é o da correlação entre f(z) e z.
    parte_linear, parte_curva = desvio-padrão de w z e de b2 h2 + b3 h3, com o mesmo sinal (somam em quadrado ~ peso^2
                 quando z ~ N(0,1), porque z, h2 e h3 são ortogonais).
    """
    st = res["estado"]
    U = EU(st, m)[:, fator]                                             # D x 3
    ps = st["m1"][:, fator]                                             # N x 3, E[psi(z_n)]
    f = ps @ U.T
    z = ps[:, 0]
    sinal = np.sign(((f - f.mean(0)) * (z - z.mean())[:, None]).sum(0))
    sinal[sinal == 0] = 1
    lin = (ps[:, :1] @ U[:, :1].T).std(0)
    cur = (ps[:, 1:] @ U[:, 1:].T).std(0)
    p = st["p"][m][:, fator]
    nomes = _nomes(visoes or {}, m, U.shape[0])
    return pd.DataFrame({"peso": sinal * f.std(0), "parte_linear": sinal * lin, "parte_curva": cur,
                         "p_associada": p[:, 1] + p[:, 2], "p_nao_linear": p[:, 2]}, index=nomes)


def figura_pesos(res, visoes, m, fator=0, n=15, destacar=None, arquivo=None):
    """Como o plot_weights do MOFA: todas as features ordenadas pelo peso efetivo, com as n maiores nomeadas,
    e um painel com as n maiores (em |peso|) separando parte linear (azul) e parte curva (laranja)."""
    import matplotlib.pyplot as plt
    tab = peso_efetivo(res, m, fator, visoes)
    ordem = tab.sort_values("peso")
    curva_ = ordem.p_nao_linear > 0.5
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(ordem))
    ax[0].scatter(x[~curva_], ordem.peso[~curva_], s=4, color="#2a6fdb", label="linear")
    ax[0].scatter(x[curva_], ordem.peso[curva_], s=4, color="#e8710a", label="curva (q > 0,5)")
    rot = list(ordem.index[:n // 2]) + list(ordem.index[-(n - n // 2):]) + list(destacar or [])
    for nome in dict.fromkeys(rot):
        i = ordem.index.get_loc(nome)
        ax[0].annotate(nome, (x[i], ordem.peso.iloc[i]), fontsize=7, xytext=(3, 0), textcoords="offset points",
                       color="black" if nome not in (destacar or []) else "#c5221f")
    ax[0].axhline(0, color="#9aa0a6", lw=0.8)
    ax[0].set(xlabel="features ordenadas", ylabel="peso efetivo (sinal x dp da curva)", title=f"{m}, F{fator + 1}")
    ax[0].legend(frameon=False, fontsize=8)
    top = tab.reindex(tab.peso.abs().sort_values(ascending=False).index[:n])
    nomes_top = list(top.index)
    for nome in (destacar or []):
        if nome not in nomes_top:
            nomes_top.append(nome)
    top = tab.loc[nomes_top][::-1]
    y = np.arange(len(top))
    ax[1].barh(y + 0.2, top.parte_linear, height=0.4, color="#2a6fdb", label="parte linear (w z)")
    ax[1].barh(y - 0.2, top.parte_curva, height=0.4, color="#e8710a", label="parte curva (b2 h2 + b3 h3)")
    ax[1].set_yticks(y, top.index, fontsize=7)
    ax[1].axvline(0, color="#9aa0a6", lw=0.8)
    ax[1].set(xlabel="desvio-padrão do efeito", title=f"Top {n} por |peso efetivo|")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
    return fig
