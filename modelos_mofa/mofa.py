"""MOFA (Multi-Omics Factor Analysis), Argelaguet et al., Mol Syst Biol 2018. Implementação didática do zero.

Cada função corresponde a um passo do documento "MOFA passo a passo" (seções 2 a 5). Só verossimilhança gaussiana,
um grupo de amostras, sem dados faltantes. Vistas no formato pacientes x features, mesmos pacientes na mesma ordem.

    visoes = {"rna": dados_rna.T, "met": dados_met.T, "cnv": dados_cnv.T}     # N x D_m cada (dict, lista ou DataFrame)
    res = mofa(visoes, K=15, max_iter=1000)
    res["Z"]                 # N x K, média posterior dos fatores
    res["W"]["rna"]          # D x K, E[w] = gamma * mu (já esparso)
    res["gamma"]["rna"]      # D x K, probabilidade posterior de inclusão de cada feature em cada fator
    res["r2"]                # vistas x fatores, variância explicada
"""
import time
import numpy as np
from scipy.special import digamma, gammaln, betaln, expit

LOG2PI = np.log(2 * np.pi)


# ------------------------------------------------------------------ estado variacional
def inicializa(visoes, K, semente=0, a0_alpha=1e-14, b0_alpha=1e-14, a0_tau=1e-14, b0_tau=1e-14,
               a0_theta=1.0, b0_theta=1.0):
    """Cria todos os parâmetros variacionais da tabela da seção 3.

    Hiperparâmetros padrão = os do artigo (Argelaguet et al. 2018): Gamas de alfa e tau com a0 = b0 = 1e-14,
    Beta de theta com a0 = b0 = 1.

    Z começa nas K primeiras componentes principais das vistas concatenadas (como o mofapy2 faz por padrão),
    W começa em zero com gamma = 0.5, alfa e tau começam nos valores que tornam as primeiras atualizações estáveis.
    """
    rng = np.random.default_rng(semente)
    if not isinstance(visoes, dict):                                 # lista/tupla -> nomes "visao0", "visao1", ...
        visoes = {f"visao{i}": v for i, v in enumerate(visoes)}
    visoes = {m: np.asarray(v, dtype=float) for m, v in visoes.items()}   # aceita DataFrame do pandas
    Y = {m: v - v.mean(axis=0) for m, v in visoes.items()}           # centra as features (o MOFA modela o intercepto assim)
    N = next(iter(Y.values())).shape[0]
    concat = np.hstack([Y[m] / np.linalg.norm(Y[m]) for m in Y])     # cada vista com o mesmo peso no PCA inicial
    U, S, _ = np.linalg.svd(concat, full_matrices=False)
    Zmu = U[:, :K] * np.sqrt(N)                                     # escala ~ N(0, 1)
    Zmu += 0.01 * rng.standard_normal(Zmu.shape)
    st = {
        "Y": Y, "N": N, "K": K, "visoes": list(Y),
        "pri": dict(a0_alpha=a0_alpha, b0_alpha=b0_alpha, a0_tau=a0_tau, b0_tau=b0_tau, a0_theta=a0_theta, b0_theta=b0_theta),
        "Zmu": Zmu, "Zs2": np.full(K, 1.0),                         # sigma_z^2 não depende de n (passo 4.4)
        "Wmu": {}, "Ws2": {}, "gamma": {},
        "a_alpha": {}, "b_alpha": {}, "a_theta": {}, "b_theta": {}, "a_tau": {}, "b_tau": {},
    }
    for m, y in Y.items():
        D = y.shape[1]
        st["Wmu"][m] = np.zeros((D, K))
        st["Ws2"][m] = np.ones((D, K))
        st["gamma"][m] = np.full((D, K), 0.5)
        st["a_alpha"][m] = np.full(K, 1.0); st["b_alpha"][m] = np.full(K, 1.0)      # E[alfa] = 1
        st["a_theta"][m] = np.full(K, 1.0); st["b_theta"][m] = np.full(K, 1.0)      # E[theta] = 0.5
        var = y.var(axis=0) + 1e-6
        st["a_tau"][m] = np.full(D, 1.0); st["b_tau"][m] = var                     # E[tau] = 1/variância
    return st


# ------------------------------------------------------------------ momentos (última coluna da tabela da seção 3)
def EW(st, m):  return st["gamma"][m] * st["Wmu"][m]                               # E[w] = gamma mu
def EW2(st, m): return st["gamma"][m] * (st["Wmu"][m] ** 2 + st["Ws2"][m])          # E[w^2] = gamma (mu^2 + s^2)
def EZ2(st):    return st["Zmu"] ** 2 + st["Zs2"]                                   # E[z^2] = mu^2 + s^2
def Ealpha(st, m): return st["a_alpha"][m] / st["b_alpha"][m]
def Etau(st, m):   return st["a_tau"][m] / st["b_tau"][m]
def Elogit_theta(st, m): return digamma(st["a_theta"][m]) - digamma(st["b_theta"][m])


# ------------------------------------------------------------------ passo 4.1: pesos (w_chapeu, s)
def atualiza_W(st, m, log=None):
    """Passo 4.1. O termo sum_n E[z_nk] r^(k)_nd é calculado sem montar a matriz de resíduos N x D:

        sum_n z_nk r^(k)_nd = (Z^T Y)_kd - sum_{j != k} (Z^T Z)_kj E[w_dj]

    Z^T Y e Z^T Z são calculados uma vez por vista (Z não muda durante a atualização de W). Isso dá exatamente
    o mesmo resultado da fórmula do documento, mas custa O(N D K) em vez de O(N D K^2) por iteração.
    """
    Y, Z, EZZ = st["Y"][m], st["Zmu"], EZ2(st)
    tau, alpha, lt = Etau(st, m), Ealpha(st, m), Elogit_theta(st, m)
    ZtY, ZtZ = Z.T @ Y, Z.T @ Z                                       # K x D e K x K
    Ew = EW(st, m)                                                    # D x K, atualizado coluna a coluna abaixo
    for k in range(st["K"]):
        cov = ZtY[k] - Ew @ ZtZ[:, k] + ZtZ[k, k] * Ew[:, k]          # sum_n E[z_nk] r^(k)_nd, vetor sobre d
        s2 = 1.0 / (alpha[k] + tau * EZZ[:, k].sum())               # (sigma^w_dk)^2, vetor sobre d
        mu = s2 * tau * cov                                         # mu^w_dk
        logit = lt[k] + 0.5 * np.log(alpha[k] * s2) + mu ** 2 / (2 * s2)
        g = expit(logit)
        st["Ws2"][m][:, k], st["Wmu"][m][:, k], st["gamma"][m][:, k] = s2, mu, g
        Ew[:, k] = g * mu                                           # o próximo k já usa o E[w] novo
        if log is not None and k == 0:
            log.append(dict(passo="4.1", visao=m, k=k, d=0, sum_Ez2=EZZ[:, k].sum(), Etau=tau[0], Ealpha=alpha[k],
                            cov_z_r=cov[0], s2=s2[0], mu=mu[0], logit_theta=lt[k],
                            termo_occam=0.5 * np.log(alpha[k] * s2[0]), termo_ajuste=mu[0] ** 2 / (2 * s2[0]),
                            gamma=g[0]))


# ------------------------------------------------------------------ passo 4.2: ARD alfa
def atualiza_alpha(st, m, log=None):
    D = st["Y"][m].shape[1]
    Ew_hat2 = EW2(st, m) + (1 - st["gamma"][m]) / Ealpha(st, m)    # E[w_chapeu^2]: parte slab + parte prior (s = 0)
    st["a_alpha"][m] = np.full(st["K"], st["pri"]["a0_alpha"] + D / 2)
    st["b_alpha"][m] = st["pri"]["b0_alpha"] + 0.5 * Ew_hat2.sum(axis=0)
    if log is not None:
        log.append(dict(passo="4.2", visao=m, a=st["a_alpha"][m][0], b=st["b_alpha"][m][0], Ealpha=Ealpha(st, m)[0]))


# ------------------------------------------------------------------ passo 4.3: theta
def atualiza_theta(st, m, log=None):
    D, g = st["Y"][m].shape[1], st["gamma"][m].sum(axis=0)
    st["a_theta"][m] = st["pri"]["a0_theta"] + g
    st["b_theta"][m] = st["pri"]["b0_theta"] + D - g
    if log is not None:
        log.append(dict(passo="4.3", visao=m, a=st["a_theta"][m][0], b=st["b_theta"][m][0],
                        Etheta=st["a_theta"][m][0] / (st["a_theta"][m][0] + st["b_theta"][m][0])))


# ------------------------------------------------------------------ passo 4.4: fatores Z
def atualiza_Z(st, log=None):
    """Passo 4.4. Mesmo truque do passo 4.1: com W fixo, para cada vista

        sum_d E[tau_d] E[w_dk] r^(k)_nd = (Y T E[W])_nk - sum_{j != k} z_nj (E[W]^T T E[W])_jk,   T = diag(E[tau])

    então Y T E[W] (N x K) e E[W]^T T E[W] (K x K) são calculados uma vez por vista.
    """
    YTW, G, prec = 0.0, 0.0, np.ones(st["K"])
    for m in st["visoes"]:
        Ew, tau = EW(st, m), Etau(st, m)
        TW = tau[:, None] * Ew
        YTW = YTW + st["Y"][m] @ TW                                  # N x K, somado nas vistas
        G = G + Ew.T @ TW                                            # K x K, somado nas vistas
        prec = prec + (tau[:, None] * EW2(st, m)).sum(axis=0)        # 1 + sum_m sum_d E[tau] E[w^2], por fator
    Z = st["Zmu"]
    for k in range(st["K"]):
        num = YTW[:, k] - Z @ G[:, k] + Z[:, k] * G[k, k]             # vetor sobre n
        st["Zs2"][k] = 1.0 / prec[k]
        Z[:, k] = num / prec[k]
        if log is not None and k == 0:
            log.append(dict(passo="4.4", k=k, n=0, precisao=prec[k], s2=1 / prec[k], mu=Z[0, k]))


# ------------------------------------------------------------------ passo 4.5: ruído tau
def erro_quadratico_esperado(st, m):
    """sum_n E[(y_nd - z_n w_d)^2] para cada d (fórmula do fim do passo 4.5)."""
    Y, Z, EZZ, Ew, Eww = st["Y"][m], st["Zmu"], EZ2(st), EW(st, m), EW2(st, m)
    res = Y - Z @ Ew.T
    return (res ** 2).sum(axis=0) + EZZ.sum(axis=0) @ Eww.T - (Z ** 2).sum(axis=0) @ (Ew ** 2).T


def atualiza_tau(st, m, log=None):
    sqe = erro_quadratico_esperado(st, m)
    st["a_tau"][m] = np.full_like(sqe, st["pri"]["a0_tau"] + st["N"] / 2)
    st["b_tau"][m] = st["pri"]["b0_tau"] + 0.5 * sqe
    if log is not None:
        log.append(dict(passo="4.5", visao=m, d=0, sqe=sqe[0], a=st["a_tau"][m][0], b=st["b_tau"][m][0],
                        Etau=Etau(st, m)[0]))


# ------------------------------------------------------------------ seção 5: ELBO
def _kl_gama(a, b, a0, b0):
    """E_q[log q] - E_q[log p] para q = Gama(a, b) e p = Gama(a0, b0)."""
    return ((a - a0) * digamma(a) - gammaln(a) + gammaln(a0) + a0 * (np.log(b) - np.log(b0)) + a * (b0 - b) / b).sum()


def elbo(st):
    p = st["pri"]
    # fatores: E[log p(z)] - E[log q(z)]
    L = (-0.5 * EZ2(st) + 0.5 * np.log(st["Zs2"]) + 0.5).sum()
    for m in st["visoes"]:
        D = st["Y"][m].shape[1]
        a_t, b_t = st["a_tau"][m], st["b_tau"][m]
        Elogtau = digamma(a_t) - np.log(b_t)
        # verossimilhança: E[log p(Y | Z, W, tau)]
        L += (st["N"] / 2 * (Elogtau - LOG2PI) - 0.5 * (a_t / b_t) * erro_quadratico_esperado(st, m)).sum()
        # pesos: E[log p(w_chapeu | alfa)] + E[log p(s | theta)] - E[log q(w_chapeu, s)]
        g, mu, s2 = st["gamma"][m], st["Wmu"][m], st["Ws2"][m]
        Ea = Ealpha(st, m); Eloga = digamma(st["a_alpha"][m]) - np.log(st["b_alpha"][m])
        L += (0.5 * Eloga - 0.5 * Ea * g * (mu ** 2 + s2) + 0.5 * g * np.log(s2) + 0.5 * g
              - 0.5 * (1 - g) * np.log(Ea)).sum()
        Elt = digamma(st["a_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        El1t = digamma(st["b_theta"][m]) - digamma(st["a_theta"][m] + st["b_theta"][m])
        gc = np.clip(g, 1e-12, 1 - 1e-12)
        L += (g * Elt + (1 - g) * El1t - gc * np.log(gc) - (1 - gc) * np.log(1 - gc)).sum()
        # hiperparâmetros: -KL(q || p) de alfa, theta e tau
        L -= _kl_gama(st["a_alpha"][m], st["b_alpha"][m], p["a0_alpha"], p["b0_alpha"])
        L -= _kl_gama(a_t, b_t, p["a0_tau"], p["b0_tau"])
        a, b = st["a_theta"][m], st["b_theta"][m]
        L -= ((a - p["a0_theta"]) * Elt + (b - p["b0_theta"]) * El1t - betaln(a, b) + betaln(p["a0_theta"], p["b0_theta"])).sum()
    return L


# ------------------------------------------------------------------ seção 5: variância explicada e poda
def r2(st):
    """R^2 de cada fator em cada vista: 1 - SQ(Y - z_k w_k^T) / SQ(Y).

    Expandindo o quadrado: SQ(Y - z_k w_k^T) = SQ(Y) - 2 z_k^T Y w_k + (z_k^T z_k)(w_k^T w_k), sem matriz N x D.
    """
    out, Z = {}, st["Zmu"]
    for m in st["visoes"]:
        Y, Ew = st["Y"][m], EW(st, m)
        sq = st.setdefault("sqY", {}).get(m)
        if sq is None:
            sq = st["sqY"][m] = (Y ** 2).sum()
        cruz = np.einsum("kd,dk->k", Z.T @ Y, Ew)                    # z_k^T Y w_k para cada k
        out[m] = 1 - (sq - 2 * cruz + (Z ** 2).sum(0) * (Ew ** 2).sum(0)) / sq
    return out


def poda(st, limiar):
    """Remove fatores cujo R^2 máximo entre as vistas fica abaixo do limiar."""
    R = r2(st)
    manter = np.max(np.vstack([R[m] for m in st["visoes"]]), axis=0) >= limiar
    if manter.all():
        return 0
    st["Zmu"], st["Zs2"] = st["Zmu"][:, manter], st["Zs2"][manter]
    for m in st["visoes"]:
        for c in ("Wmu", "Ws2", "gamma"):
            st[c][m] = st[c][m][:, manter]
        for c in ("a_alpha", "b_alpha", "a_theta", "b_theta"):
            st[c][m] = st[c][m][manter]
    st["K"] = int(manter.sum())
    return int((~manter).sum())


# ------------------------------------------------------------------ seção 5: o algoritmo
def iteracao(st, log=None):
    """Uma varredura completa, na ordem do documento: W, alfa, theta (por vista), Z, tau (por vista)."""
    for m in st["visoes"]:
        atualiza_W(st, m, log)
        atualiza_alpha(st, m, log)
        atualiza_theta(st, m, log)
    atualiza_Z(st, log)
    for m in st["visoes"]:
        atualiza_tau(st, m, log)


def mofa(visoes, K=10, max_iter=1000, tol=1e-6, limiar_poda=0.005, inicio_poda=5, semente=0, verbose=True,
         mostrar_a_cada=25, **hiper):
    t0 = time.time()
    st = inicializa(visoes, K, semente, **hiper)
    historico = []
    for it in range(max_iter):
        iteracao(st)
        if it >= inicio_poda:
            removidos = poda(st, limiar_poda)
            if removidos and verbose:
                print(f"iter {it}: {removidos} fator(es) podado(s), K = {st['K']}")
        L = elbo(st)
        historico.append(L)
        if verbose and (it + 1) % mostrar_a_cada == 0:
            print(f"iter {it + 1}: ELBO = {L:.1f}, K = {st['K']}, {time.time() - t0:.0f} s")
        if it > 0 and abs(L - historico[-2]) / abs(historico[-2]) < tol:
            break
    if verbose:
        print(f"convergiu em {it + 1} iterações ({time.time() - t0:.0f} s), ELBO = {L:.1f}, K final = {st['K']}")
    return {"Z": st["Zmu"], "W": {m: EW(st, m) for m in st["visoes"]}, "gamma": st["gamma"],
            "alpha": {m: Ealpha(st, m) for m in st["visoes"]}, "tau": {m: Etau(st, m) for m in st["visoes"]},
            "r2": r2(st), "elbo": np.array(historico), "estado": st}
