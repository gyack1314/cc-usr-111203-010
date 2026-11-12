"""独立暴力验证：按事件数做精确分布卷积，与 aggregate 的递推交叉核对。

本模块故意使用与 aggregate.py 不同的实现路径（Fraction 字典上的逐事件
卷积 + 独立浮点 Panjer），仅用于测试，不进入生产命令。
"""
from fractions import Fraction
import math

from climate_data.aggregate import (
    annual_contract, parse_inputs, event_payments, severity_envelope)


def brute_force_annual(v, n_max):
    """对 N=0..n_max 逐层卷积事件赔付分布，再施加年度合同。

    与 Panjer 完全独立的路径：状态是 Fraction 金额 -> Fraction 条件概率，
    最后按 Poisson 事件数权重混合。返回 (Y 原子 dict, 余尾概率 float)。
    """
    cfg = parse_inputs(v)
    events = event_payments(cfg)
    one_step = {}
    for e in events:
        one_step[e['event_payment']] = (
            one_step.get(e['event_payment'], Fraction(0))
            + e['probability'])
    D, L = cfg['annual_deductible'], cfg['annual_limit']
    lam = cfg['lambda']

    # cond[n]: 给定 N=n 时总赔付 S 的精确条件分布
    cond = {Fraction(0): Fraction(1)}
    atoms = {}

    def absorb(n, dist):
        w = Fraction.from_float(poisson_pmf_frac(lam, n))
        for s, p in dist.items():
            y = annual_contract(s, D, L)
            atoms[y] = atoms.get(y, Fraction(0)) + w * p

    absorb(0, cond)
    for n in range(1, n_max + 1):
        nxt = {}
        for s, p in cond.items():
            for x, q in one_step.items():
                nxt[s + x] = nxt.get(s + x, Fraction(0)) + p * q
        cond = nxt
        absorb(n, cond)
    covered = math.fsum(float(p) for p in atoms.values())
    return atoms, 1.0 - covered


def poisson_pmf_frac(lam, n):
    """浮点 lambda 的 Poisson pmf（独立实现）。"""
    return math.exp(-lam) * lam ** n / math.factorial(n)


def independent_panjer(lam, f_float, K):
    """教科书式直接递推，独立于 aggregate 的 log 路径。"""
    p = [0.0] * (K + 1)
    p[0] = math.exp(lam * (f_float[0] - 1.0))
    for n in range(1, K + 1):
        s = 0.0
        for j in range(1, n + 1):
            s += (lam * j / n) * f_float[j] * p[n - j]
        p[n] = s
    return p


def envelope_cdf_check(v):
    """核对 floor/ceil 包络确实夹住真实严重度且各自质量为1。"""
    cfg = parse_inputs(v)
    events = event_payments(cfg)
    lo, up = severity_envelope(events, cfg['h'], cfg['K'])
    lo_mass = sum((lo.get(n, Fraction(0)) for n in range(cfg['K'] + 1)),
                  Fraction(0)) + lo['tail']
    up_mass = sum((up.get(n, Fraction(0)) for n in range(cfg['K'] + 1)),
                  Fraction(0)) + up['tail']
    assert lo_mass == 1 and up_mass == 1
    for e in events:
        x = e['event_payment']
        nf, nc = x // cfg['h'], (x // cfg['h'] + (1 if x % cfg['h'] else 0))
        assert nf * cfg['h'] <= x <= nc * cfg['h']
    return True
