"""年度聚合赔付分布：复合Poisson、Panjer递推与正式分位/TVaR界。

口径见 docs/data-contract.md：
- 一年事件数 N~Poisson(lambda)；每次事件按场景表的精确概率独立抽取，
  同一场景的多地区损失同时发生（地区间不独立）。
- 先在场景内按每地区免赔额与赔付上限算地区赔付，相加得单次赔付 X。
- 年度总赔付 S=sum_{i=1..N} X_i；年度合同在 S 之上施加
  Y=min(max(S-D,0),L)，两层顺序不可调换。
- 每次赔付向下/向上取到 h 格点形成两条随机序包络；复合Poisson用Panjer
  递推算 0..K 格概率，含零额严重度质量 f0，K 以外保留尾质量，不重归一。
- 大 lambda 下 p0 浮点下溢时使用 log-domain 稳定路径；另以事件数截断的
  卷积级数独立核对，余尾用Poisson余尾界量化。
全部计算只用标准库；金额的精确口径用 fractions.Fraction，数值递推用 float。
"""
import math
from fractions import Fraction

from .io import fraction, integer, number

# ---------------------------------------------------------------------------
# 输入解析与单场景赔付（精确有理数）
# ---------------------------------------------------------------------------

def parse_inputs(v):
    """校验并解析资料，返回统一口径的参数字典。"""
    lam = number(v['annual_rate'])
    if lam < 0:
        raise ValueError('事件率不能负')
    regions = set(v['regional_contract'])
    scenarios = v['scenarios']
    if not scenarios:
        raise ValueError('至少需要一个联合场景')
    parsed = []
    prob_total = Fraction(0)
    for s in scenarios:
        p = fraction(s['probability'])
        if p < 0:
            raise ValueError('联合场景概率不能负')
        if set(s['losses']) != regions:
            raise ValueError('联合场景的地区集合不完整')
        losses = {r: fraction(x) for r, x in s['losses'].items()}
        if any(x < 0 for x in losses.values()):
            raise ValueError('损失不能负')
        prob_total += p
        parsed.append({'id': s['id'], 'probability': p, 'losses': losses})
    if prob_total != 1:
        raise ValueError('场景概率和必须为1')
    regional = {}
    for r, c in v['regional_contract'].items():
        d, lim = fraction(c['deductible']), fraction(c['limit'])
        if d < 0 or lim < 0:
            raise ValueError('地区合同金额不能负')
        regional[r] = (d, lim)
    ac = v['annual_contract']
    ad, al = fraction(ac['deductible']), fraction(ac['limit'])
    if ad < 0 or al < 0:
        raise ValueError('年度合同金额不能负')
    h = fraction(v['step'])
    if h <= 0:
        raise ValueError('格点步长应正')
    K = integer(v['max_grid'], minimum=0)
    alphas = v['alpha']
    if not isinstance(alphas, list) or not alphas:
        raise ValueError('alpha 应为非空列表')
    for a in alphas:
        number(a)
        if not (0 < a < 1):
            raise ValueError('alpha 必须严格在0和1之间')
    tol = number(v['numeric_tolerance'])
    if not (0 < tol < 1):
        raise ValueError('numeric_tolerance 应在0和1之间')
    terms = integer(v['poisson_check_terms'], minimum=1)
    return {
        'lambda': lam,
        'scenarios': parsed,
        'regions': sorted(regions),
        'regional': regional,
        'annual_deductible': ad,
        'annual_limit': al,
        'h': h,
        'K': K,
        'alphas': [float(a) for a in alphas],
        'tol': float(tol),
        'check_terms': terms,
    }


def regional_payment(loss, deductible, limit):
    """单地区赔付 min(max(loss-deductible,0), limit)。"""
    x = loss - deductible
    if x <= 0:
        return Fraction(0)
    return x if x < limit else limit


def event_payments(cfg):
    """每个场景的单次事件赔付：场景内逐地区处理后求和（精确分数）。"""
    out = []
    for s in cfg['scenarios']:
        total = Fraction(0)
        per_region = {}
        for r in cfg['regions']:
            d, lim = cfg['regional'][r]
            pay = regional_payment(s['losses'][r], d, lim)
            per_region[r] = pay
            total += pay
        out.append({'id': s['id'], 'probability': s['probability'],
                    'regional_payment': per_region, 'event_payment': total})
    return out


# ---------------------------------------------------------------------------
# 严重度格点包络（精确分数概率；超出 K 的严重度质量单列，不丢弃）
# ---------------------------------------------------------------------------

def severity_envelope(events, h, K):
    """构造下/上格点严重度分布。

    返回两个 dict：键为格点 n（0..K），值为该格点的精确概率 Fraction；
    另含键 'tail' 表示落在 K 以外的严重度概率质量，'f0' 为零额质量。
    下包络 floor(x/h)，上包络 ceil(x/h)；x=0 时两者都落 0 格。
    """
    lo, up = {}, {}
    lo_tail = up_tail = Fraction(0)
    for e in events:
        p, x = e['probability'], e['event_payment']
        n_floor = x // h
        n_ceil = n_floor if (n_floor * h == x) else n_floor + 1
        if n_floor <= K:
            lo[n_floor] = lo.get(n_floor, Fraction(0)) + p
        else:
            lo_tail += p
        if n_ceil <= K:
            up[n_ceil] = up.get(n_ceil, Fraction(0)) + p
        else:
            up_tail += p
    lo['tail'] = lo_tail
    up['tail'] = up_tail
    lo['f0'] = lo.get(0, Fraction(0))
    up['f0'] = up.get(0, Fraction(0))
    return lo, up


# ---------------------------------------------------------------------------
# Panjer 递推（复合Poisson）
# ---------------------------------------------------------------------------

def _logsumexp(pairs):
    """log(sum(exp(a)))，输入为 (log_a, 系数) 对，系数非负；空集返回 -inf。"""
    m = max((a for a, w in pairs if w > 0.0), default=None)
    if m is None:
        return -math.inf
    return m + math.log(math.fsum(w * math.exp(a - m) for a, w in pairs))


def panjer_log(lam, f, K):
    """log-domain Panjer 递推，p0 下溢时仍稳定。

    f: 长度 K+1 的严重度概率浮点数组（f[0] 可为正，允许 f 质量和<1，
    缺失部分自动进入递推尾质量）。返回 (log_p, diagnostics)。
    """
    log_p = [0.0] * (K + 1)
    log_p[0] = lam * (f[0] - 1.0)  # log p0 = lambda*(f0-1)
    log_f = [math.log(f[j]) if f[j] > 0.0 else math.nan for j in range(K + 1)]
    underflow_zero = []
    if lam == 0.0:
        # N 恒为 0：p0=1，其余格结构为零
        log_p = [0.0] + [-math.inf] * K
        diag = {
            'path': 'log', 'log_p0': 0.0, 'p0': 1.0,
            'p0_underflow': False,
            'positive_terms_flushed_to_zero': [], 'flushed_term_count': 0,
        }
        return log_p, diag
    log_lam = math.log(lam)
    support = [j for j in range(1, K + 1) if f[j] > 0.0]
    for n in range(1, K + 1):
        pairs = []
        for j in support:
            if j > n:
                break
            if not math.isinf(log_p[n - j]):
                # log(lambda/n * j * f_j * p_{n-j})
                pairs.append((log_lam - math.log(n) + math.log(j)
                              + log_f[j] + log_p[n - j], 1.0))
        z = _logsumexp(pairs)
        log_p[n] = z
        # 严格为正的项被舍入到 subnormal 以下时记账（-inf 是结构零，不算）
        if not math.isinf(z) and z < math.log(np_min_subnormal()):
            underflow_zero.append(n)
    # p0 下溢：log_p0 有限但 exp 已得 0.0
    p0_underflow = math.exp(log_p[0]) == 0.0
    diag = {
        'path': 'log',
        'log_p0': log_p[0],
        'p0': math.exp(log_p[0]),
        'p0_underflow': p0_underflow,
        'positive_terms_flushed_to_zero': underflow_zero[:10],
        'flushed_term_count': len(underflow_zero),
    }
    return log_p, diag


def np_min_subnormal():
    """最小正 subnormal 浮点（5e-324），避免 import numpy。"""
    return 5e-324


def panjer_direct(lam, f, K):
    """直接浮点 Panjer 递推；p0 下溢时整条退化，仅作核对用途。"""
    p = [0.0] * (K + 1)
    p[0] = math.exp(lam * (f[0] - 1.0))
    support = [j for j in range(1, K + 1) if f[j] > 0.0]
    for n in range(1, K + 1):
        acc = 0.0
        for j in support:
            if j > n:
                break
            acc += (lam / n) * j * f[j] * p[n - j]
        p[n] = acc
    diag = {
        'path': 'direct',
        'log_p0': lam * (f[0] - 1.0),
        'p0': p[0],
        'p0_underflow': p[0] == 0.0,
        'positive_terms_flushed_to_zero':
            [n for n in range(K + 1) if p[n] == 0.0][:10],
        'flushed_term_count': sum(1 for x in p if x == 0.0),
    }
    return p, diag


def run_panjer(lam, severity, K, tol):
    """对一条严重度包络跑复合Poisson递推并做质量账。

    返回 dict：格点概率 p、log_p、尾质量（含未算格点与数值残差）、诊断。
    尾质量 = 1 - sum(p)，绝不重归一。
    """
    f = [float(severity.get(n, Fraction(0))) for n in range(K + 1)]
    f_sev_tail = float(severity['tail'])
    log_p, diag = panjer_log(lam, f, K)
    p = [0.0 if math.isinf(z) or math.isnan(z) else math.exp(z)
         for z in log_p]
    computed_mass = math.fsum(p)
    if not math.isfinite(computed_mass):  # 极端下溢账：已算质量按0计
        computed_mass = 0.0
    tail_mass = 1.0 - computed_mass  # 未计算尾质量（含舍入残差），绝不重归一
    # 直接路径仅在不会整体下溢时可用于独立数值核对
    direct = None
    if 0.0 < math.exp(lam * (f[0] - 1.0)):
        pd, ddiag = panjer_direct(lam, f, K)
        direct = {
            'p': pd,
            'max_abs_diff': max(abs(a - b) for a, b in zip(p, pd)),
            'direct_diag': ddiag,
        }
    # 理论零事件概率与递推 p0 的对照（不能用它替换递推 p0）
    diag['theoretical_no_event'] = math.exp(-lam)
    diag['severity_tail_mass'] = f_sev_tail
    diag['computed_mass_0_to_K'] = computed_mass
    diag['tail_mass_beyond_K'] = max(tail_mass, 0.0)
    diag['roundoff_residual'] = abs(tail_mass) if tail_mass < 0 else 0.0
    diag['direct_path_available'] = direct is not None
    return {'p': p, 'log_p': log_p, 'f': f, 'diag': diag,
            'direct': direct, 'severity_tail': f_sev_tail}


# ---------------------------------------------------------------------------
# 事件数截断卷积（小 lambda 独立核对）与 Poisson 余尾界
# ---------------------------------------------------------------------------

def poisson_pmf(lam, n):
    if lam == 0:
        return 1.0 if n == 0 else 0.0
    return math.exp(-lam) * lam ** n / math.factorial(n)


def poisson_remainder_bound(lam, m):
    """P(N>m) 的可计算上界：1 - sum_{n=0..m} pmf（fsum 保精度）。"""
    if lam == 0:
        return 0.0
    terms = [poisson_pmf(lam, n) for n in range(m + 1)]
    return max(0.0, 1.0 - math.fsum(terms))


def truncated_convolution(lam, f, K, m):
    """事件数截断到 m 的卷积级数：p_n=sum_{r=0..m} P(N=r) * f^{*r}_n。

    f[0] 可正（零额事件），r=0 时只有 0 格质量 1。卷积只在 f 的非零
    支撑集上推进，避免 O(m K^2) 空转。
    """
    support = [j for j in range(K + 1) if f[j] > 0.0]
    conv = [0.0] * (K + 1)
    dist = [0.0] * (K + 1)
    dist[0] = 1.0  # f^{*0}
    for r in range(m + 1):
        w = poisson_pmf(lam, r)
        if w > 0.0:
            for n in range(K + 1):
                conv[n] += w * dist[n]
        if r == m:
            break
        nxt = [0.0] * (K + 1)
        for i in range(K + 1):
            if dist[i] == 0.0:
                continue
            for j in support:
                if i + j > K:
                    break
                nxt[i + j] += dist[i] * f[j]
        dist = nxt
    return conv


def cross_check(lam, f, K, panjer_p, terms, tol):
    """截断卷积 vs Panjer 的独立核对，差异用 Poisson 余尾界解释。"""
    m = terms
    conv = truncated_convolution(lam, f, K, m)
    rem = poisson_remainder_bound(lam, m)
    diffs = [abs(a - b) for a, b in zip(panjer_p, conv)]
    max_diff = max(diffs)
    # 每个格点的差异不应超过被截掉的事件数总质量，外加浮点容差
    ok = max_diff <= rem + tol
    return {
        'truncation_terms': m,
        'poisson_remainder_bound_P_N_gt_m': rem,
        'max_abs_diff_vs_panjer': max_diff,
        'per_grid_diffs': diffs,
        'passes_within_remainder_plus_tol': ok,
    }


# ---------------------------------------------------------------------------
# 年度合同与指标（VaR 左分位、TVaR 原子分摊）
# ---------------------------------------------------------------------------

def annual_contract(s, deductible, limit):
    """Y = min(max(S-D,0), L)，作用在年度总赔付 S 上（两层不可换）。"""
    y = s - deductible
    if y <= 0:
        return Fraction(0)
    return y if y < limit else limit


def complete_distribution(p, tail_mass, tail_value, h, K, contract):
    """把尾质量放到指定赔付值，得到经年度合同映射后的完整原子分布。

    p[n] 的总赔付金额 n*h 先过年度合同 g(·)=min(max(·-D,0),L)，同值原子
    合并；尾质量统一放 tail_value（调用方给极小或极大可行值，且本身已是
    合同赔付口径）。返回 [(value_Fraction, mass)]，按赔付升序。
    """
    merged = {}
    for n in range(K + 1):
        if p[n] > 0.0:
            value = contract(n * h)
            merged[value] = merged.get(value, 0.0) + p[n]
    if tail_mass > 0.0:
        merged[tail_value] = merged.get(tail_value, 0.0) + tail_mass
    return sorted(merged.items(), key=lambda kv: kv[0])


def var_tvar(atoms, alpha):
    """给定完整原子分布，算左分位 VaR 与原子分摊 TVaR。

    VaR = 使累计概率首次达到 alpha 的最小赔付 q；
    TVaR = (E[Y 1_{Y>q}] + q*(F(q)-alpha)) / (1-alpha)
         = q + E[(Y-q)_+]/(1-alpha)。
    """
    total = math.fsum(m for _, m in atoms)
    cum = 0.0
    q = None
    f_q = 0.0
    for value, mass in atoms:
        cum += mass
        if q is None and cum >= alpha:  # 左分位：累计首次达到 alpha
            q = value
            f_q = cum
    if q is None:  # 舍入使累计略低于 alpha 时退化为最大原子
        q, f_q = atoms[-1][0], total
    excess = math.fsum(mass * float(value - q) for value, mass in atoms
                       if value > q)
    tvar = float(q) + excess / (1.0 - alpha)
    mean = math.fsum(mass * float(value) for value, mass in atoms)
    return {'var': q, 'tvar': tvar, 'mean': mean, 'F_q': f_q}


def envelope_indicators(run, cfg, alpha):
    """一条包络（格点分布+尾质量）下指标的正式可行区间。

    尾质量在年度合同下的可行赔付为 [g(Kh), L]（g 单调）；
    把尾质量分别放到两个极端，得到该包络内的最小/最大指标分布。
    """
    h, K = cfg['h'], cfg['K']
    D, L = cfg['annual_deductible'], cfg['annual_limit']

    def contract(s):
        return annual_contract(s, D, L)

    tail_mass = run['diag']['tail_mass_beyond_K']
    tail_lo_value = contract(K * h)   # g(Kh)，尾域可行赔付下界
    tail_hi_value = L                 # 年度上限，尾域可行赔付上界
    low_atoms = complete_distribution(run['p'], tail_mass, tail_lo_value,
                                      h, K, contract)
    high_atoms = complete_distribution(run['p'], tail_mass, tail_hi_value,
                                       h, K, contract)
    lo_stats = var_tvar(low_atoms, alpha)
    hi_stats = var_tvar(high_atoms, alpha)
    return {
        'tail_mass': tail_mass,
        'tail_payment_lower': str(tail_lo_value),
        'tail_payment_upper': str(tail_hi_value),
        'tail_degenerate': tail_lo_value == tail_hi_value,
        'var_lower': str(lo_stats['var']),
        'var_upper': str(hi_stats['var']),
        'tvar_lower': lo_stats['tvar'],
        'tvar_upper': hi_stats['tvar'],
        'mean_lower': lo_stats['mean'],
        'mean_upper': hi_stats['mean'],
        'F_q_lower_placement': lo_stats['F_q'],
        'F_q_upper_placement': hi_stats['F_q'],
    }


def _fr(x):
    """字符串化的有理数/浮点转回 Fraction（VaR 端点是分数字符串）。"""
    return Fraction(x) if isinstance(x, str) else Fraction(float(x)).limit_denominator(10 ** 18)


def combine_bounds(lo_env, up_env, cfg, alpha):
    """合并两条随机序包络，给最终区间与确定性判定。"""
    tol = cfg['tol']
    h, L = cfg['h'], cfg['annual_limit']
    mean_lo = min(lo_env['mean_lower'], up_env['mean_lower'])
    mean_hi = max(lo_env['mean_upper'], up_env['mean_upper'])
    # 数值误差账：按容差给出均值/TVaR 的保守展宽
    mean_budget = tol * float(L)
    tvar_budget = tol * float(L) / (1.0 - alpha)
    mean_lo = max(0.0, mean_lo - mean_budget)
    mean_hi = mean_hi + mean_budget
    var_lo = min(_fr(lo_env['var_lower']), _fr(up_env['var_lower']))
    var_hi = max(_fr(lo_env['var_upper']), _fr(up_env['var_upper']))
    tvar_lo = min(lo_env['tvar_lower'], up_env['tvar_lower']) - tvar_budget
    tvar_hi = max(lo_env['tvar_upper'], up_env['tvar_upper']) + tvar_budget
    # 确定性判定：尾质量放置是否改变该包络的分位，以及两包络的格点展宽
    tail_matters = (max(lo_env['tail_mass'], up_env['tail_mass']) > tol)
    tail_degenerate = lo_env['tail_degenerate'] and up_env['tail_degenerate']
    var_tail_undetermined = (
        _fr(lo_env['var_lower']) != _fr(lo_env['var_upper']) or
        _fr(up_env['var_lower']) != _fr(up_env['var_upper']))
    mean_tail_undetermined = (
        lo_env['mean_lower'] != lo_env['mean_upper'] or
        up_env['mean_lower'] != up_env['mean_upper'])
    # 剔除尾放置影响后，纯格点（h）造成的包络展宽：各包络取同一种尾放置
    grid_var_lo = min(_fr(lo_env['var_lower']), _fr(up_env['var_lower']))
    grid_var_hi = max(_fr(lo_env['var_lower']), _fr(up_env['var_lower']))
    grid_gap = grid_var_hi - grid_var_lo
    var_determined = (not var_tail_undetermined) and grid_gap <= h
    advice = []
    if var_tail_undetermined:
        advice.append('尾质量足以改变分位：需要更大的 max_grid K')
    elif tail_matters and not tail_degenerate:
        advice.append('尾质量超过容差且尾域赔付非退化：建议增大 K 复核')
    if grid_gap > h:
        advice.append('上下包络差距来自格点：需要更小的 step h')
    if not advice:
        advice.append('区间宽度不超过一个格点且尾质量可忽略（或尾域退化为单点），指标已夹住')
    return {
        'alpha': alpha,
        'mean_interval': [mean_lo, mean_hi],
        'mean_determined': not mean_tail_undetermined,
        'var_interval': [str(var_lo), str(var_hi)],
        'tvar_interval': [max(0.0, tvar_lo), tvar_hi],
        'var_determined': var_determined,
        'tail_mass_max': max(lo_env['tail_mass'], up_env['tail_mass']),
        'tail_degenerate': tail_degenerate,
        'advice': advice,
    }


# ---------------------------------------------------------------------------
# 诊断：零额事件、上限原子、非整步金额、下溢
# ---------------------------------------------------------------------------

def diagnostics(cfg, events, sev_lo, sev_up, run_lo, run_up):
    h, K = cfg['h'], cfg['K']
    D, L = cfg['annual_deductible'], cfg['annual_limit']
    zero_events = [e['id'] for e in events if e['event_payment'] == 0]
    off_grid = []
    for e in events:
        x = e['event_payment']
        if x % h != 0:
            off_grid.append({'id': e['id'], 'event_payment': str(x),
                             'floor_grid': str(x // h),
                             'ceil_grid': str(x // h + 1)})
    # 上限原子：哪些总赔付格点经年度合同映射到 L
    def contract(s):
        return annual_contract(s, D, L)
    cap_grids = [n for n in range(K + 1) if contract(n * h) == L]
    cap_mass_lo = math.fsum(run_lo['p'][n] for n in cap_grids)
    cap_mass_up = math.fsum(run_up['p'][n] for n in cap_grids)
    return {
        'zero_payment_scenarios': zero_events,
        'zero_event_probability_exact': str(sum(
            (e['probability'] for e in events if e['event_payment'] == 0),
            Fraction(0))),
        'non_integer_step_amounts': off_grid,
        'cap_atom': {
            'annual_limit': str(L),
            'grids_mapping_to_cap': [cap_grids[0], cap_grids[-1]] if cap_grids else None,
            'computed_cap_mass_lower': cap_mass_lo,
            'computed_cap_mass_upper': cap_mass_up,
            'tail_also_maps_to_cap': annual_contract(K * h, D, L) == L,
        },
        'p0_underflow': {
            'lower_envelope': run_lo['diag']['p0_underflow'],
            'upper_envelope': run_up['diag']['p0_underflow'],
            'stable_path': 'log-domain Panjer，p0 以 log_p0 保留，不置零重归一',
        },
    }


def annual_aggregate(v):
    """主入口：从资料对象算出可逐层追溯的年度聚合分布与指标界。"""
    cfg = parse_inputs(v)
    events = event_payments(cfg)
    sev_lo, sev_up = severity_envelope(events, cfg['h'], cfg['K'])
    run_lo = run_panjer(cfg['lambda'], sev_lo, cfg['K'], cfg['tol'])
    run_up = run_panjer(cfg['lambda'], sev_up, cfg['K'], cfg['tol'])

    # 精确参考值（不依赖格点）：E[S]=lambda*E[X]，应落在包络之间
    ex = sum((e['probability'] * e['event_payment'] for e in events),
             Fraction(0))
    lam_frac = Fraction(cfg['lambda']).limit_denominator(10 ** 18)
    exact_mean_S = lam_frac * ex

    checks = []
    for name, run in (('lower', run_lo), ('upper', run_up)):
        cc = cross_check(cfg['lambda'], run['f'], cfg['K'], run['p'],
                         cfg['check_terms'], cfg['tol'])
        cc['envelope'] = name
        checks.append(cc)

    indicators = []
    for alpha in cfg['alphas']:
        lo_env = envelope_indicators(run_lo, cfg, alpha)
        up_env = envelope_indicators(run_up, cfg, alpha)
        indicators.append(combine_bounds(lo_env, up_env, cfg, alpha))

    return {
        'parameters': {
            'lambda': cfg['lambda'], 'h': str(cfg['h']), 'K': cfg['K'],
            'annual_deductible': str(cfg['annual_deductible']),
            'annual_limit': str(cfg['annual_limit']),
            'tolerance': cfg['tol'],
        },
        'event_payments': [
            {'id': e['id'], 'probability': str(e['probability']),
             'regional_payment': {r: str(x) for r, x in e['regional_payment'].items()},
             'event_payment': str(e['event_payment'])}
            for e in events
        ],
        'severity': {
            'lower': {str(n): str(sev_lo.get(n, Fraction(0)))
                      for n in range(cfg['K'] + 1) if sev_lo.get(n, 0) != 0},
            'upper': {str(n): str(sev_up.get(n, Fraction(0)))
                      for n in range(cfg['K'] + 1) if sev_up.get(n, 0) != 0},
            'f0_lower': str(sev_lo['f0']), 'f0_upper': str(sev_up['f0']),
            'severity_tail_lower': str(sev_lo['tail']),
            'severity_tail_upper': str(sev_up['tail']),
        },
        'panjer': {
            'lower': _run_report(run_lo),
            'upper': _run_report(run_up),
        },
        'exact_reference': {
            'E_event_payment_exact': str(ex),
            'E_annual_sum_before_annual_contract_exact': str(exact_mean_S),
        },
        'poisson_cross_check': checks,
        'indicators': indicators,
        'diagnostics': diagnostics(cfg, events, sev_lo, sev_up, run_lo, run_up),
    }


def _run_report(run):
    d = run['diag']
    return {
        'path': d['path'],
        'log_p0': d['log_p0'],
        'p0': d['p0'],
        'theoretical_no_event_exp_neg_lambda': d['theoretical_no_event'],
        'p0_underflow': d['p0_underflow'],
        'computed_mass_0_to_K': d['computed_mass_0_to_K'],
        'tail_mass_beyond_K': d['tail_mass_beyond_K'],
        'severity_tail_mass': d['severity_tail_mass'],
        'roundoff_residual': d['roundoff_residual'],
        'direct_path_available': d['direct_path_available'],
        'direct_max_abs_diff': None if run['direct'] is None
        else run['direct']['max_abs_diff'],
        'flushed_term_count': d['flushed_term_count'],
        'grid_probabilities': run['p'],
        'log_grid_probabilities': [
            None if math.isinf(z) else z for z in run['log_p']],
    }
