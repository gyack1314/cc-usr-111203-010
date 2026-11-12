"""年度聚合分布、Panjer递推、尾界与TVaR原子分摊的核对。"""
import math
import unittest
from fractions import Fraction

from climate_data.io import load
from climate_data.aggregate import (
    annual_aggregate, annual_contract, cross_check, diagnostics,
    envelope_indicators, event_payments, parse_inputs,
    poisson_remainder_bound, regional_payment, run_panjer,
    severity_envelope, truncated_convolution, var_tvar,
)
from tests.brute_force import (
    brute_force_annual, envelope_cdf_check, independent_panjer)


def base_input(**overrides):
    v = {
        'annual_rate': 2.0,
        'scenarios': [
            {'id': 'none', 'probability': '1/5',
             'losses': {'east': '0', 'west': '0'}},
            {'id': 'river', 'probability': '1/2',
             'losses': {'east': '21/2', 'west': '5'}},
            {'id': 'storm', 'probability': '3/10',
             'losses': {'east': '25', 'west': '41/2'}},
        ],
        'regional_contract': {
            'east': {'deductible': '2', 'limit': '18'},
            'west': {'deductible': '3', 'limit': '12'}},
        'annual_contract': {'deductible': '10', 'limit': '40'},
        'step': '1', 'max_grid': 250,
        'alpha': [0.9, 0.95, 0.99],
        'numeric_tolerance': 1e-10, 'poisson_check_terms': 30,
    }
    v.update(overrides)
    return v


class ContractLayerTests(unittest.TestCase):
    def test_regional_payment_formula(self):
        self.assertEqual(regional_payment(Fraction(1), Fraction(2),
                                          Fraction(18)), 0)
        self.assertEqual(regional_payment(Fraction(5), Fraction(2),
                                          Fraction(18)), 3)
        self.assertEqual(regional_payment(Fraction(999), Fraction(2),
                                          Fraction(18)), 18)

    def test_event_payment_regions_not_independent(self):
        # 同一场景两地区同时赔付，先地区后求和
        cfg = parse_inputs(base_input())
        ev = {e['id']: e for e in event_payments(cfg)}
        # river: east min(10.5-2,18)=8.5=17/2; west min(5-3,12)=2
        self.assertEqual(ev['river']['regional_payment']['east'],
                         Fraction(17, 2))
        self.assertEqual(ev['river']['regional_payment']['west'], 2)
        self.assertEqual(ev['river']['event_payment'], Fraction(21, 2))
        # storm: east 封顶18; west min(20.5-3,12)=12 封顶；合计30
        self.assertEqual(ev['storm']['event_payment'], 30)
        self.assertEqual(ev['none']['event_payment'], 0)

    def test_annual_layer_order(self):
        # 年度合同作用在事件赔付之和之上，不能与地区层调换
        self.assertEqual(annual_contract(Fraction(0), 10, 40), 0)
        self.assertEqual(annual_contract(Fraction(15), 10, 40), 5)
        self.assertEqual(annual_contract(Fraction(60), 10, 40), 40)
        # 免赔吃掉全部的情形
        self.assertEqual(annual_contract(Fraction(10), 10, 40), 0)

    def test_probability_mass_and_envelope_order(self):
        self.assertTrue(envelope_cdf_check(base_input()))


class SeverityAndPanjerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = parse_inputs(base_input())
        self.events = event_payments(self.cfg)
        self.lo, self.up = severity_envelope(
            self.events, self.cfg['h'], self.cfg['K'])

    def test_floor_ceil_grids_with_f0(self):
        # f0=1/5 必须保留，不得被 exp(-lambda) 替换
        self.assertEqual(self.lo['f0'], Fraction(1, 5))
        self.assertEqual(self.up['f0'], Fraction(1, 5))
        # river 21/2: floor 10, ceil 11；storm 30 整格
        self.assertEqual(self.lo[10], Fraction(1, 2))
        self.assertEqual(self.up[11], Fraction(1, 2))
        self.assertEqual(self.lo[30], Fraction(3, 10))
        self.assertEqual(self.up[30], Fraction(3, 10))
        self.assertEqual(self.lo['tail'], 0)

    def test_panjer_p0_formula_and_mass_book(self):
        run = run_panjer(2.0, self.lo, self.cfg['K'], 1e-10)
        # p0 = exp(lambda*(f0-1)) = exp(2*(-4/5))=exp(-1.6)
        self.assertAlmostEqual(run['p'][0], math.exp(-1.6), places=14)
        self.assertNotAlmostEqual(run['p'][0], math.exp(-2.0))
        # 0..K 质量 + 尾质量 = 1，不重归一
        self.assertAlmostEqual(
            run['diag']['computed_mass_0_to_K']
            + run['diag']['tail_mass_beyond_K'], 1.0, places=12)
        # 所有概率非负
        self.assertTrue(all(x >= 0.0 for x in run['p']))

    def test_log_path_matches_independent_direct_recursion(self):
        run = run_panjer(2.0, self.lo, self.cfg['K'], 1e-10)
        direct = independent_panjer(2.0, run['f'], self.cfg['K'])
        for a, b in zip(run['p'], direct):
            self.assertAlmostEqual(a, b, places=14)

    def test_severity_tail_mass_is_kept_not_renormalized(self):
        # K=5：所有正严重度都在格点之外，f 只剩 f0
        lo, up = severity_envelope(self.events, self.cfg['h'], 5)
        run = run_panjer(2.0, lo, 5, 1e-10)
        self.assertAlmostEqual(run['p'][0], math.exp(-1.6), places=14)
        self.assertGreater(run['diag']['tail_mass_beyond_K'], 0.79)
        self.assertAlmostEqual(
            run['diag']['computed_mass_0_to_K']
            + run['diag']['tail_mass_beyond_K'], 1.0, places=12)

    def test_lambda_zero_is_degenerate(self):
        run = run_panjer(0.0, self.lo, self.cfg['K'], 1e-10)
        self.assertEqual(run['p'][0], 1.0)
        self.assertTrue(all(x == 0.0 for x in run['p'][1:]))
        self.assertEqual(run['diag']['tail_mass_beyond_K'], 0.0)
        self.assertFalse(run['diag']['p0_underflow'])

    def test_large_lambda_p0_underflow_stable(self):
        run = run_panjer(1000.0, self.lo, 200, 1e-10)
        self.assertTrue(run['diag']['p0_underflow'])
        self.assertEqual(run['p'][0], 0.0)          # 浮点确实下溢
        self.assertEqual(run['diag']['log_p0'], -800.0)  # log 口径保留
        self.assertFalse(run['diag']['direct_path_available'])
        # 不重归一：已算质量极小，尾质量接近 1，且无 NaN
        self.assertTrue(all(math.isfinite(x) for x in run['p']))
        self.assertGreater(run['diag']['tail_mass_beyond_K'], 0.999)


class CrossCheckTests(unittest.TestCase):
    def test_truncated_convolution_matches_panjer(self):
        cfg = parse_inputs(base_input())
        events = event_payments(cfg)
        lo, _ = severity_envelope(events, cfg['h'], cfg['K'])
        run = run_panjer(2.0, lo, cfg['K'], cfg['tol'])
        cc = cross_check(2.0, run['f'], cfg['K'], run['p'],
                         30, cfg['tol'])
        self.assertTrue(cc['passes_within_remainder_plus_tol'])
        self.assertLess(cc['max_abs_diff_vs_panjer'], 1e-12)
        # lambda=2、30 项时 Poisson 余尾在机器精度内为 0
        self.assertEqual(cc['poisson_remainder_bound_P_N_gt_m'], 0.0)

    def test_poisson_remainder_bound_monotone_and_finite(self):
        b1 = poisson_remainder_bound(2.0, 5)
        b2 = poisson_remainder_bound(2.0, 30)
        self.assertGreater(b1, b2)
        self.assertEqual(poisson_remainder_bound(0.0, 0), 0.0)
        # 大 lambda、少项数时余尾界接近 1，仍然是有限数
        self.assertAlmostEqual(poisson_remainder_bound(1000.0, 30), 1.0)

    def test_cross_check_remainder_explains_difference(self):
        # 故意只截 2 项：差异应被余尾界包住
        cfg = parse_inputs(base_input())
        events = event_payments(cfg)
        lo, _ = severity_envelope(events, cfg['h'], 30)
        run = run_panjer(2.0, lo, 30, cfg['tol'])
        cc = cross_check(2.0, run['f'], 30, run['p'], 2, cfg['tol'])
        self.assertLessEqual(
            cc['max_abs_diff_vs_panjer'],
            cc['poisson_remainder_bound_P_N_gt_m'] + cfg['tol'])


class VaRTVaRTests(unittest.TestCase):
    def test_split_atom_tvar_formula(self):
        # 手工原子：P(Y=0)=0.5, P(Y=10)=0.3, P(Y=20)=0.2；alpha=0.7
        atoms = [(Fraction(0), 0.5), (Fraction(10), 0.3),
                 (Fraction(20), 0.2)]
        st = var_tvar(atoms, 0.7)
        # VaR=10（F(10)=0.8 首次达到 0.7）
        self.assertEqual(st['var'], 10)
        # TVaR=(10*(0.8-0.7)+20*0.2)/0.3=(1+4)/.3=16.666...
        self.assertAlmostEqual(st['tvar'], 50 / 3, places=12)
        # 等价 stop-loss 形式 q + E[(Y-q)+]/(1-alpha)
        self.assertAlmostEqual(st['tvar'],
                               10 + (0.2 * 10) / 0.3, places=12)

    def test_tvar_does_not_count_full_q_atom(self):
        # 若错误地把整个 q 原子算进上尾会得到 (10*0.3+20*0.2)/.3=23.33
        st = var_tvar([(Fraction(0), 0.5), (Fraction(10), 0.3),
                       (Fraction(20), 0.2)], 0.7)
        self.assertLess(st['tvar'], 20.0)

    def test_tvar_at_cap_respects_atom_formula(self):
        # q 就是年度上限时同样遵守原子公式，不得直接返回 L 而忽略分摊
        atoms = [(Fraction(0), 0.2), (Fraction(40), 0.8)]
        st = var_tvar(atoms, 0.95)
        self.assertEqual(st['var'], 40)
        # (40*(1.0-0.95))/0.05 = 40
        self.assertAlmostEqual(st['tvar'], 40.0)
        atoms2 = [(Fraction(0), 0.2), (Fraction(40), 0.8)]
        st2 = var_tvar(atoms2, 0.5)
        # (40*(1.0-0.5))/0.5=40
        self.assertAlmostEqual(st2['tvar'], 40.0)

    def test_brute_force_annual_distribution_inside_envelope(self):
        v = base_input(max_grid=60)
        brute, omitted = brute_force_annual(v, n_max=16)
        result = annual_aggregate(v)
        # 暴力枚举的期望年度赔付（忽略极小余尾）应落在包络均值区间内
        brute_mean = sum(float(y) * float(p) for y, p in brute.items())
        lo_mean = min(i['mean_interval'][0]
                      for i in result['indicators'])
        hi_mean = max(i['mean_interval'][1]
                      for i in result['indicators'])
        # lambda=2、截断16项的 Poisson 余尾约 1e-11
        self.assertLess(omitted, 1e-9)
        self.assertLessEqual(brute_mean - omitted * 40 - 1e-9, hi_mean)
        self.assertGreaterEqual(brute_mean + 1e-9, lo_mean)
        # 暴力 VaR（左分位）应落在报告的 VaR 区间内
        for alpha in (0.9, 0.95, 0.99):
            cum = 0.0
            q = None
            for y in sorted(brute):
                cum += float(brute[y])
                if cum >= alpha - 1e-12:
                    q = y
                    break
            band = next(i for i in result['indicators']
                        if i['alpha'] == alpha)
            self.assertGreaterEqual(q, Fraction(band['var_interval'][0]))
            self.assertLessEqual(q, Fraction(band['var_interval'][1]))


    def test_formal_bounds_hold_against_independent_convolution(self):
        # 独立卷积（非 Panjer 路径）算出的真实指标必须落在正式区间内
        v = base_input(max_grid=40)
        brute, omitted = brute_force_annual(v, 20)
        self.assertLess(omitted, 1e-12)
        items = sorted((float(y), float(p)) for y, p in brute.items())
        result = annual_aggregate(v)
        for ind in result['indicators']:
            alpha = ind['alpha']
            cum = 0.0
            for y, p in items:
                cum += p
                if cum >= alpha - 1e-12:
                    q = y
                    break
            exc = sum(p * (y - q) for y, p in items if y > q)
            tvar = q + exc / (1 - alpha)
            mean = sum(y * p for y, p in items)
            self.assertGreaterEqual(
                q, float(Fraction(ind['var_interval'][0])))
            self.assertLessEqual(
                q, float(Fraction(ind['var_interval'][1])))
            self.assertGreaterEqual(tvar, ind['tvar_interval'][0] - 1e-9)
            self.assertLessEqual(tvar, ind['tvar_interval'][1] + 1e-9)
            self.assertGreaterEqual(mean, ind['mean_interval'][0] - 1e-9)
            self.assertLessEqual(mean, ind['mean_interval'][1] + 1e-9)

    def test_interior_quantile_bracketed_by_grid_envelope(self):
        # 分位落在非上限内部原子时，非整格真实值被上下格夹住
        v = base_input(annual_rate=0.5,
                       annual_contract={'deductible': '5', 'limit': '200'},
                       max_grid=80, alpha=[0.8])
        brute, _ = brute_force_annual(v, 15)
        items = sorted((float(y), float(p)) for y, p in brute.items())
        cum = 0.0
        for y, p in items:
            cum += p
            if cum >= 0.8 - 1e-12:
                q = y
                break
        result = annual_aggregate(v)
        band = result['indicators'][0]
        self.assertEqual(q, 5.5)  # 真实非整格分位
        self.assertEqual(band['var_interval'], ['5', '6'])


class FullMaterialTests(unittest.TestCase):
    def test_all_valid_materials_run(self):
        for name in ('normal', 'zero-arrival-rate',
                     'underflow-arrival-rate',
                     'substantial-uncomputed-tail'):
            with self.subTest(file=name):
                v, _ = load(f'data/{name}.json')
                r = annual_aggregate(v)
                # 质量账闭合
                for env in ('lower', 'upper'):
                    rep = r['panjer'][env]
                    self.assertAlmostEqual(
                        rep['computed_mass_0_to_K']
                        + rep['tail_mass_beyond_K'], 1.0, places=10)
                # 指标不超过年度上限
                L = float(Fraction(r['parameters']['annual_limit']))
                for ind in r['indicators']:
                    self.assertLessEqual(
                        float(Fraction(ind['var_interval'][1])), L)
                    self.assertLessEqual(ind['tvar_interval'][1],
                                         L + 1e-6)
                    self.assertGreaterEqual(
                        float(Fraction(ind['var_interval'][0])), 0.0)

    def test_substantial_tail_is_undetermined(self):
        v, _ = load('data/substantial-uncomputed-tail.json')
        r = annual_aggregate(v)
        for ind in r['indicators']:
            self.assertFalse(ind['var_determined'])
            self.assertFalse(ind['mean_determined'])
            self.assertEqual(ind['var_interval'], ['0', '40'])
            self.assertTrue(any('max_grid' in a for a in ind['advice']))

    def test_underflow_diagnostic_reports_stable_path(self):
        v, _ = load('data/underflow-arrival-rate.json')
        r = annual_aggregate(v)
        diag = r['diagnostics']['p0_underflow']
        self.assertTrue(diag['lower_envelope'])
        self.assertTrue(diag['upper_envelope'])
        # 尾域在 K=200 时已映射到上限（退化单点），指标仍可夹住
        for ind in r['indicators']:
            self.assertTrue(ind['tail_degenerate'])

    def test_zero_lambda_indicators_zero(self):
        v, _ = load('data/zero-arrival-rate.json')
        r = annual_aggregate(v)
        for ind in r['indicators']:
            self.assertEqual(ind['var_interval'], ['0', '0'])
            self.assertLess(ind['mean_interval'][1], 1e-6)

    def test_diagnostics_report_zero_cap_offgrid(self):
        v, _ = load('data/normal.json')
        r = annual_aggregate(v)
        d = r['diagnostics']
        self.assertEqual(d['zero_payment_scenarios'], ['none'])
        self.assertEqual(d['zero_event_probability_exact'], '1/5')
        self.assertEqual(d['non_integer_step_amounts'][0]['id'], 'river')
        self.assertEqual(
            d['non_integer_step_amounts'][0]['ceil_grid'], '11')
        self.assertEqual(d['cap_atom']['annual_limit'], '40')
        # 已计算格点内的上限原子质量；其余上限概率在映射到 cap 的尾质量中
        self.assertAlmostEqual(
            d['cap_atom']['computed_cap_mass_lower'], 0.2109, places=3)
        self.assertTrue(d['cap_atom']['tail_also_maps_to_cap'])

    def test_invalid_inputs_rejected(self):
        bad_cases = [
            {'annual_rate': -1},
            {'alpha': [1.0]},
            {'alpha': [0.0]},
            {'step': '0'},
            {'max_grid': -1},
            {'numeric_tolerance': 0},
        ]
        for kw in bad_cases:
            with self.subTest(kw=kw):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    annual_aggregate(base_input(**kw))


if __name__ == '__main__':
    unittest.main()
