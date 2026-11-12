# 联合损失与年度合同

同一场景的多地区losses同时发生，金额为精确非负有理数。每地区赔付min(max(loss-deductible,0),limit)，单次赔付为地区赔付之和。年度合同对未扣年度免赔前的单次赔付总和S作用为Y=min(max(S-annual_deductible,0),annual_limit)。annual_rate为有限非负实数，表示所有事件（含零额）到达率。

step=h为正有理数，每次严重度floor(x/h)与ceil(x/h)分别构造下/上格分布。f0可正。Panjer递推的p0=exp(lambda*(f0-1))，其余pn=lambda/n*sum_{j=1..n}(j*f_j*p_{n-j})；用到该表达式不代表下溢时可直接丢掉概率。max_grid限制S的格点，不是Y，年度合同在该总额分布之后应用。截断尾质量在缺少更多分布信息时位于S>(K*h)，将此域通过年度单调合同映射得到有限付款界，不重归一。

alpha严格在0和1之间，VaR为左分位q。TVaR=（E[Y*1(Y>q)]+q*(F(q)-alpha)）/(1-alpha)，上尾在q原子上只取必要部分；即使q为年度上限，也须遵守原子公式。上下格点包络夹住实际Y，均值、VaR和TVaR都可以给随机序界并额外纳入未算尾质量，不声称h趋零之前精确。

## 年度聚合实现口径（climate_data.aggregate）

`reserve`命令在`inspect`的原始摘要之外，另输出逐层可追溯的年度聚合结果，资料文件不被修改。

1. 单场景赔付：场景内对每地区应用`min(max(loss-deductible,0),limit)`，地区赔付之和为该场景单次赔付X；同场景地区损失同时发生，不假设地区独立。金额全程为Fraction。
2. 严重度包络：按场景概率聚合X后，下包络取`floor(X/h)`、上包络取`ceil(X/h)`（X=0两包络均落0格）。f0是零额严重度质量，允许为正；超过K格的严重度质量单列`severity_tail`，不丢弃。
3. 复合Poisson：log-domain Panjer递推计算0..K格，`p0=exp(lambda*(f0-1))`，绝不以`exp(-lambda)`替换。lambda=0退化为p0=1、其余结构零。大lambda下p0浮点下溢时以`log_p0`保留并标记`p0_underflow`，直接递推路径仅在p0可表示时用于数值核对。
4. 质量账：`computed_mass_0_to_K + tail_mass_beyond_K = 1`（容差内），尾质量含未算格点与舍入残差，不重归一；另报`roundoff_residual`与被冲零的格数。
5. 年度合同在总额S分布之后应用：格点金额先过`g(s)=min(max(s-D,0),L)`并合并同值原子，两层顺序不可调换。尾质量S>Kh在单调合同下的可行赔付为`[g(Kh), L]`；年度上限有限，尾界总是有限。退化情形`g(Kh)=L`时尾域为单点，不构成指标不确定性。
6. 指标：VaR为累计概率首次达到alpha的最小赔付；TVaR按q原子必要份额分摊（等价`q+E[(Y-q)+]/(1-alpha)`）。对每条包络把尾质量分别放到尾域两端构造极值分布，再合并上下包络给均值/VaR/TVaR正式区间。尾质量足以跨过分位或改变均值时返回未确定区间，advice指出应增大K（尾不足）还是减小h（格点展宽）。
7. 独立核对：小lambda另用事件数截断卷积级数（`poisson_check_terms`项）与Panjer逐格比对，差异以Poisson余尾界`P(N>m)=1-sum pmf`解释，要求落在余尾界加容差内；大lambda余尾界接近1时如实报告而不声称核对通过精度。
8. 诊断覆盖：零额事件与精确零额概率、非整步金额的floor/ceil格、年度上限原子质量（含尾是否也映到上限）、p0下溢与稳定路径。

domain只读联合场景原损失均值，不应用合同、不做年度卷积或尾指标。
