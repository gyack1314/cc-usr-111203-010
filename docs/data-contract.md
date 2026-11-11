# 联合损失与年度合同

同一场景的多地区losses同时发生，金额为精确非负有理数。每地区赔付min(max(loss-deductible,0),limit)，单次赔付为地区赔付之和。年度合同对未扣年度免赔前的单次赔付总和S作用为Y=min(max(S-annual_deductible,0),annual_limit)。annual_rate为有限非负实数，表示所有事件（含零额）到达率。

step=h为正有理数，每次严重度floor(x/h)与ceil(x/h)分别构造下/上格分布。f0可正。Panjer递推的p0=exp(lambda*(f0-1))，其余pn=lambda/n*sum_{j=1..n}(j*f_j*p_{n-j})；用到该表达式不代表下溢时可直接丢掉概率。max_grid限制S的格点，不是Y，年度合同在该总额分布之后应用。截断尾质量在缺少更多分布信息时位于S>(K*h)，将此域通过年度单调合同映射得到有限付款界，不重归一。

alpha严格在0和1之间，VaR为左分位q。TVaR=（E[Y*1(Y>q)]+q*(F(q)-alpha)）/(1-alpha)，上尾在q原子上只取必要部分；即使q为年度上限，也须遵守原子公式。上下格点包络夹住实际Y，均值、VaR和TVaR都可以给随机序界并额外纳入未算尾质量，不声称h趋零之前精确。

domain只读联合场景原损失均值，不应用合同、不做年度卷积或尾指标。
