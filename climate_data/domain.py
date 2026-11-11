"""读取联合灾损场景，展示未应用合同的均值。"""
from .io import unique,fraction,number,integer

def summarize(v):
    unique(v['scenarios'],'id');regions=set(v['regional_contract']);prob=0;mean=0
    if number(v['annual_rate'])<0:raise ValueError('事件率不能负')
    for s in v['scenarios']:
        p=fraction(s['probability']);prob+=p
        if p<0 or set(s['losses'])!=regions:raise ValueError('联合场景概率或地区不完整')
        losses=[fraction(x) for x in s['losses'].values()]
        if any(x<0 for x in losses):raise ValueError('损失不能负')
        mean+=p*sum(losses)
    if prob!=1:raise ValueError('概率和不为1')
    for c in list(v['regional_contract'].values())+[v['annual_contract']]:
        if fraction(c['deductible'])<0 or fraction(c['limit'])<0:raise ValueError('合同金额无效')
    if fraction(v['step'])<=0:raise ValueError('格点步长应正')
    integer(v['max_grid'])
    return {'scenarios':len(v['scenarios']),'regions':sorted(regions),'raw_event_loss_mean':str(mean),'annual_rate':v['annual_rate']}
