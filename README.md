# 灾损合约年度聚合的离散分布与尾部界

这是独立的气候资料Python项目。`data/`保存合成资料，`climate_data.io`负责严格读取、公历日期和来源字节摘要，`climate_data.domain`提供现有描述统计，`climate_data.cli`将摘要输出为JSON。所有观测值、站点、人群、情景与金额都是虚构。字段和研究约定见[资料约定](docs/data-contract.md)。

当前Python 3.11及以上可直接使用，没有第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q climate_data tests
python -m climate_data.cli inspect data/normal.json
python -m climate_data.cli reserve data/normal.json
```

`inspect`显示当前原始摘要，不覆盖资料。`reserve`在同一资料身份（SHA-256）下输出年度聚合赔付分布：单场景地区赔付、下/上严重度格点包络（含零额质量f0与严重度尾质量）、log-domain Panjer递推的p0与质量残差、事件数截断卷积的独立核对，以及均值/VaR/TVaR的正式区间与未确定提示。尾质量不重归一；分位被尾质量改变时给出未确定区间并指出应增大`max_grid`或减小`step`。口径细节见[资料约定](docs/data-contract.md)。读取不可信资料目录时建议用`python -I`并以绝对路径传参。`invalid-`前缀文件为不合法资料，其他变体使用相同字段保留业务边界。来源字节SHA-256随摘要输出，后续结果应同时标明输入身份、量纲和适用范围。

容器使用固定基础镜像，可运行`docker build -t climate-material .`，随后`docker run --rm climate-material`检查资料。`docker run --rm climate-material python -m unittest discover -s tests -v`执行已有检查。运行不依赖远端账号或实时服务。
