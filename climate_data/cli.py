"""资料摘要与年度聚合准备金指标的命令行入口，来源文件保持不动。"""
import argparse
import json
from .io import load
from .domain import summarize
from .aggregate import annual_aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect', 'reserve'])
    parser.add_argument('file')
    a = parser.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        payload = {'source_sha256': digest, 'summary': summarize(v)}
    else:
        payload = {'source_sha256': digest,
                   'annual_aggregate': annual_aggregate(v)}
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
