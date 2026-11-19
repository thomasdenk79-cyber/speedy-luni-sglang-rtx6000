#!/usr/bin/env python3
"""Real agent-load numbers from Prometheus (sglang:* metrics of one engine).

Usage: real_load_stats.py [HOURS] [PROMETHEUS_URL]
Prints decode tok/s by running requests (median/p90/max), TTFT/ITL/E2E quantiles and
request/token counters. Source of the "Real" tables in README and docs/releases/speedy-rc2.md.
"""
import json,urllib.request,urllib.parse,statistics as st,time,sys
P=(sys.argv[2] if len(sys.argv)>2 else 'http://127.0.0.1:9090').rstrip('/')+'/api/v1/'
def q(path,**a):
    return json.load(urllib.request.urlopen(P+path+'?'+urllib.parse.urlencode(a),timeout=60))['data']['result']
end=time.time(); hours=float(sys.argv[1]) if len(sys.argv)>1 else 30
start=end-hours*3600; step=15
def rng(expr):
    out={}
    t=start
    while t<end:
        e=min(end,t+10000*step)
        for r in q('query_range',query=expr,start=t,end=e,step=step):
            for ts,v in r['values']: out[float(ts)]=float(v)
        t=e
    return out
g=rng('max(sglang:gen_throughput{instance=~".*30000.*"})') or rng('max(sglang:gen_throughput)')
n=rng('max(sglang:num_running_reqs)')
acc=rng('max(sglang:spec_accept_length)')
by={}
for ts,v in g.items():
    k=n.get(ts)
    if k is None or v<=0: continue
    by.setdefault(int(k),[]).append(v)
print(f"window {hours}h samples {len(g)} step {step}s")
print("run | samples | share | median | p90 | max | per-stream median")
tot=sum(len(v) for k,v in by.items() if k>0)
for k in sorted(by):
    if k==0: continue
    v=sorted(by[k]); p=lambda x:v[min(len(v)-1,int(x*len(v)))]
    print(f"{k} | {len(v)} | {len(v)/tot*100:.1f}% | {st.median(v):.0f} | {p(.9):.0f} | {v[-1]:.0f} | {st.median(v)/k:.0f}")
busy=[v for k,l in by.items() if k>0 for v in l]
print("all busy: median",round(st.median(busy)),"max",round(max(busy)))
a=[v for v in acc.values() if v>0]; print("accept len median",round(st.median(a),2) if a else None)
for name,m in [('TTFT','time_to_first_token_seconds'),('ITL','inter_token_latency_seconds'),('E2E','e2e_request_latency_seconds')]:
    for qq in (0.5,0.95,0.99):
        r=q('query',query=f'histogram_quantile({qq}, sum by (le)(increase(sglang:{m}_bucket[{int(hours*60)}m])))',time=end)
        print(name,qq,round(float(r[0]['value'][1]),3) if r else None)
for m in ['sglang:num_requests_total','sglang:prompt_tokens_total','sglang:generation_tokens_total','sglang:cached_tokens_total']:
    r=q('query',query=f'sum(increase({m}[{int(hours*60)}m]))',time=end); print(m, round(float(r[0]['value'][1])) if r else None)
