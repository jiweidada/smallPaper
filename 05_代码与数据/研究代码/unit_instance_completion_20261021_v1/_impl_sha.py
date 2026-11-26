import hashlib, os
d=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
exp={"geometry.py":"f537d8ad50a54fd1b063db5e86d6c1cf4534304588db3c5008cfe69fac1100c9",
"residual.py":"3f60403011f8e999c77dcc4a31aeb9974149fda9dbc7aa9216eca486704f30a2",
"runtime.py":"624d68b280314bd84eb72737eb3752c078f07c940c184b2d9d6ce72a6f369064",
"experiment.py":"27c4a578387f2a5a6791b040deed298c62ea0f2b73b2164b71b146bf324f3228",
"results.py":"963adcee81ef5b030446621d8778771e7609f74f32ac012197daf455f81200b4",
"stage_trace.py":"de34db3e48f6645594f1ecb5b71487bc321fd2ed6988dd73b2aa394901fb3fa9",
"evaluator_binding.py":"9d9b5ca539b3ae21cca9173da9a7e0711a8d01693abcce43db65a4a6798d1625",
"evaluator_sources.json":"fd37b26487848369be53c02b80fe3f58c9b67b90d973d64b7d209965ccc968e0",
"training_audit.py":"017aeca8e08321063260c345898e6ad649c43a0a8d61bf2fbaf72e5156406ccb"}
for n,e in exp.items():
    p=os.path.join(d,n)
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    print(("OK  " if h==e else "DIFF"), n, h[:16], "exp", e[:16])
