import sys, os, hashlib, glob
paths = [
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\distribution_context_decoder_20261004_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\local_candidates_20261003_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\local_candidates_20261004_v2\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\p2_candidate_screen24_20261007_v1\native_reference_B\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\p2_candidate_seed1_full132_20261007_v1\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\p2_seed1_replay6_20261008_v1\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\query_context_gap_20261004_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\range_context_decoder_20261005_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\range_context_decoder_20261005_v3\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\range_context_experiments_20261005_v1\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\range_context_experiments_20261005_v2\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\range_context_experiments_20261005_v4\native\engine\deim\deim.py",
 r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\scap_ext24_acceptance_20261007_v1\engine\deim\deim.py",
]
target="e2e05310499c33e1adfc2189c82fe25dc81118f61bdd9627838723a63c1f52bc"
for p in paths:
    if not os.path.exists(p): print("MISSING", p); continue
    d=open(p,'rb').read().replace(b'\r\n',b'\n')
    h=hashlib.sha256(d).hexdigest()
    print(("*** MATCH *** " if h==target else ""), h[:16], p.replace(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\\",""))
