from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'A49_下一主线有界筛选_20261101_v1'/'supervise_A49.py'
text=old.read_text(encoding='utf-8')
text=text.replace("'stages': ['B12', 'M12', 'Q12']", "'stages': ['M12', 'Q12'], 'B12_reused': '/root/uav_a49_20261101_v1/arm_B'")
text=text.replace("for arm in ['B', 'M', 'Q']:","for arm in ['M', 'Q']:")
text=text.replace("'train_A49.py'","'train_A51.py'").replace('20 * 3600','6 * 3600').replace('Own A49','Own A51')
text=text.replace("rows = {a: json.loads((ROOT / ('arm_' + a) / 'result.json').read_text()) for a in ['B', 'M', 'Q']}",
 "rows = {a: json.loads((ROOT / ('arm_' + a) / 'result.json').read_text()) for a in ['M', 'Q']}\n        rows['B'] = json.loads(Path('/root/uav_a49_20261101_v1/arm_B/result.json').read_text())")
text=text.replace("flows = {a: json.loads((ROOT / ('arm_' + a) / ('epoch%02d_input_flow.json' % epoch)).read_text()) for a in rows}",
 "flows = {a: json.loads((ROOT / ('arm_' + a) / ('epoch%02d_input_flow.json' % epoch)).read_text()) for a in ['M','Q']}\n            flows['B'] = json.loads((Path('/root/uav_a49_20261101_v1/arm_B') / ('epoch%02d_input_flow.json' % epoch)).read_text())")
text=text.replace("'formal_model_forwards': sum(r['full_model_forwards'] for r in rows.values()),", "'new_formal_model_forwards': sum(rows[a]['full_model_forwards'] for a in ['M','Q']),\n            'B_reused_model_forwards_not_new': rows['B']['full_model_forwards'],")
text=text.replace("'formal_optimizer_updates': sum(r['optimizer_updates'] for r in rows.values()),", "'new_formal_optimizer_updates': sum(rows[a]['optimizer_updates'] for a in ['M','Q']),\n            'B_reused_optimizer_updates_not_new': rows['B']['optimizer_updates'],")
dest=ROOT/'supervise_A51.py';assert not dest.exists();dest.write_text(text,encoding='utf-8')
bridge=ROOT.parent/'A50_编码解码新机制有界筛选_20261101_v1'/'remote_A50.py'
dest=ROOT/'remote_A51.py';assert not dest.exists()
dest.write_text(bridge.read_text(encoding='utf-8').replace('/root/uav_a50_20261101_v1','/root/uav_a51_20261101_v1'),encoding='utf-8')
print('Supervisor prepared: reused B12, only new M12/Q12, no retry or downstream queue.')
