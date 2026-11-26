"""Validate the UI-exported meeting; does not call models, servers, or experiments."""
import collections, datetime as dt, hashlib, json, pathlib

root = pathlib.Path(__file__).resolve().parent
path = root / '讨论原始导出_第6轮完成.json'
raw = path.read_bytes()
d = json.loads(raw)
assert d['id'] == '99accfb2-292b-458c-bf58-96641b4b5d34'
assert d['status'] == 'completed' and d['round'] == d['rounds'] == 6
assert d.get('error') is None
messages = d['messages']
assert len(messages) == 24
assert all(m.get('text', '').strip() for m in messages)
expected_experts = {x['id'] for x in d['experts']}
assert len(expected_experts) == 4
by_round = []
transcript = []
for rnd in range(1,7):
    rows = [m for m in messages if m['round'] == rnd]
    assert len(rows) == 4
    assert {m['expertId'] for m in rows} == expected_experts
    prior = {m['id'] for m in messages if m['round'] < rnd}
    checks = []
    transcript.append(f'第{rnd}轮｜4份完整回答\n')
    for expert in d['experts']:
        m = next(m for m in rows if m['expertId'] == expert['id'])
        seen = set(m.get('seenMessageIds', []))
        assert seen == prior, (rnd,m['expertName'],len(seen),len(prior))
        checks.append(dict(expert=m['expertName'], characters=len(m['text']),elapsed_ms=m.get('elapsedMs'),seen_message_count=len(seen),context_hash=m.get('contextHash'), requested=m.get('requestedSettings'), actual_model=m.get('model'),actual_provider=m.get('provider'),actual_reasoning=m.get('reasoningEffort')))
        transcript.append(f"--- {m['expertName']} ---\n{m['text']}\n")
    by_round.append(dict(round=rnd,answers=len(rows),checks=checks))
(root/'六轮完整发言_24份.txt').write_text('\n'.join(transcript),encoding='utf8')
(root/'本次追加三轮完整发言_12份.txt').write_text('\n'.join(transcript[15:]),encoding='utf8')
now=dt.datetime.now(dt.timezone.utc)
inputs=[]
for p in sorted(root.glob('第*轮共同续问.txt')):
    b=p.read_bytes();inputs.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
receipt=dict(meeting_id=d['id'],status=d['status'],rounds=6,total_complete_answers=len(messages),new_rounds=[4,5,6],new_complete_answers=12,export_sha256=hashlib.sha256(raw).hexdigest(),export_path=str(path),export_meeting_updated_at=d.get('updatedAt'),checked_at_utc=now.isoformat(),checked_at_America_Los_Angeles=now.astimezone(dt.timezone(dt.timedelta(hours=-7))).isoformat(),timezone_note='America/Los_Angeles is UTC-07 on 2026-10-19/20',same_round_answers_hidden=all(len(set(m.get('seenMessageIds',[])))==4*(m['round']-1) for m in messages),scope=dict(new_training=0,new_model_forward=0,remote_operations=0,monitor_management=0,manuscript_edits=0),inputs=inputs,round_details=by_round,model_provenance_limit='Actual served model/provider/reasoning fields are null in export. Seats/adapters and requested settings verified; do not claim four independent model families or verified served model.',save_state=d.get('saveState'),storage_error=d.get('storageError'))
(root/'本次继续讨论核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:receipt[k] for k in ['meeting_id','status','rounds','total_complete_answers','new_complete_answers','same_round_answers_hidden','save_state','storage_error']},ensure_ascii=False,indent=2))
