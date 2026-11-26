from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path(__file__).resolve().parents[3]
out = root / 'outputs/model_structure_html_20261010_v1'
artifact = out / 'deim-dfine-s-full-network.html'
delivery_path = out / 'delivery.json'
state_path = root / 'outputs/sod_roi12_20261009/execution_state.json'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

browser = read(out / 'browser_verification.json')
assert browser['templateEntryRedirect'] and not browser['errors']
assert browser['train']['nodes'] == 1719 and browser['eval']['nodes'] == 1326
delivery = read(delivery_path)
old_sha = delivery['artifact_sha256']
current_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
now = datetime.now(timezone.utc).isoformat()
receipt = {
    'status': 'TEMPLATE_ENTRY_REDIRECT_FIXED_AND_BROWSER_VERIFIED',
    'observed_utc': now,
    'cause': 'User screenshot shows network-template.html opened directly; its JSON placeholder has no embedded graph data.',
    'fix': 'Detect unrendered placeholder before JSON.parse and redirect to complete offline HTML, with a visible fallback link.',
    'previous_artifact_sha256': old_sha,
    'artifact_sha256': current_sha,
    'browser_verification': browser,
    'model_computation_changed': False,
    'model_forward_calls_added': 0,
    'training_epochs_added': 0,
    'optimizer_steps': 0
}
write(out / 'template_entry_fix.json', receipt)
delivery['artifact_sha256'] = current_sha
delivery['artifact_bytes'] = artifact.stat().st_size
delivery['browser_verification'] = browser
delivery['template_entry_fix'] = receipt
write(delivery_path, delivery)
state = read(state_path)
state['network_structure_html'] = delivery
state['latest_user_steering'] = '用户截图打开生成模板而一直显示载入；已修复模板入口自动跳转并核验完整图。'
state['research_iteration_history'].append({'utc': now, 'status': receipt['status'], 'evidence': str(out / 'template_entry_fix.json'), 'detector_epochs_added': 0, 'optimizer_steps': 0})
write(state_path, state)
with (out / '使用说明与核验.txt').open('a', encoding='utf-8') as stream:
    stream.write('\n入口修复：直接打开code目录中的network-template.html时，现在会自动转到outputs目录的完整HTML；浏览器跳转和完整图交互核验通过。已打开旧模板页面的用户按F5刷新即可。\n')
with state_path.with_name('continuation_state.txt').open('a', encoding='utf-8') as stream:
    stream.write(f'\n{now} HTML ENTRY FIX: User opened network-template.html; unrendered JSON placeholder caused persistent loading display. Added placeholder guard + relative redirect to complete artifact, fallback link. Browser template redirect and train1719/eval1326 interaction checks pass, noJS errors. Current HTML SHA{current_sha}; native graph data/computation unchanged, no model runs or training added.\n')
print(json.dumps({'status': receipt['status'], 'artifact_sha256': current_sha}))
