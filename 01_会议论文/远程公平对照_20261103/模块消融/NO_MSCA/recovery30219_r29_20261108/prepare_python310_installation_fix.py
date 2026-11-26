"""Keep the failed Python3.8 preparation attempt and continue its still-empty tree with Python3.10."""
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
original = here / 'install_recovery.py'
text = original.read_text(encoding='utf-8')
old = 'assert not target.exists()\n'
assert text.count(old) == 1
text = text.replace(old, "assert target.is_dir() and not any(target.iterdir()), 'Partial target is not the expected empty directory'\n")
assert text.count("(prep / 'installation.lock').open('x')") == 1
text = text.replace("(prep / 'installation.lock').open('x')", "(prep / 'installation_python310.lock').open('x')")
assert text.count('target.mkdir(exist_ok=False)\n') == 1
text = text.replace('target.mkdir(exist_ok=False)\n', "assert (prep / 'installation.lock').is_file(), 'Missing original failed-preparation marker'\n")
path = here / 'install_recovery_python310.py'
with path.open('x', encoding='utf-8') as stream: stream.write(text)
report = {'original_installer_sha256': sha256(original.read_bytes()).hexdigest(),
    'fixed_preparation_sha256': sha256(path.read_bytes()).hexdigest(),
    'failure': 'System /usr/bin/python3 lacks pathlib.Path.is_relative_to; failed before the first archive member was written.',
    'resolution': 'Use frozen Python3.10.18; resume only if the newly created target tree is empty; retain original marker and failure receipt.',
    'training_was_started': False}
(here / 'installation_python38_failure_and_fix.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
