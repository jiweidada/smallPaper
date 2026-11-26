"""Record a source-access correction and unselected question without reopening A56."""
from pathlib import Path
import hashlib
import json

P=Path(__file__).resolve().parent
ROOT=P.parent.parent.parent


def load(p):return json.loads(p.read_text('utf-8-sig'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)


def main():
    terminal=load(P/'A56_terminal_preservation_receipt.json')
    assert all(sha(Path(p))==r['sha256'] for p,r in terminal['files'].items())
    source=load(P/'post_terminal_source_screen/MambaSOD_source_receipt.json')
    assert source['source_imported_or_executed'] is False and len(source['files'])==9
    for r in source['files']:
        assert sha(P/'post_terminal_source_screen/MambaSOD_pinned'/r['path'])==r['sha256']
    report=P/'15_下一候选初筛与源码纠错.txt'
    status={'status':'SOURCE_ACCESS_CORRECTED_NEW_QUESTION_UNSELECTED', 'goal_status':'active',
            'mainline_validated':False,'algorithm_increment_selected':False,'candidate_selected_for_pilot':False,
            'training_allocation':0,'new_detector_forwards':0,'new_optimizer_updates':0,
            'A56_decision':'RETIRED_EXACT_RECIPE_FIXED12','A56_reopened':False,
            'next_question':'Does coupling context-grid length to query count restrict fine-scale evidence in Mamba query-feature interaction?',
            'question_is_hypothesis_not_measured_failure':True,
            'draft_only':'Decouple context-state advance from geometric query readout; exclude ordinary sampling/attention and interleaved-query prior before allocating a pilot.',
            'strong_controls_not_yet_frozen':True,'MambaSOD_fulltext_read':False,
            'MambaSOD_source_commit':source['commit'],'native_MambaSOD_reproducibility_verified':False,
            'next_required':'Bounded nearest-operator and cost check; do not train by merely changing host or repackaging A56.',
            'report':str(report),'source_receipt':str(P/'post_terminal_source_screen/MambaSOD_source_receipt.json')}
    save(P/'post_terminal_source_screen/screen_status.json',status)
    save(P.parent/'POST_A56_SCREEN_CURRENT.json',status)
    supplement=ROOT/'文献总库_20261101_v1/03_索引与核对/后续补链_MambaSOD源码访问纠错.json'
    save(supplement,{'title':'MambaSOD: Small Object Detection from Drone Perspective in a Mamba-based Query-Feature Interaction Framework',
                    'doi':'10.1142/S2301385028500343','source_receipt':status['source_receipt'],
                    'repo':source['repo'],'commit':source['commit'],'new_fulltext_PDFs':0,
                    'paper_read_scope':'fulltext unavailable; author bibliography and pinned selected code inspection only',
                    'historical_reference':'A17 source URL omitted hyphen in small-object; old404 is access history, not a method result',
                    'base_catalogue_records_unchanged':423,'report':str(report)})
    paths=[report,P/'audit_new_MambaSOD_source.py',Path(__file__),supplement,
           P.parent/'POST_A56_SCREEN_CURRENT.json',P/'post_terminal_source_screen/screen_status.json',
           P/'post_terminal_source_screen/MambaSOD_source_receipt.json',
           P/'post_terminal_source_screen/MambaSOD_git_tree.json',P/'post_terminal_source_screen/web_records_v1.json']
    paths += [P/'post_terminal_source_screen/MambaSOD_pinned'/r['path'] for r in source['files']]
    save(P/'post_terminal_source_screen/preservation_receipt.json',
         {'status':'SOURCE_SCREEN_PRESERVED','goal_status':'active','A56_terminal_receipt_unchanged_and_verified':True,
          'files':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths},
          'new_model_forwards':0,'new_optimizer_updates':0})
    print(json.dumps({'status':status['status'],'files_hashed':len(paths),
                      'new_training_jobs':0,'mainline_validated':False}))


if __name__=='__main__':main()
