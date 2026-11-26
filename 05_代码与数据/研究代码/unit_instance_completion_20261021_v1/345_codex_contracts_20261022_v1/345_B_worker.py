"""Private CPU miniature acceptance worker. One invocation is one fresh process."""
import importlib, json, sys, traceback, os
C = importlib.import_module('345_context')
import train345 as E

if __name__ == '__main__':
    job = C.read(sys.argv[1])
    try:
        result = E.execute(job['arm'], job['seed'], job['run_dir'], 'cpu', job['max_steps'],
                           _mode=job['mode'], _resume=job.get('resume'))
        print(json.dumps({'worker_pid': os.getpid(), 'arm': job['arm'], 'seed': job['seed'],
                          'step': result['step'], 'counts': result['counts_this_process']}, ensure_ascii=False), flush=True)
    except Exception as ex:
        C.write('345_B_worker_实现失败_停止.json', {'status': '实现失败，立即停止', 'job': job,
                'exception': repr(ex), 'traceback': traceback.format_exc(),
                'input_counts': dict(E.T.INPUT_COUNTS), 'real_training_steps': 0})
        raise
