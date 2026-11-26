"""Pure integer stopping rules. No file, data, model, or network access."""

def _rule(per_seed, numerator, denominator, multiplier):
    values={}
    for seed in (1,2):
        n,d=per_seed[seed][numerator],per_seed[seed][denominator]
        if type(n) is not int or type(d) is not int or not 0<=n<=d:
            raise ValueError('Counts must be integers with 0 <= numerator <= denominator')
        values[seed]={'numerator':n,'denominator':d,
                      'ratio':n/d if d else None,
                      'below':multiplier*n<d if d else None}
    flags=[v['below'] for v in values.values()]
    state=('NA' if None in flags else 'both_below' if all(flags)
           else 'mixed' if any(flags) else 'neither_below')
    return {'per_seed':values,'state':state,'stop':None if state=='NA' else state=='both_below'}

def decide(per_seed):
    """M/T is omitted GT rate; D/B is Ustar-vs-anchor assigned-GT difference."""
    if set(per_seed)!={1,2}:raise ValueError('Exactly the two preregistered seeds are required')
    rules={'missing':_rule(per_seed,'M','T',200),'different':_rule(per_seed,'D','B',100)}
    reasons=[name for name,r in rules.items() if r['stop'] is True]
    status=('stop_current_candidate' if reasons else
            'insufficient_applicable_assets' if any(r['state']=='NA' for r in rules.values()) else
            'mixed_seed_results' if any(r['state']=='mixed' for r in rules.values()) else
            'not_rejected_by_these_two_thresholds')
    return {'rules':rules,'stop':bool(reasons),'stop_reasons':reasons,'status':status,
            'automatic_progression':False,'other_prefrozen_checks_still_required':True}
