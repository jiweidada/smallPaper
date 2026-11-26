"""Pure source instrumentation, independently parseable without torch."""
import ast


def instrument_source(source):
    anchor='C = C.view(bs, num_queries, -1).cpu()'
    lines=source.splitlines(keepends=True)
    selected=[(i,l) for i,l in enumerate(lines) if anchor in l]
    assert len(selected)==1
    index,line=selected[0]
    indent=line[:len(line)-len(line.lstrip())]
    insertion=(indent+'if getattr(self, "_support_matching_penalty", None) is not None:\n'
        +indent+'    assert C.shape == self._support_matching_penalty.shape\n'
        +indent+'    C = C + self.cost_giou * self._support_matching_penalty\n')
    result=''.join(lines[:index])+insertion+''.join(lines[index:])
    original, changed=ast.parse(source),ast.parse(result)
    original_fn=next(n for n in original.body if isinstance(n,ast.FunctionDef))
    changed_fn=next(n for n in changed.body if isinstance(n,ast.FunctionDef))
    added=[n for n in changed_fn.body if isinstance(n,ast.If) and '_support_matching_penalty' in ast.unparse(n.test)]
    assert len(added)==1 and len(added[0].body)==2 and not added[0].orelse
    changed_fn.body.remove(added[0])
    assert ast.dump(original,include_attributes=False)==ast.dump(changed,include_attributes=False)
    compile(result,'<preflight-instrumented-matcher>','exec')
    return result
