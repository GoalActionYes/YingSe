"""Close confirmed capture-family edges against all consumed holdout families."""


def heldout_alias_closure(edges,heldout):
    blocked=set(heldout)
    if any(not isinstance(x,str) or not x for x in blocked):raise ValueError('Invalid heldout identity')
    graph={}
    for edge in edges:
        if not isinstance(edge,(tuple,list)) or len(edge)!=2:raise ValueError('Expected alias pair')
        a,b=edge
        if any(not isinstance(x,str) or not x for x in edge) or a==b:raise ValueError('Invalid alias identities')
        graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    queue=list(blocked)
    while queue:
        for other in graph.get(queue.pop(),()):
            if other not in blocked:blocked.add(other);queue.append(other)
    return blocked
