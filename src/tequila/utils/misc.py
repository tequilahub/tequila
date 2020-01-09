def has_variable(object,var):
    tv= type(var)
    if tv is dict:
        back={}
        for k in var.keys():
            back[k]=has_variable(object,k)
    elif tv is str:
        v=var
        back=inner_has_var(object,v)
    elif hasattr(var,'name') and hasattr(var,'value'):
        v=var.name
        back=inner_has_var(object,v)
    else:
        back=None
    return back

def inner_has_var(object,v):
    if hasattr(object,'args') or hasattr(object,'name'):
        for key in object.extract_variables().keys():
            if key == v:
                return True
    if hasattr(object,'gates'):
        if any([inner_has_var(g,v)==True for g in object.gates]):
            return True
    if hasattr(object,'parameter'):
        return inner_has_var(object.parameter,v)

    return False