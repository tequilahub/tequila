from dataclasses import dataclass


def parametrizedX(Cls, ptype):
    class NewCls(object):
        def __init__(self, parameter: ptype = 0, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)
            self.parameter = parameter


def parametrizedY(ptype):
    def decorator(Cls):
        def decorated(ptype):
            class NewCls:
                def __init__(self, p: ptype = None):
                    if p is None:
                        self.parameters = ptype()
                    else:
                        self.parameters = p

                def greet(self):
                    print("WaaaaaasGeeeeeehtAaaaaaab")
                    print("My Parameters are: ", self.parameters)

            return NewCls

        return decorated(ptype)

    return decorator

def parametrized(_cls=None, ptype=None):

    def wrap(cls):
        # implement __init__ and other default functions we need
        def new_greet(self):
            print('Jowaheeeee')

        def init(self, p:ptype = None):
            if p is None:
                self.parameters = ptype()
            else:
                self.parameters = p

        setattr(cls, 'greet', new_greet)
        setattr(cls, '__init__', init)
        setattr(cls, 'parameters', 0)

        return cls

    # makes calling without arguments possible
    if _cls is None:
        return wrap
    else:
        return wrap(cls)




@dataclass
class Parameter:
    param1: int = 1
    param2: int = 2

@dataclass
class Parameter2:
    paramx: int = 99
    paramy: int = 100


@parametrized(ptype=Parameter)
class ASD:

    def greet(self):
        print("suuuuuup")

@parametrized(ptype=Parameter2)
class DasBeste(ASD):

    def greet(self):
        print("wenn es um rap geht")


if __name__ == "__main__":
    param = Parameter()
    print("param=", param)
    print("type=", type(param))

    asd = ASD()
    asd.greet()
    print("type=",type(asd))

    a = DasBeste()
    a.greet()
    print(a.parameters)
