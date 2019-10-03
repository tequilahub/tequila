from copy import deepcopy


def dictdata(_cls=None):
    """
    Decorator to wrap dictionary in higher dataclass
    automatically implements +,-,+=,-=, ==, and scalar multiplication
    as well as: len, [], in
    :param data_name: Name the data member in your final class
    :param _cls: failsafe that the decorator can be set with () as well as without
    :return: decorated class
    """

    data_name: str = "data"

    def decorator(cls):

        @property
        def impldata(self):
            if getattr(self, "_" + str(data_name)) is None:
                return dict()
            else:
                return getattr(self, "_" + str(data_name))

        def get_item(self, item):
            return getattr(self, str(data_name))[item]

        def set_item(self, key, value):
            getattr(self, str(data_name))[key] = value
            return self

        def add(self, other):
            result = deepcopy(self)
            data = getattr(result, "_" + str(data_name))
            for k, v in other.items():
                if k in data:
                    data[k] += v
                else:
                    data[k] = v
            return result

        def iadd(self, other):
            data = getattr(self, "_" + str(data_name))
            for k, v in other.items():
                if k in data:
                    data[k] += v
                else:
                    data[k] = v
            return self

        def isub(self, other):
            data = getattr(self, "_" + str(data_name))
            for k, v in other.items():
                if k in data:
                    data[k] -= v
                else:
                    data[k] = -1.0 * v
            return self

        def sub(self, other):
            result = deepcopy(self)
            data = getattr(result, "_" + str(data_name))
            for k, v in other.items():
                if k in data:
                    data[k] -= v
                else:
                    data[k] = -1.0 * v
            return result

        def rmul(self, value):
            result = deepcopy(self)
            for k, v in result.items():
                result[k] *= value
            return result

        def repr(self):
            return str(getattr(self, "_" + str(data_name)))

        def items(self):
            return getattr(self, str(data_name)).items()

        def values(self):
            return getattr(self, str(data_name)).values()

        def keys(self):
            return getattr(self, str(data_name)).keys()

        def len_(self):
            return len(getattr(self, str(data_name)))

        def eq(self, other):
            return getattr(self, str(data_name)) == getattr(other, str(data_name))

        def contains(self, key):
            return key in getattr(self, str(data_name))

        def init(self, data: dict = None):
            if data is None:
                self._data = dict()
            else:
                self._data = data

        #setattr(cls, "_"+str(data_name), None)
        setattr(cls, "__init__", init)
        setattr(cls, str(data_name), impldata)
        setattr(cls, "items", items)
        setattr(cls, "values", values)
        setattr(cls, "keys", keys)
        setattr(cls, "__len__", len_)
        setattr(cls, "__contains__", contains)
        setattr(cls, "__eq__", eq)
        setattr(cls, "__setitem__", set_item)
        setattr(cls, "__setitem__", set_item)
        setattr(cls, "__getitem__", get_item)
        setattr(cls, "__add__", add)
        setattr(cls, "__sub__", sub)
        setattr(cls, "__rmul__", rmul)
        setattr(cls, "__repr__", repr)

        return cls

    if _cls is None:
        return decorator
    else:
        return decorator(cls=_cls)


@dictdata
class TestClass:

    def hello(self):
        print("hello")


if __name__ == "__main__":
    a = dict()
    a[0] = 1
    a[1] = 2

    test = TestClass()
    print(test.__dict__)
    #
    test[0] = 1
    test[1] = 2

    print(test.data)
    print(test)

    test2 = TestClass()
    test2[1] = test[0]
    test2[2] = 4

    print(test == test2)
    print(test.data)
    print(test2.data)
