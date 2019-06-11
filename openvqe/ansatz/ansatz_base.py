


class AnsatzBase:
    """
    Base Class for the VQE Ansatz
    Derive all specializations from this Base Class
    """

    def __call__(self):
        raise NotImplementedError()

    def greet(self):
        print("Hello from the " + type(self).__name__ + " class")
