#### these classes make the basic structure of our system work,
#### but will be replaced in total when I understand JAX. They are here in the mean time to make sense of everything.
class Add:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, main):
        return main + self.constant

    def grad(self, main):
        return 1.0


class Power:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, main):
        return main ** self.constant

    def grad(self, main):
        return self.constant * main ** (self.constant - 1.0)


class Multiply:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, main):
        return main * self.constant

    def grad(self, main):
        return self.constant


class Divide:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, main):
        return main / self.constant

    def grad(self, main):
        return 1 / self.constant
