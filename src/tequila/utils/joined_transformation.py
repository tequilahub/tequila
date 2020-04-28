class JoinedTransformation:
    '''
    class structure used to construct,track, and permit differentiation of the computation required
    by mathematical operations on ExpectationValues,Variables and Objectives thereof.
    JoinedTransformations allow operations to be combined.
    '''

    def __init__(self, left, right, split, op):
        '''
        :param left: Callable: the lefthand operation, one level down
        :param right: Callable: the righthand operation, one level down
        :param split: Int: the position in the call method, at which lefthand and righthand arguments split
        :param op:  Callable: the operation to apply.

        Example: split 2, left = numpy.add, right= numpy.subtract, op = numpy.multiply, call on list of length 4:
        then the Joined Transform performs the following computation when called:
        np.Multiply(np.add(arg_0,arg_1),np.subtract(arg_2,arg_3))
        '''
        self.split = split
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, *args, **kwargs):
        '''

        :param args: iter: the arguments to the transformation.
        :param kwargs: dict: keyword arguments to the transformation. Must be uniform.
        :return: a number, the result of the calculation.
        '''
        E_left = args[:self.split]
        E_right = args[self.split:]
        if self.op is None:
            if len(args) == 1:
                return args[0]
            return None
        if self.right is None:
            if self.left is None:
                print(len(args))
                return self.op(*args)
            else:

                return self.op(self.left(*E_left, **kwargs), *E_right)
        if self.left is None:
            if self.right is None:
                return self.op(*args)
            else:

                return self.op(*E_left, self.right(*E_right, **kwargs))
        else:
            return self.op(self.left(*E_left, **kwargs), self.right(*E_right, **kwargs))
