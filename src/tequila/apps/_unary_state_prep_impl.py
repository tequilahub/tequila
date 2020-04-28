'''
A program that can generate an arbitrary quantum state.

Some conditions: if the number of states exceed the total number of bits,
it's not guaranteed that the state can be prepared.

Code by Maha Kesebi, Toronto 2019
Implemented with OpenVQE structures

Needs complete re-implementation at some point

'''

from tequila.circuit import QCircuit
from tequila.circuit.gates import CNOT, Ry, X
from tequila.objective.objective import Variable
import sympy


class SympyVariable(Variable):
    """
    Still need this structure for the dagger operation
    """

    def __init__(self, name=None):
        self._name = name

    def __call__(self, *args, **kwargs):
        return self._name

    def __neg__(self):
        return SympyVariable(name=-self._name)


class UnaryStatePrepImpl:

    def alphabet(self, i: int) -> str:
        return self.alphabets[i]
        #return "angle_{i}".format(i=i)

    def __init__(self):
        self.left_compressed = []
        self.alph_index = 0

        # currenyly needs one-character symbols
        # needs complete reimplementation
        self.alphabets = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

        self.silenced = True

        self.coefficients = []
        self.c_i = 0

    '''
    Calculates the Hamming distance between two strings of binary numbers.
    notes: if one of the strings contains a symbol (compressed) at an index
    where the other string is not compressed, then the H_dist is incremented
    by 1.
    
    if two strings have a symbol (compressed) at the same index then the H_dist
    is 1. ex. '01a0' and '01b0' have a H_dist of 1. while '01a0' and '0100' has
    a H_dist of 2.
    TODO: revisit this notation of calculating H_dist ^^
    
    returns a list where the first element is the distance, and the second
    element is a list of the different indices of which the two strings
    differ.
    '''

    def calc_H_distance(self, s1, s2):

        distance = 0
        diff_indices = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                if (s1[i] != '0' and s1[i] != '1') or (s2[i] != '0' and s2[i] != '1'):
                    if not ((s1[i] != '0' and s1[i] != '1') and (s2[i] != '0' and s2[i] != '1')):
                        distance += 1
                distance += 1
                diff_indices.append(i)

        return [distance, diff_indices]  ### type [ int, list[int, int ...]]

    '''
    This function find the H_dist between all different combination of pairs
    in s.
    
    Returns a dictionary where the keys represent the Hamming distance, and the
    values are a list containing lists of the different pairs that have that H_dist 
    '''

    def get_pairs(self, s):

        # intialize dictionary according to length of s
        dt = {}
        for length in range(len(s[0]) + 2):
            dt[length] = []

        i = 0
        j = 1

        if len(s) == 2:  # base case
            dist = self.calc_H_distance(s[i], s[j])
            if dist[0] == 0:  # special case, strings identical
                dt[0].append([s[i], s[j]])  # appends pair to dt

            dt[dist[0]].append([s[i], s[j]])

            return dt

        while (i < len(s) - 1):
            if j >= len(s) - 1 and i == len(s) - 2:
                break

            if j == len(s):
                i += 1
                j = i + 1

            dist = self.calc_H_distance(s[i], s[j])
            if dist[0] == 0:  # special case, strings identical
                dt[0].append([s[i], s[j]])  # appends pair to dt
                i += 2
                j += 2

            dt[dist[0]].append([s[i], s[j]])
            j += 1

        return dt  ### tpye dict{ key=int: value = list[ list[string, string],  list[str, str], ..] }

    '''
    This function takes s (a list of the states) and 2 indices. It find all
    possible CNOT moves to be made 
    
    returns a dictionary, where the ket represents a target bit's index
    and the value is a list of tuples, where the tuples hold possible control bit
    index for that target index, and the control type represented by '0' for anti
    and '1' for a normal control.
    '''

    def get_CNOT_moves(self, s, i1, i2):
        targets = self.calc_H_distance(s[i1], s[i2])[1]

        poss_moves = {}  # {target: [ (control index, type), (,)], target2: ... }

        s_to_check = s.copy()
        s_to_check.remove(s[i1])
        s_to_check.remove(s[i2])

        for target in targets:
            poss_moves[target] = []

            for index in range(len(s[i1])):  # index of a bit in the string that could possibly be a control.
                if index != target and s[i1][index] != s[i2][index] and (
                        s[i1][index] == '0' or s[i1][index] == '1') and (
                        s[i2][index] == '0' or s[i2][index] == '1'):  # control != target & != other string & != 'a'
                    i = 0  # use i to loop through strings to check for control
                    flag = True
                    while i < len(s_to_check) and flag == True:
                        if s[i1][index] == s_to_check[i][index] or not (
                                s_to_check[i][index] == '0' or s_to_check[i][index] == '1'):  # making sure its a 1 or 0
                            flag = False  # can't be used for control
                        i += 1

                    if flag == True:
                        poss_moves[target].append((index, s[i1][index]))

                    ##check with s[i2]
                    flag = True
                    i = 0  # loop through strings and check for control

                    while i < len(s_to_check) and flag == True:
                        if s[i2][index] == s_to_check[i][index] or not (
                                s_to_check[i][index] == '0' or s_to_check[i][index] == '1'):
                            flag = False
                        i += 1

                    if flag == True:
                        poss_moves[target].append((index, s[i2][index]))

        return poss_moves  # type  is  dict{key = int(taget bit's index):  value = list[ tuple( int(control bit index), str(control type)),  .. ] }

    '''
    
    '''

    def get_CNOT_move_non_opt(self, s, i1, i2, target_index):  # TODO: make this more opt

        for index in range(len(s[i1])):

            if index != target_index and s[i1][index] != s[i2][index] and (
                    s[i1][index] == '0' or s[i1][index] == '1') and (s[i2][index] == '0' or s[i2][index] == '1'):
                return index

        raise Exception('index not found')

    '''
    
    '''

    def get_a_compression(self, s, i1, i2):
        global alph_index
        target_index = self.calc_H_distance(s[i1], s[i2])[1][
            0]  # collect target index, the bit where they are different
        move_param = None

        if not self.silenced: print("compression of index {} in strings {} and {}".format(target_index, s[i1], s[i2]))

        string_changed = s[i1]
        string = s[i1]
        string = list(string)
        string[target_index] = self.alphabet(self.alph_index)
        self.alph_index += 1
        string = "".join(string)
        s[i1] = string

        if s[i2] in self.left_compressed:
            self.left_compressed.remove(s[i2])
        if string_changed in self.left_compressed:
            self.left_compressed.remove(string_changed)

        s.remove(s[i2])

        if not self.silenced: print(s)

        return [s, move_param]

    def get_0_rotation(self, s, i):

        for a_index in range(len(s[i])):
            if not (s[i][a_index] == '0' or s[i][a_index] == '1'):
                break
        target_index = a_index

        # find control
        s_to_check = s.copy()
        s_to_check.remove(s[i])

        possible_indicies = []
        c_type = None

        for index in range(len(s[i])):
            if index != target_index:

                flag = True

                for string in s_to_check:
                    if string[index] == s[i][index]:
                        flag = False  # not a possible control

                if flag == True:
                    c_type = s[i][index]

                    possible_indicies.append([index, c_type])  # tpye  is a list[ int(index of control, str(c_type)]

        if len(possible_indicies) == 0:
            # a compression is left with a symbol 'a' not rotated
            # add that string to a global variable
            self.left_compressed.append(s[i])
        else:
            self.coefficients.append(s[i][a_index])
        return possible_indicies

    def make_move(self, s, i1, i2, move, m_type):

        move_param = None
        if m_type == 'ROT':

            i = 0  # will be used to loop through s

            while i < len(s) and i >= 0:

                control_bit_value = s[i][move[0]]  # a string
                control_type = move[2]  # a string

                control_bit_index = move[0]  # an int
                target_bit_index = move[1]  # an int

                if control_bit_value == control_type:  # control matches type -> make move

                    string = s[i]
                    string = list(string)
                    string[target_bit_index] = '0'
                    string = "".join(string)
                    s[i] = string

                    if s.count(s[i]) > 1:  # after making the move, on the string, if it's repeated remove
                        s.remove(s[i])  # here should add coefficients
                        i -= 1

                    if control_type == '0':
                        c_type = 'anti'
                    else:
                        c_type = 'positive'

                    # if not silenced: print('move: {} control rotation from control bit index = {} and target bit index = {}'.format(c_type, control_bit_index, target_bit_index))
                    if c_type == 'anti':
                        m_type = 'aCROT'
                    else:
                        m_type = 'CROT'

                    move_param = [m_type, control_bit_index, target_bit_index]
                    if not self.silenced: print(move_param)

                i += 1

            return [s, move_param]


        elif m_type == 'CNOT':

            i = 0
            while i < len(s) and i >= 0:

                control_bit_value = s[i][move[0]]  # a string
                control_type = move[2]  # a string

                control_bit_index = move[0]  # an int
                target_bit_index = move[1]  # an int

                if control_bit_value == control_type:  # control matches type -> make move
                    if s[i][target_bit_index] == '0':

                        string = s[i]
                        string = list(string)
                        string[target_bit_index] = '1'
                        string = "".join(string)
                        s[i] = string


                    else:
                        string = s[i]
                        string = list(string)
                        string[target_bit_index] = '0'
                        string = "".join(string)
                        s[i] = string

                    if s.count(s[i]) > 1:  # after making the move, on the string, if it's repeated remove
                        s.remove(s[i])  # here should add coefficients
                        i -= 1

                    if control_type == '0':
                        c_type = 'anti'
                    else:
                        c_type = 'positive'

                    # if not silenced: print('move: {} control NOT from control bit index = {} and target bit index = {}'.format(c_type, control_bit_index, target_bit_index))
                    if c_type == 'anti':
                        m_type = 'aCNOT'
                    move_param = [m_type, control_bit_index, target_bit_index]

                    if not self.silenced: print(move_param)
                i += 1

            return [s, move_param]

    def get_next_move(self, s, moves_list):

        # first_key = next(iter(d))
        strings = s

        if len(strings) == 1:  # done

            # add X gates to get '0000'
            for index in range(len(strings[0])):
                if strings[0][index] == '1':
                    move = ['X', None, index]
                    moves_list.append(move)
            if not self.silenced: print('DONE', ' s=', s)
            return

        d = self.get_pairs(strings)  # d is a dictionary returned from get_pairs fcn call

        found = False

        i = 0

        keys = list(d.keys())
        first_key = keys[i]

        while found == False and i < len(keys) - 1:
            if len(d[first_key]) != 0:
                found = True
            else:
                first_key = keys[i + 1]
            i += 1

        pair = d[first_key][0]  # gets the first pair of strings to operate on - will be the one w/ lowest H_dist

        index1 = strings.index(pair[0])
        index2 = strings.index(pair[1])

        if first_key == 0:  # there are 2 strings that are the same
            strings.remove(strings[index1])  # here is where you would add coefficients
            self.get_next_move(strings)  # recursive call

        if first_key == 1:  # need a rot move

            strings = self.get_a_compression(strings, index1, index2)[0]
            ####TODO add moves hereeee

            possible_rot_controls = self.get_0_rotation(strings, index1)

            if len(possible_rot_controls) > 0:  # you can do a rotation
                control_bit_index = possible_rot_controls[0][0]
                control_type = possible_rot_controls[0][1]
                for a_index in range(len(strings[index1])):
                    if not (strings[index1][a_index] == '0' or strings[index1][a_index] == '1'):
                        break
                target_bit_index = a_index

                move = [control_bit_index, target_bit_index, control_type]  ###Type [int, int, str]

                strings_moves = self.make_move(strings, index1, index2, move,
                                               'ROT')  # fcn makes move and change all strings
                strings = strings_moves[0]
                moves_list.append(strings_moves[1])

                if not self.silenced: print(strings)

            # recursively call fcn on new strings
            self.get_next_move(strings, moves_list)


        elif first_key >= 2:  # needs a CNOT move

            a_count1 = 0
            a_count2 = 0
            for index in range(len(strings[index1])):
                if not (strings[index1][index] == '0' or strings[index1][index] == '1') and not (
                        strings[index2][index] == '0' or strings[index2][index] == '1'):
                    a_count1 += 0
                    # basically do nothing
                elif not (strings[index1][index] == '0' or strings[index1][index] == '1'):
                    a_count1 += 1
                elif not (strings[index2][index] == '0' or strings[index2][index] == '1'):
                    a_count2 += 1

            if (
                    a_count1 > 0 or a_count2 > 0):  # TODO: test if there will be problems if bth strings contain 'a' @ same position
                if a_count1 > a_count2:
                    first_key -= a_count1
                else:
                    first_key -= a_count2

                if first_key == 1:
                    # do what u would do in first_key == 1
                    strings = self.get_a_compression(strings, index1, index2)[0]

                    possible_rot_controls = self.get_0_rotation(strings, index1)

                    if len(possible_rot_controls) > 0:  # you can do a rotation
                        control_bit_index = possible_rot_controls[0][0]
                        control_type = possible_rot_controls[0][1]
                        for a_index in range(len(strings[index1])):
                            if not (strings[index1][a_index] == '0' or strings[index1][a_index] == '1'):
                                break
                        target_bit_index = a_index

                        move = [control_bit_index, target_bit_index, control_type]  ###Type [int, int, str]

                        strings_moves = self.make_move(strings, index1, index2, move,
                                                       'ROT')  # fcn makes move and change all strings
                        strings = strings_moves[0]
                        moves_list.append(strings_moves[1])

                        if not self.silenced: print(strings)

                    # recursively call fcn on new strings
                    self.get_next_move(strings, moves_list)
                    return 'Done'

            poss_c_moves = self.get_CNOT_moves(strings, index1, index2)
            first_poss_move = next(iter(poss_c_moves))  # gets first key
            target_bit_index = first_poss_move
            if len(poss_c_moves[target_bit_index]) == 0:
                if not self.silenced: print(target_bit_index)
                # find a c-move where u will have to change another string with it
                # tlater:  make this more optimal -> find the index in which the least # of states will be effected

                control_bit_index = self.get_CNOT_move_non_opt(strings, index1, index2, target_bit_index)

                if control_bit_index == None:
                    # must be something with a rotation
                    string_to_rotate = self.left_compressed[0]
                    possible_rot_controls = self.get_0_rotation(strings, strings.index(string_to_rotate))
                    if len(possible_rot_controls) > 0:  # you can do a rotation
                        control_bit_index = possible_rot_controls[0][0]
                        control_type = possible_rot_controls[0][1]
                        for a_index in range(len(strings[index1])):
                            if not (strings[index1][a_index] == '0' or strings[index1][a_index] == '1'):
                                break
                        target_bit_index = a_index

                        move = [control_bit_index, target_bit_index, control_type]  ###Type [int, int, str]

                        strings_moves = self.make_move(strings, index1, index2, move,
                                                       'ROT')  # fcn makes move and change all strings
                        strings = strings_moves[0]
                        moves_list.append(strings_moves[1])

                        if not self.silenced: print(strings)
                        self.left_compressed.remove(strings.index(string_to_rotate))
                    # recursively call fcn on new strings
                    self.get_next_move(strings, moves_list)
                    return 'Done'

                control_type = strings[index2][
                    control_bit_index]  # can choose strings[index1] or [index2] -> make more optimal by checking which one doesn't affect a lot of other strings

            else:
                control_bit_index = poss_c_moves[target_bit_index][0][0]
                control_type = poss_c_moves[target_bit_index][0][1]

            move = [control_bit_index, target_bit_index, control_type]  ###Type [int, int, str]

            strings_moves = self.make_move(strings, index1, index2, move, 'CNOT')
            strings = strings_moves[0]
            moves_list.append(strings_moves[1])

            if not self.silenced: print(strings)

            self.get_next_move(strings, moves_list)

        return moves_list

    #### symbolic circuit code ####
    def create_sub_circ(self, gate, control_bit, target_bit):
        sub_circ = None

        if gate == "CNOT":
            sub_circ = CNOT(control=control_bit, target=target_bit)

        elif gate == "aCNOT":
            sub_circ = X(control_bit)
            sub_circ += CNOT(control=control_bit, target=target_bit)
            sub_circ += X(control_bit)

        elif gate == "CROT":
            a_angle = self.coefficients[self.c_i]
            sa = sympy.Symbol(a_angle)
            self.c_i += 1
            sub_circ = Ry(control=control_bit, target=target_bit, angle=SympyVariable(sa))

        elif gate == "aCROT":
            a_angle = self.coefficients[self.c_i]
            sa = sympy.Symbol(a_angle)
            self.c_i += 1
            sub_circ = X(control_bit)
            sub_circ += Ry(control=control_bit, target=target_bit, angle=SympyVariable(sa))
            sub_circ += X(control_bit)

        elif gate == "X":
            sub_circ = X(target_bit)

        elif gate == "ROT":
            a_angle = self.coefficients[self.c_i]
            self.c_i += 1
            sub_circ = Ry(control=None, target=target_bit, angle=SympyVariable(sympy.Symbol(a_angle)))

        return sub_circ

    #####

    '''
    This function extracts the equations from the state and returns them
    '''

    def get_equations(f_state):

        equations = []

        for s, v in f_state.state.items():
            equations.append(v)

        return equations

    ### main ###

    def get_circuit(self, s):
        '''
        s = []
        for counter in range(4):
            string = input("Enter your state: ")
            s.append(string)
        if not silenced: print(s)
        '''
        if not self.silenced: print("s on start=", s)
        moves_list = []
        moves_list = self.get_next_move(s, moves_list)
        if not self.silenced: print("last s=", s)

        circuit = QCircuit()
        for move in moves_list:
            circuit += self.create_sub_circ(move[0], move[1], move[2])

        return circuit.dagger()
