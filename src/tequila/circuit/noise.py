import typing
from tequila.tools import list_assignement
from tequila.utils import TequilaException
import copy


names_dict={
    'x':'x',
    'y':'y',
    'z':'z',
    'h':'h',
    'rx':'r',
    'ry':'r',
    'rz':'r',
    'r':'r',
    'phase':'r',
    'single':'single',
    'swap':'control',
    'cx':'control',
    'cy':'control',
    'cz':'control',
    'crx':'control',
    'cry':'control',
    'crz':'control',
    'control':'control',
    'cnot':'control',
    'ccnot':'multicontrol',
    'multicontrol':'multicontrol'
}

noises_available=['bit flip','phase flip','phase damp','amplitude damp','phase-amplitude damp','depolarizing']
krausses=['bit flip','phase flip','phase damp','amplitude damp','phase-amplitude damp','depolarizing']

class QuantumNoise():
    prob_length={
        'bit flip':1,
        'phase flip':1,
        'phase damp':1,
        'amplitude damp':1,
        'phase-amplitude damp':2,
        'depolarizing':1
    }
    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    def __init__(self,name:str,probs:typing.List[float],level:int):
        probs=list_assignement(probs)
        if name not in noises_available:
            raise TequilaException('The name you asked for, {}, is not recognized'.format(name))
        self._name=name
        self._level=int(level)

        if len(probs) != self.prob_length[name]:
            raise TequilaException('{} noise requires {} probabilities; recieved {}'.format(name, self.prob_length[name], len(probs)))
        if name in krausses:
            assert sum(probs)<=1.
        self.probs=list_assignement(probs)

    def __str__(self):
        back=self.name
        back+=' on ' + str(self._level) + ' qubit gates'
        back+=', probs = ' +str(self.probs)
        return back

    @staticmethod
    def from_dict(d):
        if type(d) is dict:
            return QuantumNoise(**d)
        elif type(d) is QuantumNoise:
            return d
        else:
            raise TequilaException('object provided in neither a dictionary nor a QuantumNoise.')

class NoiseModel():

    def __init__(self, noises: typing.List[typing.Union[dict, QuantumNoise]]=None):
        if noises is None:
            self.noises=[]
        else:
            self.noises=[QuantumNoise.from_dict(d) for d in list_assignement(noises)]


    def __str__(self):
        back='NoiseModel with: \n'
        for noise in self.noises:
            back += str(noise)
            back += ',\n'
        return back

    def __add__(self, other):
        new=NoiseModel()
        new.noises+=self.noises
        if type(other) is dict:
            new.noises.append(QuantumNoise.from_dict(other))
        elif type(other) is QuantumNoise:
            new.noises.append(other)
        elif hasattr(other,'noises'):
            new.noises.extend(copy.copy(other.noises))
        return new

    def __iadd__(self, other):
        if type(other) is dict:
            self.noises+=QuantumNoise.from_dict(other)
        elif hasattr(other,'noises'):
            self.noises.extend(copy.copy(other.noises))
        return self

    def without_noise_on_level(self,level):
        new=NoiseModel()
        for noise in self.noises:
            if noise.level==level:
                pass
            else:
                new.noises.append(noise)
        return new

    def without_noise_op(self,name):
        new=NoiseModel()
        for noise in self.noises:
            if noise.name==name:
                pass
            else:
                new.noises.append(noise)
        return new

    @staticmethod
    def wrap_noise(other):
        return NoiseModel(noises=other)

def BitFlip(p:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding to application of pauli X with likelihood p.

    Parameters
    ----------
    p: a float, the probability with which the noise is applied.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''

    new=NoiseModel.wrap_noise(QuantumNoise(name='bit flip', probs=list_assignement(p), level=level))
    return new

def PhaseFlip(p:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding to application of pauli Z with likelihood p.

    Parameters
    ----------
    p: a float, the probability with which the noise is applied.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''

    new=NoiseModel.wrap_noise(QuantumNoise(name='phase flip', probs=list_assignement(p), level=level))
    return new


def PhaseDamp(p:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding to phase damping; Krauss map is defined following Nielsen and Chuang;
    E_0= [[1,0],
          [0,sqrt(1-p)]]
    E_1= [[0,0],
          [0,sqrt(p)]]

    Parameters
    ----------
    p: a float, the probability with which the noise is applied.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''

    new=NoiseModel.wrap_noise(QuantumNoise(name='phase damp', probs=list_assignement(p), level=level))
    return new


def AmplitudeDamp(p:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding amplitude damping.
    this channel takes 1 to 0, but leaves 0 unaffected.
    krauss maps:

    E_0= [[1,0],
          [0,sqrt(1-p)]]
    E_1= [[0,sqrt(p)],
          [0,0]]


    Parameters
    ----------
    p: a float, the probability with which the noise is applied.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''

    new=NoiseModel.wrap_noise(QuantumNoise(name='amplitude damp', probs=list_assignement(p), level=level))
    return new

def PhaseAmplitudeDamp(p1:float,p2:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding to simultaneous phase and amplitude damping.

    Parameters
    ----------
    p1: a float, the probability with which AMPLITUDE is damped.
    p2: a float, the probability with which PHASE is damped.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''
    new=NoiseModel.wrap_noise(QuantumNoise(name='phase-amplitude damp', probs=list_assignement([p1, p2]), level=level))
    return new

def DepolarizingError(p:float,level:int):
    '''
    Returns a NoiseModel with a krauss map corresponding to equal
    probabilities of each of the three pauli matrices being applied.

    Parameters
    ----------
    p: a float, the probability with which the noise is applied.
    level: int, the # of qubits in operations to apply this noise to.

    Returns: NoiseModel
    -------

    '''
    new = NoiseModel.wrap_noise(QuantumNoise(name='depolarizing', probs=list_assignement(p), level=level))
    return new