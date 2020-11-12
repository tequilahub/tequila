from .utils_ml import preamble,TequilaMLException
from tequila.objective import Objective

def to_tensorflow(objective: Objective,compiler_args: dict = None,
                  input_vars: list = None, weight_vars: list = None):
    raise TequilaMLException('Sorry, tensorflow support not yet implemented.')