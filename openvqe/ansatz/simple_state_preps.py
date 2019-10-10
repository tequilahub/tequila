from openvqe.circuit import QCircuit, gates
from openvqe import BitString

def prepare_product_state(state:BitString) -> QCircuit:
    result = QCircuit()
    for i,v in enumerate(state.array):
        if v==1:
            result+=gates.X(target=i)
    return result