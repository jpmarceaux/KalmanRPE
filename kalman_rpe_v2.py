import pygsti
from scipy.linalg import expm
import numpy as np
from pygsti.tools import unitary_to_superop
from pygsti.modelpacks import smq1Q_XYZI
from matplotlib import pyplot as plt
from pygsti.circuits import Circuit
from quapack.pyRPE import RobustPhaseEstimation
from quapack.pyRPE.quantum import Q as _rpeQ
from tqdm import tqdm
from importlib import reload
import kalman_filters as _filters

SIGX = pygsti.tools.sigmax
SIGY = pygsti.tools.sigmay
SIGZ = pygsti.tools.sigmaz
SIGM = np.array([[0, 1], [0, 0]])
SIGP = np.array([[0, 0], [1, 0]])

MEAS_0 = (0.5*np.eye(2) + 0.5*SIGZ).flatten()
MEAS_PLUS = (0.5*np.eye(2) + 0.5*SIGX).flatten()
MEAS_RIGHT = (0.5*np.eye(2) + 0.5*SIGY.T).flatten()

PREP_0 = (0.5*np.eye(2) + 0.5*SIGZ).flatten()
PREP_PLUS = (0.5*np.eye(2) + 0.5*SIGX).flatten()
PREP_RIGHT = (0.5*np.eye(2) + 0.5*SIGY).flatten()

PREP_DICT = {
    '0': PREP_0,
    '+': PREP_PLUS,
    'R': PREP_RIGHT,
}

MEAS_DICT = {
    '0': MEAS_0,
    '+': MEAS_PLUS,
    'R': MEAS_RIGHT,
}

def make_standard_experiment_at_depth(d):
    return [
        (d, '0', '0'),
        (d, '0', 'R'),
    ]


def make_dephasing_generator():
    return 0.5*(np.kron(SIGZ, SIGZ) - np.eye(4))

def make_depolarizing_generator():
    return 0.5*(np.kron(SIGX, SIGX) + np.kron(SIGY, SIGY) + np.kron(SIGZ, SIGZ) - 3*np.eye(4))

def make_decay_generator():
    return (np.kron(np.transpose(SIGP), SIGM) - 
            0.5*(np.kron(np.eye(2), SIGP@SIGM) + np.kron(np.transpose(SIGP@SIGM), np.eye(2))))

def make_unitary_generator(P):
    return (1j/2)*(np.kron(np.eye(2), P) - np.kron(np.conj(P), np.eye(2)))


def make_model_process_matrix(theta, L1, Lphi, Ldepol):
    unitary_gen = make_unitary_generator(SIGX)
    dephasing_gen = make_dephasing_generator()
    decay_gen = make_decay_generator()
    depol_gen = make_depolarizing_generator()
    return expm((theta+np.pi/2)*unitary_gen + L1*dephasing_gen + Lphi*decay_gen + Ldepol*depol_gen)


def change_basis_to_pauli_transfer(op):
    pmats = [np.eye(2), SIGX, SIGY, SIGZ]
    op_ptransfer = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            op_ptransfer[i, j] = (1/2)*pmats[i].T.flatten()@op@pmats[j].flatten()
    return op_ptransfer.real


def probability(xstate, d, prep, meas):
    # pad the state with zeros
    xstate = np.array(xstate)
    xstate = np.pad(xstate, (0, 4 - len(xstate)))
    pmat = make_model_process_matrix(*xstate)
    rpe_op = np.linalg.matrix_power(pmat, d)
    p = MEAS_DICT[meas]@rpe_op@PREP_DICT[prep]
    assert p.imag < 1e-10
    return p.real

def dprob(xstate, d, prep, meas, epsilon=1e-6):
    grads = []
    for i in range(3):
        xstate_p = xstate.copy()
        xstate_m = xstate.copy()
        xstate_p[i] += epsilon
        xstate_m[i] -= epsilon
        grads.append((probability(xstate_p, d, prep, meas) - probability(xstate_m, d, prep, meas))/(2*epsilon))
    return np.array(grads)

def rpe_probabilities(xstate, circ_defs):
    pvec = np.zeros(len(circ_defs))
    for idx, (d_i, prep_i, meas_i) in enumerate(circ_defs):
        pvec[idx] = probability(xstate, d_i, prep_i, meas_i)
    return pvec

def rpe_grads(xstate, circ_defs):
    gvec = np.zeros((len(circ_defs), 3))
    for idx, (d_i, prep_i, meas_i) in enumerate(circ_defs):
        gvec[idx, :] = dprob(xstate, d_i, prep_i, meas_i)
    return gvec

def rpe_observation(xstate, circ_defs, num_shots):
    pvec = rpe_probabilities(xstate, circ_defs)
    pvec = np.clip(pvec, 1e-10, 1-1e-10)
    return np.random.binomial(num_shots, pvec)


def step_xstate(xstate, q):
    return xstate + np.array([np.random.normal(0, np.sqrt(q)), 0, 0, 0])

def make_xstate_timseries(xstart, q, num_steps):
    xstates = [xstart]
    x = xstart.copy()
    for i in range(num_steps-1):
        x = step_xstate(x, q)
        xstates.append(x)
    return np.array(xstates)

def dprob(xstate, d, prep, meas, epsilon=1e-6):
    grads = []
    for i in range(3):
        xstate_p = xstate.copy()
        xstate_m = xstate.copy()
        xstate_p[i] += epsilon
        xstate_m[i] -= epsilon
        grads.append((probability(xstate_p, d, prep, meas) - probability(xstate_m, d, prep, meas))/(2*epsilon))
    return np.array(grads)

def rpe_probabilities(xstate, circ_defs):
    pvec = np.zeros(len(circ_defs))
    for idx, (d_i, prep_i, meas_i) in enumerate(circ_defs):
        pvec[idx] = probability(xstate, d_i, prep_i, meas_i)
    return pvec

def rpe_grads(xstate, circ_defs):
    gvec = np.zeros((len(circ_defs), 3))
    for idx, (d_i, prep_i, meas_i) in enumerate(circ_defs):
        gvec[idx, :] = dprob(xstate, d_i, prep_i, meas_i)
    return gvec

def rpe_observation(xstate, circ_defs, num_shots):
    pvec = rpe_probabilities(xstate, circ_defs)
    pvec = np.clip(pvec, 1e-10, 1-1e-10)
    return np.random.binomial(num_shots, pvec)


