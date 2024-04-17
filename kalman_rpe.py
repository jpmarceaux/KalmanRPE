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
from scipy.optimize import minimize
from scipy.stats import multinomial, binom

def make_noisy_z_unitary(alpha):
    generator = -(1j/2) *(np.pi/2 + alpha)* pygsti.sigmaz
    return expm(generator)

def make_noisy_x_unitary(epsilon, theta):
    generator = -(1j/2) * (np.pi/2 + epsilon) * (np.cos(theta) * pygsti.sigmax + np.sin(theta) * pygsti.sigmaz)
    return expm(generator)

def make_pygsti_model(xstate, gate_depolarization=0, spam_depolarization=0):
    alpha = xstate[0]
    theta = xstate[1]
    epsilon = xstate[2]
    zunitary = make_noisy_z_unitary(alpha)
    xunitary = make_noisy_x_unitary(theta, epsilon)
    model = smq1Q_XYZI.target_model(simulator='map')
    model['Gxpi2', 0] = pygsti.tools.unitary_to_superop(xunitary)
    model['Gzpi2', 0] = pygsti.tools.unitary_to_superop(zunitary)
    model['Gxpi2', 0].depolarize(gate_depolarization)
    model['Gzpi2', 0].depolarize(gate_depolarization)
    model['rho0'].depolarize(spam_depolarization)
    model['Mdefault'].depolarize(spam_depolarization)
    return model


def make_x_cos_circuit(depth):
    return Circuit([('Gxpi2',0 )])*(4*depth)

def make_x_sin_circuit(depth):
    return Circuit([('Gxpi2', 0)])*(4*depth+1)

def make_z_cos_circuit(depth):
    Gy = Circuit([('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)])
    Gy_dagger = Circuit([('Gzpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0)])
    return Gy+Circuit([('Gzpi2',0 )])*(4*depth)+Gy_dagger

def make_z_sin_circuit(depth):
    Gy = Circuit([('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)])
    Gy_dagger = Circuit([('Gzpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0)])
    return Gy+Circuit([('Gzpi2', 0)])*(4*depth)+Circuit([('Gxpi2', 0)])

def make_interleaved_cos_circuit(depth):
    return Circuit([('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)])*depth

def make_interleaved_sin_circuit(depth):
    return Circuit([('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)])*depth+Circuit([('Gxpi2', 0)])

def make_x_sequences(depths):
    cos_circs = []
    sin_circs = []
    for d in depths:
        cos_circs.append(make_x_cos_circuit(d))
        sin_circs.append(make_x_sin_circuit(d))
    return cos_circs, sin_circs

def make_z_sequences(depths):
    cos_circs = []
    sin_circs = []
    for d in depths:
        cos_circs.append(make_z_cos_circuit(d))
        sin_circs.append(make_z_sin_circuit(d))
    return cos_circs, sin_circs

def make_interleaved_sequences(depths):
    cos_circs = []
    sin_circs = []
    for d in depths:
        cos_circs.append(make_interleaved_cos_circuit(d))
        sin_circs.append(   make_interleaved_sin_circuit(d))
    return cos_circs, sin_circs


def process_zgate_data(dataset, cos_circs, sin_circs, depths):
    experiment = _rpeQ()
    for idx, d in enumerate(depths):
        experiment.process_cos(d, (int(dataset[cos_circs[idx]]['0']), int(dataset[cos_circs[idx]]['1'])))
        experiment.process_sin(d, (int(dataset[sin_circs[idx]]['0']), int(dataset[sin_circs[idx]]['1'])))
    analysis = RobustPhaseEstimation(experiment)
    last_good_generation = analysis.check_unif_local(historical=True)
    estimates = analysis.angle_estimates
    return estimates, last_good_generation

def process_xgate_data(dataset, cos_circs, sin_circs, depths):
    experiment = _rpeQ()
    for idx, d in enumerate(depths):
        experiment.process_cos(d, (int(dataset[cos_circs[idx]]['0']), int(dataset[cos_circs[idx]]['1'])))
        experiment.process_sin(d, (int(dataset[sin_circs[idx]]['1']), int(dataset[sin_circs[idx]]['0'])))
    analysis = RobustPhaseEstimation(experiment)
    last_good_generation = analysis.check_unif_local(historical=True)
    estimates = analysis.angle_estimates
    return estimates, last_good_generation

def process_interleaved_data(dataset, cos_circs, sin_circs, depths):
    experiment = _rpeQ()
    for idx, d in enumerate(depths):
        experiment.process_cos(d, (int(dataset[cos_circs[idx]]['0']), int(dataset[cos_circs[idx]]['1'])))
        experiment.process_sin(d, (int(dataset[sin_circs[idx]]['1']), int(dataset[sin_circs[idx]]['0'])))
    analysis = RobustPhaseEstimation(experiment)
    last_good_generation = analysis.check_unif_local(historical=True)
    estimates = analysis.angle_estimates
    return estimates, last_good_generation


def rectify_phase(phase):
    if phase > np.pi:
        return phase - 2*np.pi
    elif phase < -np.pi:
        return phase + 2*np.pi
    else:
        return phase

def make_error_estimate(z_phase, x_phase, interleaved_phase):
    z_phase = rectify_phase(z_phase)
    x_phase = rectify_phase(x_phase)
    interleaved_phase = rectify_phase(interleaved_phase)
    alpha = z_phase/4
    epsilon = x_phase/4
    theta = np.sin(interleaved_phase/2)/(2*np.cos(np.pi*epsilon/2)) # Equation III.11
    return alpha, epsilon, theta

def make_rpe_circuits(depths):
    x_cos_circs, x_sin_circs = make_x_sequences(depths)
    z_cos_circs, z_sin_circs = make_z_sequences(depths)
    interleaved_cos_circs, inerleaved_sin_circs = make_interleaved_sequences(depths)
    return set(x_cos_circs + x_sin_circs + z_cos_circs + z_sin_circs + interleaved_cos_circs + inerleaved_sin_circs)

def simulate_rpe_experiment(xerror, depths, gate_depolarization, spam_depolarization, samples_per_circuit, seed=None):
    model = make_pygsti_model(xerror, gate_depolarization, spam_depolarization)
    all_circs = make_rpe_circuits(depths)
    dataset = pygsti.data.simulate_data(model, all_circs, num_samples=samples_per_circuit, seed=seed)
    return dataset

def make_rpe_estimate(dataset, depths):
    x_cos_circs, x_sin_circs = make_x_sequences(depths)
    z_cos_circs, z_sin_circs = make_z_sequences(depths)
    interleaved_cos_circs, inerleaved_sin_circs = make_interleaved_sequences(depths)
    z_estimates, z_last_good_generation = process_zgate_data(dataset, z_cos_circs, z_sin_circs, depths)
    x_estimates, x_last_good_generation = process_xgate_data(dataset, x_cos_circs, x_sin_circs, depths)
    interleaved_estimates, interleaved_last_good_generation = process_interleaved_data(dataset, interleaved_cos_circs, inerleaved_sin_circs, depths)
    min_last_good_gen = min(z_last_good_generation, x_last_good_generation, interleaved_last_good_generation)
    alpha, epsilon, theta = make_error_estimate(z_estimates[min_last_good_gen], x_estimates[min_last_good_gen], interleaved_estimates[min_last_good_gen])
    direct_error_bound = np.pi/(8*2**min_last_good_gen)
    theta_error_bound = direct_error_bound/(4*np.cos(np.pi*epsilon/2)) # Equation III.11
    return [alpha, epsilon, theta], [direct_error_bound, direct_error_bound, theta_error_bound], min_last_good_gen

def run_rpe(xerror, depths, gate_depolarization, spam_depolarization, samples_per_circuit):
    dataset = simulate_rpe_experiment(xerror, depths, gate_depolarization, spam_depolarization, samples_per_circuit)
    return make_rpe_estimate(dataset, depths)

def make_observation_probabilities(xstate, d, gate_depolarization=0., spam_depolarization=0.):
    x = xstate.copy()
    alpha, epsilon, theta = x
    model = make_pygsti_model([alpha, epsilon, theta], gate_depolarization, spam_depolarization)
    x_cos_probs = model.probabilities(make_x_cos_circuit(d))
    x_sin_probs = model.probabilities(make_x_sin_circuit(d))
    z_cos_probs = model.probabilities(make_z_cos_circuit(d))
    z_sin_probs = model.probabilities(make_z_sin_circuit(d))
    interleaved_cos_probs = model.probabilities(make_interleaved_cos_circuit(d))
    interleaved_sin_probs = model.probabilities(make_interleaved_sin_circuit(d))
    return x_cos_probs, x_sin_probs, z_cos_probs, z_sin_probs, interleaved_cos_probs, interleaved_sin_probs

def make_probability_vector(xstate, d, gate_depolarization=0., spam_depolarization=0.):
    x_cos_probs, x_sin_probs, z_cos_probs, z_sin_probs, interleaved_cos_probs, interleaved_sin_probs = make_observation_probabilities(xstate, d, gate_depolarization, spam_depolarization)
    return np.array([x_cos_probs['1'], x_sin_probs['1'], z_cos_probs['1'], z_sin_probs['1'], interleaved_cos_probs['1'], interleaved_sin_probs['1']])

def sample_emperical_dist(xstate, d, num_shots):
    pvec = make_probability_vector(xstate, d)
    # each sample is a binoimal draw from the probability distribution
    return np.random.binomial(num_shots, pvec)/num_shots

def prob_vec_to_signals(prob_vector):
    x_complex_signal = (1-2*prob_vector[0]) - 1j*(1-2*prob_vector[1])
    z_complex_signal = (1-2*prob_vector[2]) + 1j*(1-2*prob_vector[3])
    interleaved_complex_signal = (1-2*prob_vector[4]) - 1j*(1-2*prob_vector[5])
    return x_complex_signal, z_complex_signal, interleaved_complex_signal

def format_observation(prob_vector, data_format="cartesian"):
    if data_format == "polar":
        x_complex_signal, z_complex_signal, interleaved_complex_signal = prob_vec_to_signals(prob_vector)
        angles = np.angle(x_complex_signal), np.angle(z_complex_signal), np.angle(interleaved_complex_signal)
        norms = np.abs(x_complex_signal), np.abs(z_complex_signal), np.abs(interleaved_complex_signal)
        return np.array(angles + norms)
    elif data_format == "cartesian":
        return prob_vector
    

def binom_loglikelihood(estimate, dataset, circuits, gate_depolarization=0.0, spam_depolarization=0.0):
    model = make_pygsti_model(estimate, gate_depolarization, spam_depolarization)
    log_likelihood = 0
    for circ in circuits:
        c1 = dataset[circ]['1']
        c0 = dataset[circ]['0']
        p = model.probabilities(circ)['1']
        log_likelihood += binom.logpmf(c1, c1+c0, p)
    return log_likelihood

# def gradient(estimate, dataset, circuits, gate_depolarization=0.0, spam_depolarization=0.0, epsilon=1e-8):
#     grad = np.zeros(3)
#     for idx in range(3):
#         perturbed_estimate = np.copy(estimate)
#         perturbed_estimate[idx] += epsilon
#         grad[idx] = (binom_loglikelihood(perturbed_estimate, dataset, circuits, gate_depolarization, spam_depolarization) - binom_loglikelihood(estimate, dataset, circuits, gate_depolarization, spam_depolarization))/epsilon
#     return grad

def make_fisher_info_matrix(estimate, dataset, circuits, gate_depolarization=0.0, spam_depolarization=0.0, epsilon=1e-8):
    fim = np.zeros((3, 3))
    score_derivatives = np.zeros((3, len(circuits)))
    for idx in range(3):
        perturbed_estimate = np.copy(estimate)
        perturbed_estimate[idx] += epsilon
        for idx2, circ in enumerate(circuits):
            score_derivatives[idx, idx2] = (binom_loglikelihood(perturbed_estimate, dataset, [circ], gate_depolarization, spam_depolarization) - binom_loglikelihood(estimate, dataset, [circ], gate_depolarization, spam_depolarization))/epsilon
    for idx in range(3):
        for idx2 in range(3):
            fim[idx, idx2] = np.sum(score_derivatives[idx]*score_derivatives[idx2])
    return fim

def make_cr_bounds_from_fmat(fisher_info_matrix):
    """cramer rao bounds"""
    # first get the diagonal elements of the inverse of the fisher information matrix
    fim_inv = np.linalg.inv(fisher_info_matrix)
    cr_bounds = np.sqrt(np.diag(fim_inv))
    # just return an array of the square root of the diagonal elements of the inverse of the fisher information matrix
    return cr_bounds

def make_cr_bounds(estimate, dataset, circuits, gate_depolarization=0.0, spam_depolarization=0.0, epsilon=1e-8):
    fim = make_fisher_info_matrix(estimate, dataset, circuits, gate_depolarization, spam_depolarization, epsilon)
    return make_cr_bounds_from_fmat(fim)

def make_mle_estimate(xinitial, dataset, depths, gate_depolarization=0.0, spam_depolarization=0.0, show_progress=False):
    x = xinitial
    xhistory = [x]
    if show_progress:
        for idx in tqdm(range(len(depths))):
            rpe_bound = np.pi/(2**idx)
            circuits = make_rpe_circuits(depths[:idx+1])
            x_sub = minimize(lambda x: -binom_loglikelihood(x, dataset, circuits, gate_depolarization, spam_depolarization), 
                             x, method='TNC')   
            x = x_sub.x
            xhistory.append(x)
    else:
        for idx in range(len(depths)):
                rpe_bound = np.pi/(8*2**idx)
                circuits = make_rpe_circuits(depths[:idx+1])
                x_sub = minimize(lambda x: -binom_loglikelihood(x, dataset, circuits, gate_depolarization, spam_depolarization), 
                                x, method='TNC')   
                x = x_sub.x
                xhistory.append(x)
    return x, xhistory
    
def make_ordered_circuit_list(d):
    x_cos_circ = make_x_cos_circuit(d)
    x_sin_circ = make_x_sin_circuit(d)
    z_cos_circ = make_z_cos_circuit(d)
    z_sin_circ = make_z_sin_circuit(d)
    interleaved_cos_circ = make_interleaved_cos_circuit(d)
    interleaved_sin_circ = make_interleaved_sin_circuit(d)
    return [x_cos_circ, x_sin_circ, z_cos_circ, z_sin_circ, interleaved_cos_circ, interleaved_sin_circ]
