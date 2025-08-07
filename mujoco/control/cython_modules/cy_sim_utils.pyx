# cy_sim_utils.pyx

# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, language_level=3

import numpy as np
cimport numpy as cnp

# --- Preprocesses drone sensors and robot sensors separately ---
def extract_sensor_values(double[:] sensorData, list sensorInfo):
    cdef dict body_values = {}
    cdef dict robot_values = {}
    cdef Py_ssize_t i = 0, dim
    cdef str name
    for info in sensorInfo:
        name = info[0]
        dim = info[1]
        if name.startswith("body_"):
            body_values[name] = np.asarray(sensorData[i:i+dim])
        elif name.startswith("robot_"):
            robot_values[name] = np.asarray(sensorData[i:i+dim])
        i += dim
    return body_values, robot_values


# --- Applies control ---
def apply_control_inputs(double[:] ctrl, dict actuatorsIDs, dict controlInputs):
    cdef str name
    cdef int idx
    for name, value in controlInputs.items():
        idx = actuatorsIDs.get(name, -1)
        if idx >= 0:
            ctrl[idx] = value