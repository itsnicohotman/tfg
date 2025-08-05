# cy_sim_utils.pyx

# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, language_level=3

import numpy as np
cimport numpy as cnp

# --- 1. Preprocesar sensores ---
def extract_sensor_values(double[:] sensorData, list sensorInfo):
    cdef dict values = {}
    cdef Py_ssize_t i = 0, dim
    cdef str name
    for info in sensorInfo:
        name = info[0]
        dim = info[1]
        values[name] = np.asarray(sensorData[i:i+dim])
        i += dim
    return values


# --- 2. Aplicar control ---
def apply_control_inputs(double[:] ctrl, dict actuatorsIDs, dict controlInputs):
    cdef str name
    cdef int idx
    for name, value in controlInputs.items():
        idx = actuatorsIDs.get(name, -1)
        if idx >= 0:
            ctrl[idx] = value