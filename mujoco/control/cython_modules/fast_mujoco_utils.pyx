# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

def extract_data(np.ndarray[DTYPE_t, ndim=2] xpos,           # (num_bodies, 3)
                 int drone_id,
                 int manip_id,
                 np.ndarray[DTYPE_t, ndim=1] qpos,           # (num_joints, 1)
                 np.ndarray[DTYPE_t, ndim=1] qvel,           # (num_joints, 1)
                 np.ndarray[ITYPE_t, ndim=2] joint_indices,  # (nj,)
                 np.ndarray[ITYPE_t, ndim=2] velocity_indices, # (nv,)
                 double current_time,):
    """
    Extrae datos optimizados desde arrays ya separados de MuJoCo.
    """

    cdef:
        DTYPE_t[3] drone_pos
        DTYPE_t[3] manip_pos
        Py_ssize_t i
        Py_ssize_t nj = joint_indices.shape[0]
        Py_ssize_t nv = velocity_indices.shape[0]
        np.ndarray[DTYPE_t, ndim=1] qpos_sel = np.empty(nj, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] qvel_sel = np.empty(nv, dtype=np.float64)

    # Extraer posiciones globales
    for i in range(3):
        drone_pos[i] = xpos[drone_id, i]
        manip_pos[i] = xpos[manip_id, i]

    # Extraer qpos y qvel teniendo en cuenta que son 2D (acceso a columna 0)
    for i in range(nj):
        qpos_sel[i] = qpos[joint_indices[i]]
    for i in range(nv):
        qvel_sel[i] = qvel[velocity_indices[i]]

    # Devolver datos como arrays independientes
    return (np.array(drone_pos, copy=True),
            np.array(manip_pos, copy=True),
            qpos_sel,
            qvel_sel,
            current_time)
