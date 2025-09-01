# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
import mujoco

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

def extract_data(np.ndarray[DTYPE_t, ndim=2] xpos,
                 int drone_id,
                 int manip_id,
                 np.ndarray[DTYPE_t, ndim=1] qpos,
                 np.ndarray[DTYPE_t, ndim=1] qvel,
                 np.ndarray[ITYPE_t, ndim=2] joint_indices,
                 np.ndarray[ITYPE_t, ndim=2] velocity_indices,

                 np.ndarray[DTYPE_t, ndim=1] drone_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] manip_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] qpos_sel_out,
                 np.ndarray[DTYPE_t, ndim=1] qvel_sel_out):

    cdef Py_ssize_t i
    cdef int n_joint = joint_indices.shape[0]
    cdef int n_vel = velocity_indices.shape[0]

    # -----------------------------
    # Copy positions using simple memoryviews
    # -----------------------------
    cdef double[:, :] xpos_view = xpos
    for i in range(3):
        drone_pos_out[i] = xpos_view[drone_id, i]
        manip_pos_out[i] = xpos_view[manip_id, i]

    # -----------------------------
    # Select qpos and qvel
    # -----------------------------
    for i in range(n_joint):
        qpos_sel_out[i] = qpos[joint_indices[i]]
    for i in range(n_vel):
        qvel_sel_out[i] = qvel[velocity_indices[i]]


