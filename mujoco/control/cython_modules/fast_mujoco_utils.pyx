# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

def extract_data(np.ndarray[DTYPE_t, ndim=2] xpos,
                 int drone_id,
                 int manip_id,
                 np.ndarray[DTYPE_t, ndim=1] qpos,
                 np.ndarray[DTYPE_t, ndim=1] qvel,
                 np.ndarray[ITYPE_t, ndim=2] joint_indices,
                 np.ndarray[ITYPE_t, ndim=2] velocity_indices,
                 np.ndarray[ITYPE_t, ndim=1] drone_hook_ids,
                 np.ndarray[ITYPE_t, ndim=1] arms_hook_ids,

                 np.ndarray[DTYPE_t, ndim=1] drone_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] manip_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] qpos_sel_out,
                 np.ndarray[DTYPE_t, ndim=1] qvel_sel_out,
                 np.ndarray[DTYPE_t, ndim=2] drone_hooks_out,
                 np.ndarray[DTYPE_t, ndim=2] arms_hooks_out):
    """
    Rellena arrays preasignados en vez de crear nuevos.
    """
    cdef Py_ssize_t i

    for i in range(3):
        drone_pos_out[i] = xpos[drone_id, i]
        manip_pos_out[i] = xpos[manip_id, i]

    for i in range(joint_indices.shape[0]):
        qpos_sel_out[i] = qpos[joint_indices[i]]
    for i in range(velocity_indices.shape[0]):
        qvel_sel_out[i] = qvel[velocity_indices[i]]

    for i in range(drone_hook_ids.shape[0]):
        drone_hooks_out[i,0] = xpos[drone_hook_ids[i],0]
        drone_hooks_out[i,1] = xpos[drone_hook_ids[i],1]
        drone_hooks_out[i,2] = xpos[drone_hook_ids[i],2]

    for i in range(arms_hook_ids.shape[0]):
        arms_hooks_out[i,0] = xpos[arms_hook_ids[i],0]
        arms_hooks_out[i,1] = xpos[arms_hook_ids[i],1]
        arms_hooks_out[i,2] = xpos[arms_hook_ids[i],2]


