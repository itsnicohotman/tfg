## distutils: language = c++
## cython: boundscheck=False
## cython: wraparound=False
## cython: nonecheck=False
## cython: language_level=3
#
#import numpy as np
#cimport numpy as np
#import mujoco
#
#ctypedef np.float64_t DTYPE_t
#ctypedef np.int32_t ITYPE_t
#
#def extract_data(np.ndarray[DTYPE_t, ndim=2] xpos,
#                 int drone_id,
#                 int manip_id,
#                 np.ndarray[DTYPE_t, ndim=1] qpos,
#                 np.ndarray[DTYPE_t, ndim=1] qvel,
#                 np.ndarray[ITYPE_t, ndim=2] joint_indices,
#                 np.ndarray[ITYPE_t, ndim=2] velocity_indices,
#                 np.ndarray[ITYPE_t, ndim=1] drone_hook_ids,
#                 np.ndarray[ITYPE_t, ndim=1] arms_hook_ids,
#
#                 np.ndarray[DTYPE_t, ndim=1] drone_pos_out,
#                 np.ndarray[DTYPE_t, ndim=1] manip_pos_out,
#                 np.ndarray[DTYPE_t, ndim=1] qpos_sel_out,
#                 np.ndarray[DTYPE_t, ndim=1] qvel_sel_out,
#                 np.ndarray[DTYPE_t, ndim=2] drone_hooks_out,
#                 np.ndarray[DTYPE_t, ndim=2] arms_hooks_out,
#
#                 object model,
#                 object data,
#                 np.ndarray[DTYPE_t, ndim=1] manip_com_pos_out,   # shape (3,)
#                 np.ndarray[DTYPE_t, ndim=1] manip_com_vel_out,   # shape (3,)
#                 np.ndarray[DTYPE_t, ndim=1] manip_com_acc_out,   # shape (3,)
#                 np.ndarray[DTYPE_t, ndim=2] manip_com_jac_out):    # shape (3, nv)
#    """
#    Rellena arrays preasignados en vez de crear nuevos.
#    """
#    cdef Py_ssize_t i
#
#    for i in range(3):
#        drone_pos_out[i] = xpos[drone_id, i]
#        manip_pos_out[i] = xpos[manip_id, i]
#
#    for i in range(joint_indices.shape[0]):
#        qpos_sel_out[i] = qpos[joint_indices[i]]
#    for i in range(velocity_indices.shape[0]):
#        qvel_sel_out[i] = qvel[velocity_indices[i]]
#
#    for i in range(drone_hook_ids.shape[0]):
#        drone_hooks_out[i,0] = xpos[drone_hook_ids[i],0]
#        drone_hooks_out[i,1] = xpos[drone_hook_ids[i],1]
#        drone_hooks_out[i,2] = xpos[drone_hook_ids[i],2]
#
#    for i in range(arms_hook_ids.shape[0]):
#        arms_hooks_out[i,0] = xpos[arms_hook_ids[i],0]
#        arms_hooks_out[i,1] = xpos[arms_hook_ids[i],1]
#        arms_hooks_out[i,2] = xpos[arms_hook_ids[i],2]
#
#    # ------ CoM subtree data ------ #
#
#    mujoco.mj_kinematics(model, data)
#    mujoco.mj_comPos(model, data)
#    mujoco.mj_comVel(model, data)
#
#    # dimensiones del modelo
#    cdef int nbody = model.nbody
#    cdef int nv = model.nv
#
#    cdef np.ndarray[DTYPE_t, ndim=2] subtree_com = np.asarray(data.subtree_com, dtype=np.float64)
#
#    # posición del CoM de la subtree (en world)
#    manip_com_pos_out[0] = subtree_com[manip_id, 0]
#    manip_com_pos_out[1] = subtree_com[manip_id, 1]
#    manip_com_pos_out[2] = subtree_com[manip_id, 2]
#
#    # velocidad lineal del CoM de la subtree (data.subtree_linvel)
#    cdef np.ndarray[DTYPE_t, ndim=2] subtree_linvel = np.asarray(data.subtree_linvel, dtype=np.float64)
#
#    manip_com_vel_out[:] = subtree_linvel[manip_id]
#
#    # Jacobian
#    cdef np.ndarray[DTYPE_t, ndim=2] J = np.zeros((3, nv), dtype=np.float64)
#    mujoco.mj_jacSubtreeCom(model, data, J, manip_id)
#    if manip_com_jac_out.shape[0] != 3 or manip_com_jac_out.shape[1] != nv:
#        raise ValueError("manip_com_jac_out debe tener shape (3, nv) con nv = model.nv")
#    manip_com_jac_out[:, :] = J[:, :]
#
#    # Jacobian dot
#    cdef np.ndarray[DTYPE_t, ndim=2] xipos = np.asarray(data.xipos, dtype=np.float64)
#    cdef np.ndarray[DTYPE_t, ndim=1] point_in_body = np.empty(3, dtype=np.float64)
#    point_in_body[:] = manip_com_pos_out - xipos[manip_id]
#
#    cdef np.ndarray[DTYPE_t, ndim=2] Jdot_p = np.zeros((3, nv), dtype=np.float64)
#    cdef np.ndarray[DTYPE_t, ndim=2] Jdot_r = np.zeros((3, nv), dtype=np.float64)   # jacr no lo necesitamos, pero hay que pasarlo
#
#    mujoco.mj_jacDot(model, data,
#                    Jdot_p, Jdot_r,
#                    point_in_body, manip_id)
#
#    # --- Aceleración lineal del CoM: a = J*qdd + Jdot*qdot ---
#    cdef np.ndarray[DTYPE_t, ndim=1]  qacc = np.asarray(data.qacc, dtype=np.float64)[:nv]
#    if qacc.shape[0] < nv:
#        raise RuntimeError("data.qacc no tiene la longitud esperada (nv). Rellena data.qacc antes.")
#    cdef np.ndarray[DTYPE_t, ndim=1] qvel_full = np.asarray(qvel, dtype=np.float64)[:nv]
#    if qvel_full.shape[0] < nv:
#        raise RuntimeError("qvel pasado a extract_data debe tener al menos nv elementos (model.nv).")
#
#    cdef np.ndarray[DTYPE_t, ndim=1] com_acc = J.dot(qacc[:nv]) + Jdot_p.dot(qvel_full[:nv])
#    manip_com_acc_out[:] = com_acc
#

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
                 np.ndarray[ITYPE_t, ndim=1] drone_hook_ids,
                 np.ndarray[ITYPE_t, ndim=1] arms_hook_ids,

                 np.ndarray[DTYPE_t, ndim=1] drone_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] manip_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] qpos_sel_out,
                 np.ndarray[DTYPE_t, ndim=1] qvel_sel_out,
                 np.ndarray[DTYPE_t, ndim=2] drone_hooks_out,
                 np.ndarray[DTYPE_t, ndim=2] arms_hooks_out,):

    cdef Py_ssize_t i
    cdef int n_joint = joint_indices.shape[0]
    cdef int n_vel = velocity_indices.shape[0]
    cdef int n_drone_hooks = drone_hook_ids.shape[0]
    cdef int n_arms_hooks = arms_hook_ids.shape[0]

    # -----------------------------
    # Copiar posiciones simples usando memoryviews
    # -----------------------------
    cdef double[:, :] xpos_view = xpos
    for i in range(3):
        drone_pos_out[i] = xpos_view[drone_id, i]
        manip_pos_out[i] = xpos_view[manip_id, i]

    # -----------------------------
    # Selección de qpos y qvel
    # -----------------------------
    for i in range(n_joint):
        qpos_sel_out[i] = qpos[joint_indices[i]]
    for i in range(n_vel):
        qvel_sel_out[i] = qvel[velocity_indices[i]]

    # -----------------------------
    # Hooks
    # -----------------------------
    for i in range(n_drone_hooks):
        drone_hooks_out[i, 0] = xpos_view[drone_hook_ids[i], 0]
        drone_hooks_out[i, 1] = xpos_view[drone_hook_ids[i], 1]
        drone_hooks_out[i, 2] = xpos_view[drone_hook_ids[i], 2]

    for i in range(n_arms_hooks):
        arms_hooks_out[i, 0] = xpos_view[arms_hook_ids[i], 0]
        arms_hooks_out[i, 1] = xpos_view[arms_hook_ids[i], 1]
        arms_hooks_out[i, 2] = xpos_view[arms_hook_ids[i], 2]



def compute_com_jac_and_accel(
                 object model,
                 object data,
                 int manip_id,
                 np.ndarray[DTYPE_t, ndim=1] manip_com_pos_out,
                 np.ndarray[DTYPE_t, ndim=1] manip_com_vel_out,
                 np.ndarray[DTYPE_t, ndim=1] manip_com_acc_out,
                 np.ndarray[DTYPE_t, ndim=2] manip_com_jac_out):

    cdef int nv = model.nv
    cdef int nbody = model.nbody

    # -----------------------------
    # Acceso a CoM de la subtree usando punteros/memoryviews
    # -----------------------------
    cdef double[:, :] subtree_com = data.subtree_com
    cdef double[:, :] subtree_linvel = data.subtree_linvel
    cdef double[:, :] xipos = data.xipos

    for i in range(3):
        manip_com_pos_out[i] = subtree_com[manip_id, i]
        manip_com_vel_out[i] = subtree_linvel[manip_id, i]

    # -----------------------------
    # Jacobiano lineal
    # -----------------------------
    cdef np.ndarray[DTYPE_t, ndim=2] J = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacSubtreeCom(model, data, J, manip_id)
    manip_com_jac_out[:, :] = J[:, :]

    # -----------------------------
    # Derivada temporal del Jacobiano
    # -----------------------------
    cdef np.ndarray[DTYPE_t, ndim=2] Jdot = np.zeros((3, nv), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] Jdot_r = np.zeros((3, nv), dtype=np.float64)

    cdef double[3] point_in_body
    for i in range(3):
        point_in_body[i] = manip_com_pos_out[i] - xipos[manip_id, i]

    mujoco.mj_jacDot(model, data, Jdot, Jdot_r, point_in_body, manip_id)

    # -----------------------------
    # Aceleración lineal: a = J*qdd + Jdot*qvel
    # -----------------------------
    cdef double[:] qacc = data.qacc
    cdef double[:] qvel_full = data.qvel

    for i in range(3):
        manip_com_acc_out[i] = 0.0
        for j in range(nv):
            manip_com_acc_out[i] += J[i, j]*qacc[j] + Jdot[i, j]*qvel_full[j]

