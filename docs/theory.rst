Background: Koopman Operator Theory
====================================

This page gives a short conceptual overview of the theory behind KKF. For the
full algorithm, see the *Notes* section of
:func:`kkf.filter.apply_koopman_kalman_filter`.

The problem
-----------

The Kalman filter is optimal for linear systems

.. math::

   x_{k+1} = A x_k + w_k, \qquad y_k = C x_k + v_k

but degrades on nonlinear dynamics, where no single linear operator describes
the state evolution exactly.

Lifting to a linear representation
-----------------------------------

The Koopman operator :math:`\mathcal{K}` acts linearly on *observables*
(functions of the state) rather than on the state itself:

.. math::

   (\mathcal{K}\phi)(x) = \phi(f(x))

for any observable :math:`\phi`. If we choose a finite dictionary of
observables :math:`\phi_1, \dots, \phi_m` (the *feature map*
:math:`\phi(x) = [\phi_1(x), \dots, \phi_m(x)]^\top`), the dynamics of
:math:`\phi(x_k)` are approximately linear:

.. math::

   \phi(x_{k+1}) \approx U \, \phi(x_k)

Kernel EDMD (kEDMD) estimates the finite-dimensional approximation
:math:`U` of :math:`\mathcal{K}` from sampled state transitions, using a
kernel (RBF, Matérn, ...) to implicitly define a rich, possibly
infinite-dimensional dictionary via the kernel trick — this is what
:class:`kkf.koopman.KoopmanOperator` computes.

Filtering in feature space
---------------------------

Once the lifted dynamics are (approximately) linear, a standard Kalman
filter can run directly in feature space :math:`z_k = \phi(x_k)`:
predict with :math:`U`, update with the observation model, and map the
estimate back to state space. This is exactly the three-phase procedure
implemented in :func:`kkf.filter.apply_koopman_kalman_filter`
(initialization, prediction, update).

Further reading
----------------

- Koopman, B. O. (1931). *Hamiltonian systems and transformation in Hilbert space*.
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). *A data-driven
  approximation of the Koopman operator: Extended dynamic mode decomposition*.
- Klus, S., et al. (2020). *Kernel-based approximation of the Koopman
  generator and Schrödinger operator*.
