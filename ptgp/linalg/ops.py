# Custom PyTensor Ops for lazy linear algebra.
# KernelLinearOp: represents K(X, X) as a LinearOperatorType
# LinearOpMatvec: computes A @ v without materialising A
# LinearOpSolve: CG-based solve, A x = b
# LinearOpLogdet: Lanczos-based log-determinant estimator
