---
orphan: true
---

(gallery-tags)=
# Kernel property tags

The covariance gallery labels each kernel with the properties below. Follow a tag from any kernel page to its definition here.

(stationary)=
## Stationary

The covariance $k(x, y)$ depends only on the separation $x - y$, so it is invariant to shifting both inputs by the same amount. The prior variance $k(x, x)$ is then constant across the whole input space.

(non-stationary)=
## Non-stationary

The covariance depends on where the inputs sit, not just on their separation. The prior variance and correlation length can vary across the input space.

(isotropic)=
## Isotropic

The covariance depends only on the distance $\lVert x - y \rVert$ between inputs, so it is invariant to rotation as well as translation. A single lengthscale controls correlation in every direction.

(smooth)=
## Smooth

Sample functions are several times mean-square differentiable, giving very regular draws. The exponentiated quadratic is infinitely differentiable; the Matérn 5/2 kernel is twice differentiable.

(rough)=
## Rough

Sample functions are at most once mean-square differentiable, giving visibly jagged draws. Examples are the Matérn 3/2 kernel and the random walk.

(very-rough)=
## Very Rough

Sample functions are continuous but nowhere differentiable, as with the exponential (Matérn 1/2) kernel.

(one-d)=
## 1-D

Defined for a single input dimension only.

(universal)=
## Universal

The kernel's reproducing-kernel Hilbert space is dense in the continuous functions, so it can approximate any continuous target arbitrarily well. The exponentiated quadratic is the canonical example.
