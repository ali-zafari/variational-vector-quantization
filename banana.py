import math
import torch
import torch.distributions as D
import matplotlib.pyplot as plt

__all__ = ['BananaDistribution', ]


def rotate(x, angle):
    """Rotate a 2D tensor by a certain angle (in degrees)."""
    angle = torch.as_tensor([angle * math.pi / 180])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot_mat = torch.as_tensor([[cos, sin], [-sin, cos]])
    return x @ rot_mat


# Borrowed from Yann Dubois
class BananaTransform(D.Transform):
    """Transform from gaussian to banana."""

    def __init__(self, curvature, factor=10):
        super().__init__()
        self.bijective = True
        self.curvature = curvature
        self.factor = factor
        self.domain = D.constraints.Constraint()
        self.codomain = D.constraints.Constraint()

    def _call(self, x):
        shift = torch.zeros_like(x)
        shift[..., 1] = self.curvature * (torch.pow(x[..., 0], 2) - self.factor ** 2)
        return x + shift

    def _inverse(self, y):
        shift = torch.zeros_like(y)
        shift[..., 1] = self.curvature * (torch.pow(y[..., 0], 2) - self.factor ** 2)
        return y - shift

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class RotateTransform(D.Transform):
    """Rotate a distribution from `angle` degrees."""

    def __init__(self, angle):
        super().__init__()
        self.bijective = True
        self.angle = angle
        self.domain = D.constraints.Constraint()
        self.codomain = D.constraints.Constraint()

    def _call(self, x):
        return rotate(x, self.angle)

    def _inverse(self, y):
        return rotate(y, -self.angle)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class BananaDistribution(D.TransformedDistribution):
    """2D banana distribution.

    Parameters
    ----------
    curvature : float, optional
        Controls the strength of the curvature of the banana-shape.

    factor : float, optional
        Controls the elongation of the banana-shape.

    location : torch.Tensor, optional
        Controls the location of the banana-shape.

    angles : float, optional
        Controls the angle rotation of the banana-shape.

    scale : float, optional
        Rescales the entire distribution (while keeping entropy of underlying distribution correct)
        This is useful to make sure that the inputs during training are not too large / small.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        curvature=0.05,
        factor=6,
        location=torch.as_tensor([-1.5, -2.0]),
        angle=-40,
        scale=1 / 2,
    ):
        std = torch.as_tensor([factor * scale, scale])
        base_dist = D.Independent(D.Normal(loc=torch.zeros(2), scale=std), 1)

        transforms = D.ComposeTransform(
            [
                BananaTransform(curvature / scale, factor=factor * scale),
                RotateTransform(angle),
                D.AffineTransform(location * scale, 1),
            ]
        )
        super().__init__(base_dist, transforms)

        self.curvature = curvature
        self.factor = factor
        self.rotate = rotate
        self.domain = D.constraints.Constraint()

    def entropy(self):
        return self.base_dist.entropy()  # log det is zero => same entropy


if __name__ == "__main__":

    laplace = D.Laplace(loc=0, scale=1)
    banana = BananaDistribution()

    laplace_data = laplace.sample((200, ))
    banana_data = banana.sample((1000, ))

    laplace_limit = torch.linspace(-5, 5, 1000)
    banana_limit_1 = torch.linspace(-5, 6, 1000)
    banana_limit_2 = torch.linspace(-5, 6, 1000)
    banana_grid_1, banana_grid_2 = torch.meshgrid(banana_limit_1, banana_limit_2)
    xy = torch.vstack((banana_grid_1.flatten(), banana_grid_2.flatten())).T

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(laplace_limit, torch.exp(laplace.log_prob(laplace_limit)), label="density")
    # ax[0].hist(laplace_data, bins=100, density=True, color='gray')
    ax[0].eventplot(laplace_data, lineoffsets=0, linelengths=0.03, color='red', alpha=.3, label="samples")
    ax[0].set_ylim(0, 0.55)
    ax[0].set_xlim(-5, 5)
    ax[0].legend()

    ax[1].contourf(banana_grid_1, banana_grid_2, torch.exp(banana.log_prob(xy).view(1000, 1000)),
                   levels=100, label="density", cmap='gist_yarg')
    ax[1].scatter(banana_data[:, 0], banana_data[:, 1], color='red', s=1, alpha=.3, label="samples")
    ax[1].set_ylim(-5, 6)
    ax[1].set_xlim(-5, 6)
    ax[1].legend()

    plt.tight_layout()
    plt.legend()
    plt.show()
