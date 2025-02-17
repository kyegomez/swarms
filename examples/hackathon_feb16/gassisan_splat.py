import torch
from torch import Tensor
from loguru import logger
from typing import Tuple
import matplotlib.pyplot as plt

try:
    # ipywidgets is available in interactive environments like Jupyter.
    from ipywidgets import interact, IntSlider

    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False
    logger.warning(
        "ipywidgets not installed. Interactive slicing will be disabled."
    )


class GaussianSplat4DStateSpace:
    """
    4D Gaussian splatting with a state space model in PyTorch.

    Each Gaussian is defined by an 8D state vector:
        [x, y, z, w, vx, vy, vz, vw],
    where the first four dimensions are the spatial coordinates and the last
    four are the velocities. Only the spatial (first four) dimensions are used
    for the 4D Gaussian splat, with a corresponding 4×4 covariance matrix.

    Attributes:
        num_gaussians (int): Number of Gaussians.
        state_dim (int): Dimension of the state vector (should be 8).
        states (Tensor): Current state for each Gaussian of shape (num_gaussians, state_dim).
        covariances (Tensor): Covariance matrices for the spatial dimensions, shape (num_gaussians, 4, 4).
        A (Tensor): State transition matrix of shape (state_dim, state_dim).
        dt (float): Time step for state updates.
    """

    def __init__(
        self,
        num_gaussians: int,
        init_states: Tensor,
        init_covariances: Tensor,
        dt: float = 1.0,
    ) -> None:
        """
        Initialize the 4D Gaussian splat model.

        Args:
            num_gaussians (int): Number of Gaussians.
            init_states (Tensor): Initial states of shape (num_gaussians, 8).
                                  Each state is assumed to be
                                  [x, y, z, w, vx, vy, vz, vw].
            init_covariances (Tensor): Initial covariance matrices for the spatial dimensions,
                                       shape (num_gaussians, 4, 4).
            dt (float): Time step for the state update.
        """
        if init_states.shape[1] != 8:
            raise ValueError(
                "init_states should have shape (N, 8) where 8 = 4 position + 4 velocity."
            )
        if init_covariances.shape[1:] != (4, 4):
            raise ValueError(
                "init_covariances should have shape (N, 4, 4)."
            )

        self.num_gaussians = num_gaussians
        self.states = init_states.clone()  # shape: (N, 8)
        self.covariances = (
            init_covariances.clone()
        )  # shape: (N, 4, 4)
        self.dt = dt
        self.state_dim = init_states.shape[1]

        # Create an 8x8 constant-velocity state transition matrix:
        # New position = position + velocity*dt, velocity remains unchanged.
        I4 = torch.eye(
            4, dtype=init_states.dtype, device=init_states.device
        )
        zeros4 = torch.zeros(
            (4, 4), dtype=init_states.dtype, device=init_states.device
        )
        top = torch.cat([I4, dt * I4], dim=1)
        bottom = torch.cat([zeros4, I4], dim=1)
        self.A = torch.cat([top, bottom], dim=0)  # shape: (8, 8)

        logger.info(
            "Initialized 4D GaussianSplatStateSpace with {} Gaussians.",
            num_gaussians,
        )

    def update_states(self) -> None:
        """
        Update the state of each Gaussian using the constant-velocity state space model.

        Applies:
            state_next = A @ state_current.
        """
        self.states = (
            self.A @ self.states.t()
        ).t()  # shape: (num_gaussians, 8)
        logger.debug("States updated: {}", self.states)

    def _compute_gaussian(
        self, pos: Tensor, cov: Tensor, coords: Tensor
    ) -> Tensor:
        """
        Compute the 4D Gaussian function over a grid of coordinates.

        Args:
            pos (Tensor): The center of the Gaussian (4,).
            cov (Tensor): The 4×4 covariance matrix.
            coords (Tensor): A grid of coordinates of shape (..., 4).

        Returns:
            Tensor: Evaluated Gaussian values on the grid with shape equal to coords.shape[:-1].
        """
        try:
            cov_inv = torch.linalg.inv(cov)
        except RuntimeError as e:
            logger.warning(
                "Covariance inversion failed; using pseudo-inverse. Error: {}",
                e,
            )
            cov_inv = torch.linalg.pinv(cov)

        # Broadcast pos over the grid
        diff = coords - pos.view(
            *(1 for _ in range(coords.ndim - 1)), 4
        )
        mahal = torch.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        gaussian = torch.exp(-0.5 * mahal)
        return gaussian

    def render(
        self,
        canvas_size: Tuple[int, int, int, int],
        sigma_scale: float = 1.0,
        normalize: bool = False,
    ) -> Tensor:
        """
        Render the current 4D Gaussian splats onto a 4D canvas.

        Args:
            canvas_size (Tuple[int, int, int, int]): The size of the canvas (d1, d2, d3, d4).
            sigma_scale (float): Scaling factor for the covariance (affects spread).
            normalize (bool): Whether to normalize the final canvas to [0, 1].

        Returns:
            Tensor: A 4D tensor (canvas) with the accumulated contributions from all Gaussians.
        """
        d1, d2, d3, d4 = canvas_size

        # Create coordinate grids for each dimension.
        grid1 = torch.linspace(
            0, d1 - 1, d1, device=self.states.device
        )
        grid2 = torch.linspace(
            0, d2 - 1, d2, device=self.states.device
        )
        grid3 = torch.linspace(
            0, d3 - 1, d3, device=self.states.device
        )
        grid4 = torch.linspace(
            0, d4 - 1, d4, device=self.states.device
        )

        # Create a 4D meshgrid (using indexing "ij")
        grid = torch.stack(
            torch.meshgrid(grid1, grid2, grid3, grid4, indexing="ij"),
            dim=-1,
        )  # shape: (d1, d2, d3, d4, 4)

        # Initialize the canvas.
        canvas = torch.zeros(
            (d1, d2, d3, d4),
            dtype=self.states.dtype,
            device=self.states.device,
        )

        for i in range(self.num_gaussians):
            pos = self.states[i, :4]  # spatial center (4,)
            cov = (
                self.covariances[i] * sigma_scale
            )  # scaled covariance
            gaussian = self._compute_gaussian(pos, cov, grid)
            canvas += gaussian
            logger.debug(
                "Rendered Gaussian {} at position {}", i, pos.tolist()
            )

        if normalize:
            max_val = canvas.max()
            if max_val > 0:
                canvas = canvas / max_val
            logger.debug("Canvas normalized.")

        logger.info("4D Rendering complete.")
        return canvas


def interactive_slice(canvas: Tensor) -> None:
    """
    Display an interactive 2D slice of the 4D canvas using ipywidgets.

    This function fixes two of the four dimensions (d3 and d4) via sliders and
    displays the resulting 2D slice (over dimensions d1 and d2).

    Args:
        canvas (Tensor): A 4D tensor with shape (d1, d2, d3, d4).
    """
    d1, d2, d3, d4 = canvas.shape

    def display_slice(slice_d3: int, slice_d4: int):
        slice_2d = canvas[:, :, slice_d3, slice_d4].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_2d, cmap="hot", origin="lower")
        plt.title(f"2D Slice at d3={slice_d3}, d4={slice_d4}")
        plt.colorbar()
        plt.show()

    interact(
        display_slice,
        slice_d3=IntSlider(min=0, max=d3 - 1, step=1, value=d3 // 2),
        slice_d4=IntSlider(min=0, max=d4 - 1, step=1, value=d4 // 2),
    )


def mip_projection(canvas: Tensor) -> None:
    """
    Render a 2D view of the 4D canvas using maximum intensity projection (MIP)
    along the 3rd and 4th dimensions.

    Args:
        canvas (Tensor): A 4D tensor with shape (d1, d2, d3, d4).
    """
    # MIP along dimension 3
    mip_3d = canvas.max(dim=2)[0]  # shape: (d1, d2, d4)
    # MIP along dimension 4
    mip_2d = mip_3d.max(dim=2)[0]  # shape: (d1, d2)

    plt.figure(figsize=(6, 6))
    plt.imshow(mip_2d.cpu().numpy(), cmap="hot", origin="lower")
    plt.title("2D MIP (Projecting dimensions d3 and d4)")
    plt.colorbar()
    plt.show()


def main() -> None:
    """
    Main function that:
      - Creates a 4D Gaussian splat model.
      - Updates the states to simulate motion.
      - Renders a 4D canvas.
      - Visualizes the 4D volume via interactive slicing (if available) or MIP.
    """
    torch.manual_seed(42)
    num_gaussians = 2

    # Define initial states for each Gaussian:
    # Each state is [x, y, z, w, vx, vy, vz, vw].
    init_states = torch.tensor(
        [
            [10.0, 15.0, 20.0, 25.0, 0.5, -0.2, 0.3, 0.1],
            [30.0, 35.0, 40.0, 45.0, -0.3, 0.4, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )

    # Define initial 4x4 covariance matrices for the spatial dimensions.
    init_covariances = torch.stack(
        [
            torch.diag(
                torch.tensor(
                    [5.0, 5.0, 5.0, 5.0], dtype=torch.float32
                )
            ),
            torch.diag(
                torch.tensor(
                    [3.0, 3.0, 3.0, 3.0], dtype=torch.float32
                )
            ),
        ]
    )

    # Create the 4D Gaussian splat model.
    model = GaussianSplat4DStateSpace(
        num_gaussians, init_states, init_covariances, dt=1.0
    )

    # Update states to simulate one time step.
    model.update_states()

    # Render the 4D canvas.
    canvas_size = (20, 20, 20, 20)
    canvas = model.render(
        canvas_size, sigma_scale=1.0, normalize=True
    )

    # Visualize the 4D data.
    if HAS_IPYWIDGETS:
        logger.info("Launching interactive slicing tool for 4D data.")
        interactive_slice(canvas)
    else:
        logger.info(
            "ipywidgets not available; using maximum intensity projection instead."
        )
        mip_projection(canvas)


if __name__ == "__main__":
    main()
