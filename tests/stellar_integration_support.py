import numpy as np

from src.config import CONFIG


class RecordingEos:
    def __init__(self, *, eps_surf=0.0):
        self.pressures = []
        self.eps_surf = eps_surf

    def __call__(self, pressure):
        self.pressures.append(float(pressure))
        return float(250.0 + pressure), 0.42


class SyntheticSolution:
    def __init__(
        self,
        *,
        initial_state,
        mass=1.0,
        radius=12.0,
        y_surface=0.8,
        surfaced=True,
        status=None,
    ):
        self.status = (1 if surfaced else 0) if status is None else status
        self.mass = mass
        self.radius = radius
        self.y_surface = y_surface
        self.initial_state = initial_state
        self.sampled_radii = None
        if surfaced:
            self.t_events = [np.array([radius])]
            self.y_events = [
                np.array(
                    [
                        [
                            mass,
                            CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"],
                            y_surface,
                        ]
                    ]
                )
            ]
        else:
            self.t_events = [np.array([])]
            self.y_events = [np.empty((0, 3))]

    def sol(self, radii):
        self.sampled_radii = np.asarray(radii)
        sample_count = len(self.sampled_radii)
        return np.vstack(
            (
                np.linspace(self.initial_state[0], self.mass, sample_count),
                np.linspace(
                    self.initial_state[1],
                    CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"],
                    sample_count,
                ),
                np.linspace(self.initial_state[2], self.y_surface, sample_count),
            )
        )


def capture_log(records):
    def capture(*args, **kwargs):
        records.append((args, kwargs))

    return capture
