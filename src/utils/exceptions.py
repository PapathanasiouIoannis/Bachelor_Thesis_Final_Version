class PhysicsSimulationError(Exception):
    """Base class for all physics simulation errors."""
    pass


class AcausalEosError(PhysicsSimulationError):
    """Raised when the Equation of State violates causality (cs^2 > 1.0 or cs^2 < 0.0)."""
    def __init__(self, cs2_value, eps_c, message=None):
        self.cs2_value = cs2_value
        self.eps_c = eps_c
        self.message = message or f"Acausality detected: cs^2 = {cs2_value} at central density {eps_c}"
        super().__init__(self.message)


class ThermodynamicInstabilityError(PhysicsSimulationError):
    """Raised when the Equation of State violates thermodynamic stability (dP/dEps <= 0)."""
    def __init__(self, deps, dp, message=None):
        self.deps = deps
        self.dp = dp
        self.message = message or f"Thermodynamic instability detected: dEps = {deps}, dP = {dp}"
        super().__init__(self.message)


class TovConvergenceError(PhysicsSimulationError):
    """Raised when the TOV ODE solver fails to converge."""
    def __init__(self, pc, reason, message=None):
        self.pc = pc
        self.reason = reason
        self.message = message or f"TOV solver failed to converge at central pressure {pc}. Reason: {reason}"
        super().__init__(self.message)


class CrustStitchingError(PhysicsSimulationError):
    """Raised when spline interpolation fails between core and crust."""
    def __init__(self, p_trans, message=None):
        self.p_trans = p_trans
        self.message = message or f"Crust stitching failed at transition pressure {p_trans}"
        super().__init__(self.message)


class ConfigurationError(Exception):
    """Base class for configuration errors."""
    pass


class UnknownGenerationModeError(ConfigurationError):
    """Raised when an unknown generation mode is requested."""
    def __init__(self, mode, message=None):
        self.mode = mode
        self.message = message or f"Unknown generation mode: {mode}. Must be 'hadronic' or 'quark'."
        super().__init__(self.message)


class DataLeakageError(Exception):
    """Raised when training data overlaps with validation/test sets in the ML pipeline, or contains invalid physical parameters."""
    pass
