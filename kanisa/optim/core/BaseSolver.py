class BaseSolver:
    """
    KANISA Base Solver Template
    ---------------------------
    This class defines the standard interface for all optimisation solvers in KANISA.

    Key Features:
    - __init__ : Accepts solver hyperparameters only.
    - REQUIRED : List of optimisation problem fields that MUST be provided.
    - build()  : Validates the optimisation problem and displays a summary.
    - compute(): Runs the optimisation algorithm (child classes override logic).
    - check_constraints(): Standard constraint handler for inequality and equality constraints.
    """

    # Default hyperparameters for all solvers
    DEFAULTS = {
        "max_iter": 200,
        "max_stall_iter": float("inf"), #No stalling iteration set
        "verbosity": 1,
        "seed": None,
    }

    # Required optimisation fields (can be extended in subclass)
    REQUIRED = ["objective", "bounds"]

    def __init__(self, hyper_params=None):
        """
        Constructor: Store solver hyperparameters.
        Does not accept the objective or constraints.
        """
        hyper_params = hyper_params or {}

        # Merge default parameters with user input
        self.params = {**self.DEFAULTS, **hyper_params}

        # Extract commonly used parameters
        self.max_iter  = self.params["max_iter"]
        self.verbosity = self.params["verbosity"]
        self.seed      = self.params["seed"]

        # These attributes will be set during build()
        self.objective = None
        self.bounds = None
        self.dim = None
        self.ineq_constraints = []
        self.eq_constraints = []

    # -----------------------------------------------------------
    # Constraint Checking
    # -----------------------------------------------------------
    def check_constraints(self, x):
        """Return True if x satisfies all constraints."""

        # Inequality constraints: g(x) <= 0
        for g in self.ineq_constraints:
            if g(x) > 0:
                return False

        # Equality constraints: h(x) = 0
        for h in self.eq_constraints:
            if abs(h(x)) > 1e-6:
                return False

        return True

    # -----------------------------------------------------------
    # Build Function: Validates the optimisation problem
    # -----------------------------------------------------------
    def build(self, objective, bounds,
              ineq_constraints=None, eq_constraints=None):
        """
        Load and validate the optimisation problem.
        Returns True if the problem is compliant; False otherwise.

        Parameters
        ----------
        objective : callable
            Objective function f(x).

        bounds : list of (lb, ub)
            Box constraints for each decision variable.

        ineq_constraints : list of callable
            Functions g_i(x) <= 0.

        eq_constraints : list of callable
            Functions h_j(x) = 0.
        """

        # Consolidate problem data for REQUIRED check
        problem_data = {
            "objective":        objective,
            "bounds":           bounds,
            "ineq_constraints": ineq_constraints or [],
            "eq_constraints":   eq_constraints or [],
        }

        # -------------------------------------------------------
        # REQUIRED FIELD VALIDATION
        # -------------------------------------------------------
        missing = [field for field in self.REQUIRED
                   if field not in problem_data or problem_data[field] is None]

        if missing:
            print(f"[ERROR] Missing required optimisation fields: {missing}")
            return False

        # -------------------------------------------------------
        # Store problem specification
        # -------------------------------------------------------
        self.objective        = objective
        self.bounds           = bounds
        self.ineq_constraints = ineq_constraints or []
        self.eq_constraints   = eq_constraints or []
        self.dim              = len(bounds)

        # -------------------------------------------------------
        # Compliance Checks
        # -------------------------------------------------------

        # Check objective
        if not callable(self.objective):
            print("[ERROR] Objective function must be callable.")
            return False

        # Check bounds format
        if not isinstance(self.bounds, list):
            print("[ERROR] Bounds must be a list of (lb, ub) pairs.")
            return False

        for pair in self.bounds:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                print("[ERROR] Each bound must be a tuple/list of (lb, ub).")
                return False

        # Check inequality constraints
        for g in self.ineq_constraints:
            if not callable(g):
                print("[ERROR] One inequality constraint is not callable.")
                return False

        # Check equality constraints
        for h in self.eq_constraints:
            if not callable(h):
                print("[ERROR] One equality constraint is not callable.")
                return False

        # -------------------------------------------------------
        # Display Summary (if verbosity enabled)
        # -------------------------------------------------------
        if self.verbosity >= 1:
            print("\n========= KANISA Build Summary =========")
            print(f"Problem Dimension:         {self.dim}")
            print(f"Max Iterations:            {self.max_iter}")
            print(f"Inequality Constraints:    {len(self.ineq_constraints)}")
            print(f"Equality Constraints:      {len(self.eq_constraints)}")
            print(f"Bounds:                    {self.bounds}")
            print("========================================\n")

        return True

    # -----------------------------------------------------------
    # Compute Function: Entry point for running the solver
    # -----------------------------------------------------------
    def compute(self, objective=None, bounds=None,
                ineq_constraints=None, eq_constraints=None):
        """
        Main function to run the optimisation algorithm.

        If objective/bounds/constraints are passed,
        build() is invoked automatically.

        Otherwise, the solver uses the previously built setup.
        """

        # If the user passed a new optimisation problem, rebuild it
        if objective is not None:
            ok = self.build(objective, bounds,
                            ineq_constraints, eq_constraints)
            if not ok:
                raise ValueError("Build failed: invalid problem specification.")

        # Ensure build() has been executed previously
        if self.objective is None:
            raise ValueError(
                "No optimisation problem loaded.\n"
                "Call solver.build() or provide data to solver.compute()."
            )

        # -------------------------------------------------------
        # Must be implemented by each solver (PSO, GA, SA, etc.)
        # -------------------------------------------------------
        raise NotImplementedError(
            "Child solver classes must override compute() with algorithm logic."
        )
