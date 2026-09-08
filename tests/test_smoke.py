def test_imports_and_pins():
    import crocoddyl
    import mujoco
    import numpy
    import pinocchio

    assert crocoddyl.__version__ == "3.2.1"
    assert pinocchio.__version__ == "4.0.0"
    assert tuple(int(x) for x in mujoco.__version__.split(".")[:2]) >= (3, 12)


def test_crocoddyl_pinocchio_abi_compatible():
    """pin 4.1.0 + crocoddyl 3.2.1 segfaults on the first calc(); this guards against that."""
    import numpy as np
    import pinocchio as pin
    import crocoddyl as croc

    model = pin.buildSampleModelHumanoid()
    state = croc.StateMultibody(model)
    actuation = croc.ActuationModelFloatingBase(state)
    nu = actuation.nu
    x0 = np.concatenate([pin.neutral(model), np.zeros(model.nv)])
    costs = croc.CostModelSum(state, nu)
    costs.addCost("xreg", croc.CostModelResidual(state, croc.ResidualModelState(state, x0, nu)), 1.0)
    contacts = croc.ContactModelMultiple(state, nu)
    dam = croc.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs)
    data = dam.createData()
    dam.calc(data, x0, np.zeros(nu))
    assert np.all(np.isfinite(data.xout))
