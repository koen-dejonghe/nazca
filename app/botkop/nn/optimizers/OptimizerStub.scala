package botkop.nn.optimizers

sealed trait OptimizerStub

case object Adam extends OptimizerStub
case object GradientDescent extends OptimizerStub
case object Momentum extends OptimizerStub
case object Nesterov extends OptimizerStub
