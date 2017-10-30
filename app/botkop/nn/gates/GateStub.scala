package botkop.nn.gates

import akka.actor.ActorSystem

sealed trait GateStub {
  def +(other: GateStub)(implicit system: ActorSystem,
                         projectName: String): Network =
    Network(List(this, other))
  def *(i: Int)(implicit system: ActorSystem, projectName: String): Network =
    Network(List.fill(i)(this))
  def category: String
  def name(layer: Int)(implicit projectName: String) =
    s"${projectName}_$category-$layer"
}

case object Relu extends GateStub {
  override val category = "relu"
}

case object Sigmoid extends GateStub {
  override val category = "sigmoid"
}

case object Linear extends GateStub {
  override val category = "linear"
}

case object Dropout extends GateStub {
  override val category = "dropout"
}

case object BatchNorm extends GateStub {
  override val category = "batchnorm"
}
