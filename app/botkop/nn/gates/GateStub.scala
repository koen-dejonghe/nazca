package botkop.nn.gates

sealed trait GateStub {
  def +(other: GateStub): NetworkBuilder = NetworkBuilder(List(this, other))
  def *(i: Int): NetworkBuilder = NetworkBuilder(List.fill(i)(this))
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

case object Output extends GateStub {
  override val category = "output"
}
