package botkop.nn.gates

import akka.actor.{ActorRef, ActorSystem}
import botkop.nn.costs.{Cost, CrossEntropy}
import botkop.nn.optimizers._

case class Network(config: List[GateConfig]) {
  def materialize(implicit system: ActorSystem,
                  projectName: String): List[ActorRef] =
    config.zipWithIndex.reverse.foldLeft(List.empty[ActorRef]) {
      case (gs, (cfg, index)) =>
        if (gs.isEmpty)
          cfg.materialize(None, index) :: gs
        else
          cfg.materialize(Some(gs.head), index) :: gs
    }
}

case class NetworkBuilder(gates: List[GateStub] = List.empty,
                          dimensions: List[Int] = List.empty,
                          cost: Cost = CrossEntropy,
                          optimizer: OptimizerStub = GradientDescent,
                          learningRate: Double = 0.001,
                          learningRateDecay: Double = 0.95,
                          regularization: Double = 0.0) {

  def +(other: NetworkBuilder) = NetworkBuilder(this.gates ++ other.gates)
  def +(layer: GateStub) = NetworkBuilder(this.gates :+ layer)
  def *(i: Int): NetworkBuilder =
    NetworkBuilder(List.tabulate(i)(_ => gates).flatten)

  def withGates(gs: GateStub*): NetworkBuilder = copy(gates = gs.toList)
  def withDimensions(dims: Int*): NetworkBuilder = copy(dimensions = dims.toList)
  def withCostFunction(cf: Cost): NetworkBuilder = copy(cost = cf)
  def withOptimizer(o: OptimizerStub): NetworkBuilder = copy(optimizer = o)
  def withLearningRate(d: Double): NetworkBuilder = copy(learningRate = d)
  def withLearningRateDecay(d: Double): NetworkBuilder = copy(learningRateDecay = d)
  def withRegularization(reg: Double): NetworkBuilder = copy(regularization = reg)

  def build(implicit system: ActorSystem, projectName: String): Network = {
    require(gates.nonEmpty)
    require(dimensions.nonEmpty)
    val numLinearGates = gates.count(g => g == Linear)
    require(numLinearGates == dimensions.length - 1)

    val op = optimizer match {
      case Adam =>
        AdamOptimizer(learningRate, learningRateDecay)
      case GradientDescent =>
        GradientDescentOptimizer(learningRate, learningRateDecay)
      case Momentum =>
        MomentumOptimizer(learningRate, learningRateDecay)
      case Nesterov =>
        NesterovOptimizer(learningRate, learningRateDecay)
    }

    val configs = gates.zipWithIndex.map { case (g, i) =>
      g match {
        case BatchNorm =>
          val shape = dimensions.slice(i, i + 1)
          BatchNormConfig(shape)
        case Dropout =>
          DropoutConfig()
        case Linear =>
          val shape = dimensions.slice(i, i + 1)
          LinearConfig(shape, regularization, op)
        case Relu =>
          ReluConfig
        case Sigmoid =>
          SigmoidConfig
      }
    }

    val output  = OutputConfig(cost)

    Network(configs :+ output)
  }

}

/*
case class Network(
    gates: List[GateStub] = List.empty,
    dimensions: Array[Int] = Array.empty,
    // costFunction: CostFunction = crossEntropyCost,
    cost: Cost = CrossEntropy,
    optimizer: () => Optimizer = () => GradientDescent(learningRate = 0.01),
    regularization: Double = 0.0,
    dropout: Double = 0.5,
    actors: List[ActorRef] = List.empty)(implicit projectName: String) {

  def +(other: Network) = Network(this.gates ++ other.gates)
  def +(layer: GateStub) = Network(this.gates :+ layer)
  def *(i: Int): Network = Network(List.tabulate(i)(_ => gates).flatten)

  def withGates(gs: GateStub*): Network = copy(gates = gs.toList)
  def withDimensions(dims: Int*): Network = copy(dimensions = dims.toArray)
  // def withCostFunction(cf: CostFunction): Network = copy(costFunction = cf)
  def withCostFunction(cf: Cost): Network = copy(cost = cf)
  def withOptimizer(o: => Optimizer): Network = copy(optimizer = () => o)
  def withRegularization(reg: Double): Network = copy(regularization = reg)
  def withDropout(p: Double): Network = copy(dropout = dropout)

  def entryGate: Option[ActorRef] = actors.headOption

  def quit(): Unit = actors.foreach(_ ! PoisonPill)

  def build(implicit system: ActorSystem): Network = {
    require(gates.nonEmpty)
    require(dimensions.nonEmpty)
    require(actors.isEmpty)
    val numLinearGates = gates.count(g => g == Linear)
    require(numLinearGates == dimensions.length - 1)

    val output =
      system.actorOf(OutputGate.props(OutputConfig(cost)), "output")
    // system.actorOf(OutputGate.props(costFunction), "output")

    val nn = build(gates.reverse, dimensions.length, List(output))

    copy(actors = nn)
  }

  @tailrec
  private def build(gates: List[GateStub], i: Int, network: List[ActorRef])(
      implicit system: ActorSystem): List[ActorRef] = gates match {

    case Nil => network

    case g :: gs =>
      g match {

        case Relu =>
          val gate =
            system.actorOf(
              ReluGate.props(network.head).withDispatcher("gate-dispatcher"),
              Relu.name(i))
          build(gs, i, gate :: network)

        case Sigmoid =>
          val gate =
            system.actorOf(
              SigmoidGate.props(network.head).withDispatcher("gate-dispatcher"),
              Sigmoid.name(i))
          build(gs, i, gate :: network)

        case Dropout =>
          val config = DropoutConfig(dropout)
          val gate =
            system.actorOf(DropoutGate
                             .props(network.head, config)
                             .withDispatcher("gate-dispatcher"),
                           Dropout.name(i))
          build(gs, i, gate :: network)

        case Linear =>
          val shape = dimensions.slice(i - 2, i).reverse
          val config = LinearConfig(shape.toList, regularization, optimizer())
          val gate =
            system.actorOf(LinearGate
                             .props(network.head, config)
                             .withDispatcher("gate-dispatcher"),
                           Linear.name(i - 1))
          build(gs, i - 1, gate :: network)

        case BatchNorm =>
          val shape = dimensions.slice(i - 2, i).reverse
          val config = BatchNormConfig(shape.toList)
          val gate = system.actorOf(BatchNormGate
                                      .props(network.head, config)
                                      .withDispatcher("gate-dispatcher"),
                                    BatchNorm.name(i))
          build(gs, i, gate :: network)
      }
  }
}

object Network {

  implicit val networkWrites: Writes[Network] = (network: Network) =>
    Json.obj(
      "gates" -> network.gates.map(_.category),
      "dimensions" -> network.dimensions
  )
}
 */
