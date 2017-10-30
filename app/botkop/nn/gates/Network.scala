package botkop.nn.gates

import akka.actor.{ActorRef, ActorSystem, PoisonPill}
import botkop.nn.costs._
import botkop.nn.optimizers.{GradientDescent, Optimizer}
import play.api.libs.json.{JsObject, Json, Writes}

import scala.annotation.tailrec

case class Network(gates: List[GateConfig]) {

  def materialize(implicit system: ActorSystem,
                  projectName: String): List[ActorRef] = {

    val last = gates.last.materialize(None, )


    // val outputConfig = gates.last.asInstanceOf[OutputConfig]
    // val outputGate = system.actorOf(OutputGate.props(outputConfig), "output")

    // build(gates.reverse, gates.length, List(outputGate))

    @tailrec
    def build(gates: List[GateConfig],
              i: Int,
              network: List[ActorRef]): List[ActorRef] = gates match {
      case Nil => network
      case gc :: gcs =>
        gc match {

          case c: BatchNormConfig =>
            val props = BatchNormGate
              .props(network.head, c)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, BatchNorm.name(i))
            build(gcs, i - 1, gate :: network)

          case c: DropoutConfig =>
            val props = DropoutGate
              .props(network.head, c)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, Dropout.name(i))
            build(gcs, i - 1, gate :: network)

          case ReluConfig =>
            val props = ReluGate
              .props(network.head)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, Relu.name(i))
            build(gcs, i - 1, gate :: network)

          case SigmoidConfig =>
            val props = SigmoidGate
              .props(network.head)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, Sigmoid.name(i))
            build(gcs, i - 1, gate :: network)

          case c: BatchNormConfig =>
            val props = BatchNormGate
              .props(network.head, c)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, BatchNorm.name(i))
            build(gcs, i - 1, gate :: network)

          case c: LinearConfig =>
            val props = LinearGate
              .props(network.head, c)
              .withDispatcher("gate-dispatcher")
            val gate = system.actorOf(props, Linear.name(i))
            build(gcs, i - 1, gate :: network)
        }

    }

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
