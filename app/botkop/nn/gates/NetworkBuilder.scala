package botkop.nn.gates

import akka.actor.ActorSystem
import botkop.nn.costs.{Cost, CrossEntropy}
import botkop.nn.optimizers._

import scala.language.postfixOps

case class NetworkBuilder(gateStubs: List[GateStub] = List.empty,
                          dimensions: List[Int] = List.empty,
                          cost: Cost = CrossEntropy,
                          optimizerStub: OptimizerStub = GradientDescent,
                          learningRate: Double = 0.001,
                          learningRateDecay: Double = 0.95,
                          regularization: Double = 0.0) {

  def +(other: NetworkBuilder) =
    NetworkBuilder(this.gateStubs ++ other.gateStubs)
  def +(layer: GateStub) = NetworkBuilder(this.gateStubs :+ layer)
  def *(i: Int): NetworkBuilder =
    NetworkBuilder(List.tabulate(i)(_ => gateStubs).flatten)

  def withGates(gs: GateStub*): NetworkBuilder = copy(gateStubs = gs.toList)
  def withDimensions(dims: Int*): NetworkBuilder =
    copy(dimensions = dims.toList)
  def withCostFunction(cf: Cost): NetworkBuilder = copy(cost = cf)
  def withOptimizer(o: OptimizerStub): NetworkBuilder = copy(optimizerStub = o)
  def withLearningRate(d: Double): NetworkBuilder = copy(learningRate = d)
  def withLearningRateDecay(d: Double): NetworkBuilder =
    copy(learningRateDecay = d)
  def withRegularization(reg: Double): NetworkBuilder =
    copy(regularization = reg)

  def makeOptimizer(stub: OptimizerStub): Optimizer = stub match {
    case Adam =>
      AdamOptimizer(learningRate, learningRateDecay)
    case GradientDescent =>
      GradientDescentOptimizer(learningRate, learningRateDecay)
    case Momentum =>
      MomentumOptimizer(learningRate, learningRateDecay)
    case Nesterov =>
      NesterovOptimizer(learningRate, learningRateDecay)
  }

  def networkConfig: NetworkConfig = {
    require(gateStubs.nonEmpty, "no gates defined")
    require(dimensions.nonEmpty, "no dimensions defined")
    val numLinearGates = gateStubs.count(g => g == Linear)
    require(
      numLinearGates == dimensions.length - 1,
      "dimension size (d) does not match number of linear gates (l): l must be = (d - 1)")

    val closedCircuit =
      if (gateStubs.last != Output) gateStubs :+ Output else gateStubs

    require(closedCircuit.count(Output ==) == 1,
            "number of output gates must be 1")

    val configs = closedCircuit.zipWithIndex.map {
      case (g, i) =>
        val numLinear = closedCircuit take (i + 1) count (Linear ==)
        val shape = dimensions.slice(numLinear - 1, numLinear + 1).reverse
        g match {
          case BatchNorm =>
            BatchNormConfig(shape)
          case Dropout =>
            DropoutConfig()
          case Linear =>
            LinearConfig(shape, regularization, makeOptimizer(optimizerStub))
          case Relu =>
            ReluConfig
          case Sigmoid =>
            SigmoidConfig
          case Output =>
            OutputConfig(cost)
        }
    }
    NetworkConfig(configs)
  }

  def build(implicit system: ActorSystem, projectName: String): Network = {
    networkConfig.materialize
  }

}
