package botkop.nn.gates

import akka.actor.ActorSystem
import botkop.nn.costs.{Cost, CrossEntropy}
import botkop.nn.optimizers._

import scala.language.postfixOps

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
  def withDimensions(dims: Int*): NetworkBuilder =
    copy(dimensions = dims.toList)
  def withCostFunction(cf: Cost): NetworkBuilder = copy(cost = cf)
  def withOptimizer(o: OptimizerStub): NetworkBuilder = copy(optimizer = o)
  def withLearningRate(d: Double): NetworkBuilder = copy(learningRate = d)
  def withLearningRateDecay(d: Double): NetworkBuilder =
    copy(learningRateDecay = d)
  def withRegularization(reg: Double): NetworkBuilder =
    copy(regularization = reg)

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

    val closedGates = if (gates.last != Output) gates :+ Output else gates

    val configs = closedGates.zipWithIndex.map {
      case (g, i) =>
        val numLinear = closedGates take (i+1) count(Linear ==)
        val shape = dimensions.slice(numLinear - 1, numLinear + 1)
        println(shape)
        g match {
          case BatchNorm =>
            BatchNormConfig(shape)
          case Dropout =>
            DropoutConfig()
          case Linear =>
            LinearConfig(shape, regularization, op)
          case Relu =>
            ReluConfig
          case Sigmoid =>
            SigmoidConfig
          case Output =>
            OutputConfig(cost)
        }
    }

    NetworkConfig(configs).materialize
  }

}

