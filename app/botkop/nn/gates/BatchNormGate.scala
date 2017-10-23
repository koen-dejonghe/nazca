package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.{numsca => ns}
import botkop.numsca.Tensor

class BatchNormGate(next: ActorRef, eps: Float, momentum: Float)
    extends Actor
    with ActorLogging {

  val name: String = self.path.name
  log.debug(s"my name is $name")

  def testActivation(x: Tensor, state: BatchNormState): Tensor = {
    import state._
    ((x - runningMean) / ns.sqrt(runningVar + eps)) * gamma + beta
  }

  def trainingActivation(x: Tensor,
                         y: Tensor,
                         state: BatchNormState): BatchNormCache = {
    import state._

    // compute per-dimension mean and std_deviation
    val mean = ns.mean(x, axis = 1)
    val variance = ns.variance(x, axis = 1)

    // normalize and zero-center (explicit for caching purposes)
    val xMu = x - mean
    val invVar = 1.0 / ns.sqrt(variance + eps)
    val xHat = xMu * invVar

    // squash
    val out = xHat * gamma + beta
    next ! Forward(out, y)

    // update running stats
    runningMean *= momentum
    runningMean += (1 - momentum) * mean
    runningVar *= momentum
    runningVar += (1 - momentum) * variance

    BatchNormCache(invVar, xHat)
  }

  def backProp(dout: Tensor,
               prev: ActorRef,
               state: BatchNormState,
               cache: BatchNormCache): Unit = {
    import state._
    import cache._

    val Array(d, n) = dout.shape

    // intermediate partial derivatives
    val dxhat = dout * gamma

    // final partial derivatives
    // val dx = (1.0 / n) * invVar * (n * dxhat - ns.sum(dxhat, axis = 1) - ns.sum(dxhat * xHat, axis = 1) * xHat)
    val dx = (
      (n * dxhat) - ns.sum(dxhat, axis = 1) -
        (xHat * ns.sum(dxhat * xHat, axis = 1))
    ) * invVar * (1.0 / n)

    prev ! Backward(dx)

    val dbeta = ns.sum(dout, axis = 1)
    val dgamma = ns.sum(xHat * dout, axis = 1)

    // note sure about this...
    beta -= dbeta
    gamma -= dgamma
  }

  override def receive: Receive = {
    // 1st time
    case Forward(x, y) =>
      val Array(d, n) = x.shape
      val runningMean = ns.zeros(d, 1)
      val runningVar = ns.zeros(d, 1)
      val gamma = ns.ones(d, 1)
      val beta = ns.zeros(d, 1)
      val state = BatchNormState(runningMean, runningVar, gamma, beta)
      val cache = trainingActivation(x, y, state)
      context become accept(sender(), state, cache)
  }

  def accept(prev: ActorRef,
             state: BatchNormState,
             cache: BatchNormCache): Receive = {
    case Forward(x, y) =>
      val cache = trainingActivation(x, y, state)
      context become accept(sender(), state, cache)

    case Eval(source, id, x, y) =>
      val out = testActivation(x, state)
      next forward Eval(source, id, out, y)

    case Backward(dout) =>
      backProp(dout, prev, state, cache)
  }

  case class BatchNormState(runningMean: Tensor,
                            runningVar: Tensor,
                            gamma: Tensor,
                            beta: Tensor)

  case class BatchNormCache(invVar: Tensor, xHat: Tensor)

}

object BatchNormGate {
  def props(next: ActorRef, eps: Float = 1e-5f, momentum: Float = 0.9f) =
    Props(new BatchNormGate(next, eps, momentum))
}
