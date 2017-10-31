package botkop.nn.gates

import akka.actor.{ActorLogging, ActorRef, ActorSystem, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import akka.persistence._
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import play.api.libs.json.{Format, Json}

class BatchNormGate(next: ActorRef, config: BatchNormConfig)
    extends PersistentActor
    with ActorLogging {

  import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  val Array(d, n) = shape.toArray

  val runningMean: Tensor = ns.zeros(d, 1)
  val runningVar: Tensor = ns.zeros(d, 1)
  val gamma: Tensor = ns.ones(d, 1)
  val beta: Tensor = ns.zeros(d, 1)

  def testActivation(x: Tensor): Tensor = {
    ((x - runningMean) / ns.sqrt(runningVar + eps)) * gamma + beta
  }

  def trainingActivation(x: Tensor, y: Tensor): BatchNormCache = {

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

  def backProp(dout: Tensor, prev: ActorRef, cache: BatchNormCache): Unit = {
    import cache._

    // intermediate partial derivatives
    val dxhat = dout * gamma

    // final partial derivatives
    val dx = (
      (n * dxhat) - ns.sum(dxhat, axis = 1) -
        (xHat * ns.sum(dxhat * xHat, axis = 1))
    ) * invVar * (1.0 / n)

    prev ! Backward(dx)

    val dbeta = ns.sum(dout, axis = 1)
    val dgamma = ns.sum(xHat * dout, axis = 1)

    // not sure about this...
    beta -= dbeta
    gamma -= dgamma
  }

  def accept(prev: ActorRef, cache: BatchNormCache): Receive = {
    case Forward(x, y) =>
      val cache = trainingActivation(x, y)
      context become accept(sender(), cache)

    case Eval(source, id, x, y) =>
      val out = testActivation(x)
      next forward Eval(source, id, out, y)

    case Backward(dout) =>
      backProp(dout, prev, cache)

    case ss: SaveSnapshotSuccess =>
      deleteSnapshots(
        SnapshotSelectionCriteria.create(ss.metadata.sequenceNr,
                                         ss.metadata.timestamp - 1000))

    case Persist =>
      saveSnapshot(BatchNormState(runningMean, runningVar, gamma, beta))

  }

  override def receiveCommand: Receive = {
    case Forward(x, y) =>
      val cache = trainingActivation(x, y)
      context become accept(sender(), cache)
    case Eval(source, id, x, y) =>
      val out = testActivation(x)
      next forward Eval(source, id, out, y)
  }

  override def persistenceId: String = name

  override def receiveRecover: Receive = {
    case SnapshotOffer(meta: SnapshotMetadata, snapshot: BatchNormState) =>
      log.debug(s"$name: received snapshot ${meta.persistenceId}")
      runningMean := snapshot.runningMean
      runningVar := snapshot.runningVar
      gamma := snapshot.gamma
      beta := snapshot.beta
  }
}

object BatchNormGate {
  def props(next: ActorRef, config: BatchNormConfig) =
    Props(new BatchNormGate(next, config))
}

case class BatchNormState(runningMean: Tensor,
                          runningVar: Tensor,
                          gamma: Tensor,
                          beta: Tensor)

case class BatchNormCache(invVar: Tensor, xHat: Tensor)

case class BatchNormConfig(shape: List[Int],
                           eps: Float = 1e-5f,
                           momentum: Float = 0.9f)
    extends GateConfig {

  def materialize(next: Option[ActorRef], index: Int)(
      implicit system: ActorSystem, projectName: String): ActorRef = {
    val props = BatchNormGate
      .props(next.get, this)
      .withDispatcher("gate-dispatcher")
    system.actorOf(props, BatchNorm.name(index))
  }

}
object BatchNormConfig {
  implicit val f: Format[BatchNormConfig] = Json.format
}
