package botkop.nn.gates

import akka.actor.{ActorContext, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import akka.persistence._
import botkop.nn.optimizers.Optimizer
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import play.api.libs.json.{Format, Json}

import scala.language.postfixOps

class LinearGate(next: ActorRef, config: LinearConfig)
    extends PersistentActor
    with ActorLogging {

  import config._

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)
  mediator ! Subscribe("monitor", self)

  ns.rand.setSeed(seed)

  val name: String = self.path.name
  log.debug(s"my name is $name")

  log.debug(self.path.toString)

  var w: Tensor = ns.randn(shape.toArray) * math.sqrt(2.0 / shape(1))
  var b: Tensor = ns.zeros(shape.head, 1)
  var cache: Option[(ActorRef, Tensor)] = None

  def activate(x: Tensor): Tensor = w.dot(x) + b

  def accept(): Receive = {

    case Forward(x, y) =>
      val z = activate(x)
      next ! Forward(z, y)
      cache = Some(sender(), x)

    case Predict(x) =>
      val z = activate(x)
      next forward Predict(z)

    case Eval(source, id, x, y) =>
      val z = activate(x)
      next forward Eval(source, id, z, y)

    case Backward(dz) if cache isDefined =>
      val (prev, a) = cache.get

      val da = w.T.dot(dz)
      prev ! Backward(da)

      val m = a.shape(1)
      val dw = dz.dot(a.T) / m

      // adjusting regularization, if needed
      if (regularization != 0)
        dw += regularization * w

      val db = ns.sum(dz, axis = 1) / m

      optimizer.update(List(w, b), List(dw, db))

    case ss: SaveSnapshotSuccess =>
      deleteSnapshots(
        SnapshotSelectionCriteria.create(ss.metadata.sequenceNr,
                                         ss.metadata.timestamp - 1000))

    case Persist =>
      saveSnapshot(LinearState(w, b, optimizer))

    case SetLearningRate(lr) =>
      optimizer.setLearningRate(lr)

    case Epoch(_, _) => // adjust learning rate every epoch
      optimizer.updateLearningRate()

  }

  override def receiveRecover: Receive = {
    case SnapshotOffer(meta: SnapshotMetadata, snapshot: LinearState) =>
      log.debug(s"$name: received snapshot ${meta.persistenceId}")
      w = snapshot.w
      b = snapshot.b
      optimizer = snapshot.optimizer
  }

  override def persistenceId: String = name

  override def receiveCommand: Receive = accept()

}

object LinearGate {
  def props(next: ActorRef, config: LinearConfig): Props =
    Props(new LinearGate(next, config))
}

case class LinearState(w: Tensor, b: Tensor, optimizer: Optimizer)

case class LinearConfig(shape: List[Int],
                        regularization: Double,
                        var optimizer: Optimizer,
                        seed: Long = 231L)
    extends GateConfig {
  override def materialize(next: Option[ActorRef], index: Int)(
      implicit context: ActorContext,
      projectName: String): ActorRef = {
    context.actorOf(LinearGate.props(next.get, this), Linear.name(index))
  }
}

object LinearConfig {
  implicit val f: Format[LinearConfig] = Json.format
}
