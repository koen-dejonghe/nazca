package botkop.nn.gates

import akka.actor.{ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import akka.persistence._
import botkop.nn.optimizers.Optimizer
import botkop.numsca
import botkop.numsca.Tensor

import scala.language.postfixOps

class LinearGate(shape: Array[Int],
                 next: ActorRef,
                 regularization: Double,
                 var optimizer: Optimizer,
                 seed: Long = 231)
    extends PersistentActor
    with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  numsca.rand.setSeed(seed)

  val name: String = self.path.name
  log.debug(s"my name is $name")

  var w: Tensor = numsca.randn(shape) * math.sqrt(2.0 / shape(1))
  var b: Tensor = numsca.zeros(shape.head, 1)
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
      next ! Eval(source, id, z, y)

    case Backward(dz) if cache isDefined =>
      val (prev, a) = cache.get

      val da = w.transpose.dot(dz)
      prev ! Backward(da)

      val m = a.shape(1)
      val dw = dz.dot(a.T) / m

      // adjusting regularization, if needed
      if (regularization != 0)
        dw += regularization * w

      val db = numsca.sum(dz, axis = 1) / m

      optimizer.update(List(w, b), List(dw, db))

    case ss: SaveSnapshotSuccess =>
      deleteSnapshots(
        SnapshotSelectionCriteria.create(ss.metadata.sequenceNr,
                                         ss.metadata.timestamp - 1000))

    case Persist =>
      saveSnapshot(LinearState(w, b, optimizer))

    case SetLearningRate(lr) =>
      optimizer.setLearningRate(lr)
  }

  override def receiveRecover: Receive = {
    case SnapshotOffer(meta: SnapshotMetadata, snapshot: LinearState) =>
      log.debug(s"$name: received snapshot ${meta.persistenceId}")
      w = snapshot.w
      b = snapshot.b
      optimizer = snapshot.optimizer
      accept()
  }

  override def persistenceId: String = name

  override def receiveCommand: Receive = accept()

}

object LinearGate {
  def props(shape: Array[Int],
            next: ActorRef,
            regularization: Double,
            optimizer: Optimizer) =
    Props(new LinearGate(shape, next, regularization, optimizer))
}

case class LinearState(w: Tensor, b: Tensor, optimizer: Optimizer)
