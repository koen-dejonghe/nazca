package botkop.nn.data

import akka.actor.{Actor, ActorLogging, ActorRef, PoisonPill, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.nn.data.loaders.DataLoader
import botkop.nn.gates._
import botkop.numsca.Tensor

import scala.language.postfixOps

class Evaluator(source: String, dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor
    with ActorLogging {

  val parallelism: Int = Runtime.getRuntime.availableProcessors() / 2

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("monitor", self)
  mediator ! Subscribe("control", self)

  override def receive: Receive = {
    case Epoch(epoch, _) =>
      val iterator = dataLoader.iterator

      for (_ <- 1 to parallelism if iterator.hasNext) {
        entryGate ! Eval(source, epoch, iterator.next())
      }

      val entry = EvalEntry(source, epoch, 0.0, 0.0)
      context become accumulate(iterator, epoch, entry)

    case Quit =>
      self ! PoisonPill
  }

  def accumulate(iterator: Iterator[(Tensor, Tensor)],
                 epoch: Int,
                 accumulator: EvalEntry): Receive = {
    case ee: EvalEntry =>
      val cost = accumulator.cost + ee.cost
      val accuracy = accumulator.accuracy + ee.accuracy
      val nextEntry = accumulator.copy(cost = cost, accuracy = accuracy)

      if (iterator.hasNext) {
        entryGate ! Eval(source, epoch, iterator.next())
        context become accumulate(iterator, epoch, nextEntry)
      } else {
        val cost = nextEntry.cost / dataLoader.numBatches
        val accuracy = nextEntry.accuracy / dataLoader.numBatches
        mediator ! Publish("monitor", EvalEntry(source, epoch, cost, accuracy))
        context become receive
      }
  }

}

object Evaluator {
  def props(source: String, dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new Evaluator(source, dataLoader, entryGate))
}
