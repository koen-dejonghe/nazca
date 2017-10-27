package botkop.nn.data

import akka.actor.{Actor, ActorLogging, ActorRef, PoisonPill, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import akka.util.Timeout
import botkop.nn.data.loaders.DataLoader
import botkop.nn.gates._
import botkop.numsca.Tensor

import scala.concurrent.duration._
import scala.language.postfixOps

class Evaluator(source: String, dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor
    with ActorLogging {

  implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

  val parallelism = 4

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("monitor", self)
  mediator ! Subscribe("control", self)

  override def receive: Receive = {
    case Epoch(epoch, _) =>
      val iterator = dataLoader.iterator

      for (_ <- 1 to parallelism if iterator.hasNext) {
        entryGate ! Eval(source, epoch, iterator.next())
      }

      // if (iterator.hasNext) {
        entryGate ! Eval(source, epoch, iterator.next())
        val entry = EvalEntry(source, epoch, 0.0, 0.0)
        context become accumulate(iterator, epoch, entry)
      // } else {
        // log.error("iterator has no elements")
      // }
    /*
      val s: (Int, Double, Double) =
        dataLoader.foldLeft((0, 0.0, 0.0)) {
          case ((n, c, a), batch) =>
            val f = (entryGate ? Eval(source, epoch, batch)).mapTo[EvalEntry]
            val r = Await.result(f, timeout.duration)
            (n + 1, c + r.cost, a + r.accuracy)
        }

      val cost = s._2 / s._1
      val acc = s._3 / s._1

      mediator ! Publish("monitor", EvalEntry(source, epoch, cost, acc))
     */

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
