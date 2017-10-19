package botkop.nn.data

import akka.actor.{Actor, ActorLogging, ActorRef, PoisonPill, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import akka.pattern.ask
import akka.util.Timeout
import botkop.data.DataLoader
import botkop.nn.gates._

import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

class Evaluator(source: String, dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor with ActorLogging {

  implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("monitor", self)
  mediator ! Subscribe("control", self)

  override def receive: Receive = {
    case Epoch(epoch) =>

      log.debug("received epoch {}", epoch)

      val s: (Int, Double, Double) =
        dataLoader.foldLeft((0, 0.0, 0.0)) {
          case ((n, c, a), batch) =>
            val f = (entryGate ? Eval(source, epoch, batch)).mapTo[EvalEntry]
            val r = Await.result(f, 1 second)
            (n + 1, c + r.cost, a + r.accuracy)
        }

      val cost = s._2 / s._1
      val acc = s._3 / s._1

      mediator ! Publish("monitor", EvalEntry(source, epoch, cost, acc))

    case Quit =>
      self ! PoisonPill
  }

  /*
  timers.startPeriodicTimer(Tick, Tick, interval)
  override def receive: Receive = {
    case nn: Network =>
      val it = dataLoader.iterator
      context become accept(nn, it, it.next())
  }

  def accept(nn: Network,
             it: Iterator[(Tensor, Tensor)],
             batch: (Tensor, Tensor),
             iteration: Int = 1): Receive = {

    case nnn: Network =>
      val nit = dataLoader.iterator
      context become accept(nnn, nit, nit.next())

    case Tick if nn.entryGate.isDefined =>
      nn.entryGate.get ! Eval(source, iteration, batch)
      if (it.hasNext)
        context become accept(nn, it, it.next(), iteration + 1)
      else {
        val nit = dataLoader.iterator
        context become accept(nn, nit, nit.next(), iteration + 1)
      }
  }
case object Tick
   */

  /*
  override def receive: Receive = {
    case nn: Network => context become accept(nn, dataLoader.nextBatch)
  }

  def accept(nn: Network,
             batch: (Tensor, Tensor),
             iteration: Int = 1): Receive = {
    case nnn: Network =>
      context become accept(nnn, dataLoader.nextBatch)

    case Tick if nn.entryGate.isDefined =>
      nn.entryGate.get ! Eval(source, iteration, batch)
      context become accept(nn, dataLoader.nextBatch, iteration + 1)
  }
 */

}

object Evaluator {
  def props(source: String, dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new Evaluator(source, dataLoader, entryGate))
}
