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
    extends Actor
    with ActorLogging {

  implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("monitor", self)
  mediator ! Subscribe("control", self)

  override def receive: Receive = {
    case Epoch(epoch) =>
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

    case Quit =>
      self ! PoisonPill
  }

}

object Evaluator {
  def props(source: String, dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new Evaluator(source, dataLoader, entryGate))
}
