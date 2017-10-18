package botkop.nn.data

import akka.actor.{Actor, ActorRef, Props, Timers}
import akka.cluster.pubsub.DistributedPubSub
import botkop.data.DataLoader
import botkop.nn.gates.{Eval, Network}
import botkop.numsca.Tensor

import scala.concurrent.duration._
import scala.language.postfixOps

class Evaluator(source: String,
                dataLoader: DataLoader,
                interval: FiniteDuration)
    extends Actor
    with Timers {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator

  timers.startPeriodicTimer(EvalTick, EvalTick, interval)

  override def receive: Receive = {
    case nn: Network => context become accept(nn, dataLoader.nextBatch)
  }

  def accept(nn: Network,
             batch: (Tensor, Tensor),
             iteration: Int = 1): Receive = {
    case nnn: Network =>
      context become accept(nnn, dataLoader.nextBatch)

    case EvalTick if nn.entryGate.isDefined =>
      nn.entryGate.get ! Eval(source, iteration, batch)
      context become accept(nn, dataLoader.nextBatch, iteration + 1)
  }

}

object Evaluator {
  def props(source: String, dataLoader: DataLoader, interval: FiniteDuration) =
    Props(new Evaluator(source, dataLoader, interval))
}

case object EvalTick
