package botkop.nn.data

import akka.actor.{Actor, ActorLogging, ActorRef, PoisonPill, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.data.DataLoader
import botkop.nn.gates.{Epoch, Forward, NextBatch, Quit}
import botkop.numsca.Tensor

class MiniBatcher(dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor
    with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  override def receive: Receive = {
    log.info("starting epoch 1")
    val it = dataLoader.iterator
    accept(it.next(), it, 1)
  }

  def accept(batch: (Tensor, Tensor),
             it: Iterator[(Tensor, Tensor)],
             epoch: Int): Receive = {

    case NextBatch =>
      entryGate forward Forward(batch)
      if (it.hasNext)
        context become accept(it.next(), it, epoch)
      else {
        val newEpoch = epoch + 1
        log.info(s"starting epoch $newEpoch")
        mediator ! Publish("monitor", Epoch(newEpoch))
        val nit = dataLoader.iterator
        context become accept(nit.next(), nit, newEpoch)
      }

    case Quit =>
      self ! PoisonPill
  }

}

object MiniBatcher {
  def props(dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new MiniBatcher(dataLoader, entryGate))
}
