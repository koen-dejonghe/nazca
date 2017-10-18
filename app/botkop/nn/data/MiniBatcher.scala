package botkop.nn.data

import akka.actor.{Actor, ActorRef, PoisonPill, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import botkop.data.DataLoader
import botkop.nn.gates.{Forward, NextBatch, Quit}
import botkop.numsca.Tensor

class MiniBatcher(dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  override def receive: Receive = accept(dataLoader.nextBatch)

  def accept(batch: (Tensor, Tensor)): Receive = {
    case NextBatch =>
      entryGate forward Forward(batch)
      context become accept(dataLoader.nextBatch)

    case Quit =>
      self ! PoisonPill
  }
}

object MiniBatcher {
  def props(dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new MiniBatcher(dataLoader, entryGate))
}
