package botkop.nn.data

import akka.actor.{Actor, ActorRef, Props}
import botkop.data.DataLoader
import botkop.nn.gates.{Forward, NextBatch}
import botkop.numsca.Tensor

class MiniBatchFetcher(dataLoader: DataLoader, entryGate: ActorRef)
    extends Actor {

  override def receive: Receive = accept(dataLoader.nextBatch)

  def accept(batch: (Tensor, Tensor)): Receive = {
    case NextBatch =>
      entryGate forward Forward(batch)
      context become accept(dataLoader.nextBatch)
  }
}

object MiniBatchFetcher {
  def props(dataLoader: DataLoader, entryGate: ActorRef) =
    Props(new MiniBatchFetcher(dataLoader, entryGate))
}
