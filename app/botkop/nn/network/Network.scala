package botkop.nn.network

import akka.actor.{ActorRef, PoisonPill}

case class Network(projectName: String,
                   gates: List[ActorRef],
                   config: NetworkConfig) {
  val entryGate: ActorRef = gates.head
  def quit(): Unit = gates.foreach(_ ! PoisonPill)
}

