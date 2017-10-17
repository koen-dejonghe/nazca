package actors.sockets

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import botkop.nn.gates.{CostLogEntry, EvalEntry}
import play.api.libs.json.Json

class MonitorSocket(socket: ActorRef) extends Actor with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("monitor", self)

  override def receive: Receive = {
    case cl: CostLogEntry =>
      socket ! Json.stringify(Json.toJson(cl))
    case ee: EvalEntry =>
      socket ! Json.stringify(Json.toJson(ee))
  }
}

object MonitorSocket {
  def props(socket: ActorRef) =
    Props(new MonitorSocket(socket))
}
