package sockets

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.nn.Project
import botkop.nn.gates.CanvasMessage
import play.api.libs.json.{JsError, JsResult, JsSuccess, Json}

class CanvasSocket(socket: ActorRef) extends Actor with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  override def receive: Receive = {

    // from socket -> publish to driver
    case json: String =>

      println(json)

      // should be valid project json
      val result = Json.parse(json).validate[Project]

      // Pattern matching
      result match {
        case s: JsSuccess[Project] =>
          log.info("publishing project")
          mediator ! Publish("control", s.get)
        case e: JsError =>
          log.error("Errors: " + JsError.toJson(e).toString())
      }

    // from mediator -> publish to socket
    case CanvasMessage("socket", message) =>
      socket ! message
  }
}

object CanvasSocket {
  def props(socket: ActorRef) =
    Props(new CanvasSocket(socket))
}
