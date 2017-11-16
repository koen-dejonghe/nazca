package controllers

import java.io.File
import javax.inject._

import akka.actor.{ActorRef, ActorSystem}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Publish
import akka.stream.Materializer
import botkop.nn.gates.CanvasMessage
import botkop.nn.{NetDriver, Project}
import com.typesafe.scalalogging.LazyLogging
import play.api.libs.Files
import play.api.libs.json.Json
import play.api.libs.streams.ActorFlow
import play.api.mvc._
import sockets.{CanvasSocket, ControlSocket, MonitorSocket}

import scala.io.Source

@Singleton
class NetController @Inject()(cc: ControllerComponents)(
    implicit system: ActorSystem,
    mat: Materializer)
    extends AbstractController(cc)
    with LazyLogging {

  val mediator: ActorRef = DistributedPubSub(system).mediator
  val driver: ActorRef = system.actorOf(NetDriver.props(), "driver")

  def index = Action {
    Ok(views.html.index())
  }

  def controlSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      ControlSocket.props(out)
    }
  }

  def monitorSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      MonitorSocket.props(out)
    }
  }

  def canvasSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      CanvasSocket.props(out)
    }
  }

  def upload: Action[MultipartFormData[Files.TemporaryFile]] =
    Action(parse.multipartFormData) { request =>

      logger.info("recevied files " + request.body.files.toList)

      request.body
        .file("project")
        .map { json =>
          val filename = json.filename
          json.contentType match {
            case Some("application/json") =>
              try {
                val tmpFile = File.createTempFile("botkop.nn-", null)
                json.ref.moveTo(tmpFile, replace = true)
                val contents =
                  Source.fromFile(tmpFile).getLines().mkString("\n")
                tmpFile.delete()
                mediator ! Publish("control",
                                   CanvasMessage(target = "socket",
                                                 message = contents))

                Ok(s"$filename uploaded")
              } catch {
                case t: Throwable =>
                  logger.error("error during upload", t)
                  BadRequest("error during upload")
              }
            case _ =>
              logger.error("unknown content type")
              BadRequest("unknown content type")
          }
        }
        .getOrElse{
          logger.error("error during upload")
          BadRequest
        }
    }
}
