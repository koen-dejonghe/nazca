package botkop.nn.cluster

import akka.actor.ActorSystem
import botkop.nn.cluster.data.MnistDataLoader
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging

import scala.language.postfixOps

object ClusterApp extends App with LazyLogging {

  // todo: figure out how to make this more dynamic
  val dataLoader = new MnistDataLoader(take=Some(100))

  val config = ConfigFactory.load("cluster")

  implicit val system: ActorSystem =
    ActorSystem("NeuralSystem", config)

  logger.debug(s"${system.name} is ready")

  val proxy = system.actorOf(Proxy.props(dataLoader, 10000), "proxy")

}
