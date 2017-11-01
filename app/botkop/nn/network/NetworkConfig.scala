package botkop.nn.network

import akka.actor.{ActorRef, ActorSystem}
import botkop.nn.gates.GateConfig
import play.api.libs.json.{Format, Json}

case class NetworkConfig(configs: List[GateConfig]) {
  def materialize(implicit system: ActorSystem,
                  projectName: String): Network = {
    val gates: List[ActorRef] =
      configs.zipWithIndex.reverse.foldLeft(List.empty[ActorRef]) {
        case (gs, (cfg, index)) =>
          if (gs.isEmpty)
            cfg.materialize(None, index) :: gs
          else
            cfg.materialize(Some(gs.head), index) :: gs
      }
    Network(projectName, gates, this)
  }
}

object NetworkConfig {
  implicit val f: Format[NetworkConfig] = Json.format
}
