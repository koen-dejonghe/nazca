package botkop.nn

import botkop.nn.network.NetworkConfig
import play.api.libs.json.{Format, Json}

case class Project(name: String,
                   miniBatchSize: Int,
                   dataSet: String,
                   persistenceFrequency: Int,
                   template: NetworkConfig)

object Project {
  implicit val f: Format[Project] = Json.format
}
