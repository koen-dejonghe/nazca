package botkop.nn.templates

import play.api.libs.json._

case class Network(gates: List[Gate]) {
}

object Network {
  implicit val f: Format[Network] = Json.format
}
