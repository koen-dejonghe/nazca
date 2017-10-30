package botkop.nn.templates

import play.api.libs.json._

sealed trait Gate

case class Linear(shape: List[Int],
                  regularization: Double,
                  optimizer: Optimizer,
                  seed: Long)
    extends Gate
object Linear {
  implicit val f: Format[Linear] = Json.format
}

case class Output(cost: Cost) extends Gate
object Output {
  implicit val f: Format[Output] = Json.format
}

case object Relu extends Gate

object Gate {
  def reads(json: JsValue): JsResult[Gate] = {

    def from(name: String, data: Option[JsObject]): JsResult[Gate] =
      name match {
        case "Linear" =>
          Json.fromJson[Linear](data.get)
        case "Output" =>
          Json.fromJson[Output](data.get)
        case "Relu" =>
          JsSuccess(Relu)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validateOpt[JsObject]
      result <- from(name, data)
    } yield result
  }

  implicit val w: Writes[Gate] = (foo: Gate) => {
    val (prod: Product, sub: Option[JsValue]) = foo match {
      case b: Linear =>
        (b, Some(Json.toJson(b)(Linear.f)))
      case b: Output =>
        (b, Some(Json.toJson(b)(Output.f)))
      case Relu => (Relu, None)
    }

    sub match {
      case None =>
        JsObject(Seq("class" -> JsString(prod.productPrefix)))
      case Some(s) =>
        JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> s))
    }
  }

  implicit val r: Reads[Gate] = Gate.reads
}
