package botkop.nn.gates

import akka.actor.{ActorContext, ActorRef}
import com.typesafe.scalalogging.LazyLogging
import play.api.libs.json._

trait GateConfig {
  def materialize(next: Option[ActorRef], index: Int)(
      implicit context: ActorContext,
      projectName: String): ActorRef
}

object GateConfig extends LazyLogging {
  def reads(json: JsValue): JsResult[GateConfig] = {

    def from(name: String, data: Option[JsObject]): JsResult[GateConfig] =
      name match {
        case "BatchNormConfig" =>
          Json.fromJson[BatchNormConfig](data.get)
        case "ConvConfig" =>
          Json.fromJson[ConvConfig](data.get)
        case "DropoutConfig" =>
          Json.fromJson[DropoutConfig](data.get)
        case "LinearConfig" =>
          Json.fromJson[LinearConfig](data.get)
        case "OutputConfig" =>
          Json.fromJson[OutputConfig](data.get)
        case "ReluConfig" =>
          JsSuccess(ReluConfig)
        case "SigmoidConfig" =>
          JsSuccess(SigmoidConfig)
        case _ =>
          logger.error(s"unknown class $name")
          JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validateOpt[JsObject]
      result <- from(name, data)
    } yield result
  }

  implicit val w: Writes[GateConfig] = (foo: GateConfig) => {
    val (prod: Product, sub: Option[JsValue]) = foo match {
      case b: BatchNormConfig =>
        (b, Some(Json.toJson(b)(BatchNormConfig.f)))
      case b: ConvConfig =>
        (b, Some(Json.toJson(b)(ConvConfig.f)))
      case b: DropoutConfig =>
        (b, Some(Json.toJson(b)(DropoutConfig.f)))
      case b: LinearConfig =>
        (b, Some(Json.toJson(b)(LinearConfig.f)))
      case b: OutputConfig =>
        (b, Some(Json.toJson(b)(OutputConfig.f)))
      case ReluConfig    => (ReluConfig, None)
      case SigmoidConfig => (SigmoidConfig, None)
      case c =>
        throw new IllegalArgumentException(s"unknown class ${c.getClass}")
    }

    sub match {
      case None =>
        JsObject(Seq("class" -> JsString(prod.productPrefix)))
      case Some(s) =>
        JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> s))
    }
  }

  implicit val r: Reads[GateConfig] = GateConfig.reads
}
