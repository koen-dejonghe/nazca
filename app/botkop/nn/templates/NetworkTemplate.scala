package botkop.nn.templates

import play.api.libs.json._

//============================
// optimizer section

sealed trait OptimizerParameters

case class AdamOptimizerParameters(
    learningRate: Double,
    learningRateDecay: Double,
    beta1: Double,
    beta2: Double,
    epsilon: Double
) extends OptimizerParameters
object AdamOptimizerParameters {
  /*
  def apply(s: String): AdamOptimizerParameters = {
    val json = Json.parse(s)
    val lr = (json \ "learningRate").as[Double]
    val lrd = (json \ "learningRateDecay").as[Double]
    val b1 = (json \ "beta1").as[Double]
    val b2 = (json \ "beta2").as[Double]
    val eps = (json \ "eps").as[Double]
    AdamOptimizerParameters(lr, lrd, b1, b2, eps)
  }
   */
  implicit val r: Reads[AdamOptimizerParameters] = Json.reads
  implicit val w: Writes[AdamOptimizerParameters] =
    Json.writes[AdamOptimizerParameters]
}

case class NesterovOptimizerParameters(
    learningRate: Double,
    learningRateDecay: Double,
    beta: Double
) extends OptimizerParameters
object NesterovOptimizerParameters {
  /*
  def apply(s: String): NesterovOptimizerParameters = {
    val json: JsValue = Json.parse(s)
    val lr = (json \ "learningRate").as[Double]
    val lrd = (json \ "learningRateDecay").as[Double]
    val beta = (json \ "beta").as[Double]
    NesterovOptimizerParameters(lr, lrd, beta)
  }
   */
  implicit val r: Reads[NesterovOptimizerParameters] = Json.reads
  implicit val w: Writes[NesterovOptimizerParameters] =
    Json.writes[NesterovOptimizerParameters]
}

sealed trait OptimizerTemplate {
  def parameters: OptimizerParameters
}
object OptimizerTemplate {
  def reads(json: JsValue): JsResult[OptimizerTemplate] = {

    def from(name: String, data: JsObject): JsResult[OptimizerTemplate] =
      name match {
        case "AdamOptimizerTemplate" =>
          Json.fromJson[AdamOptimizerTemplate](data)
        case "NesterovOptimizerTemplate" =>
          Json.fromJson[NesterovOptimizerTemplate](data)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validate[JsObject]
      result <- from(name, data)
    } yield result
  }

  def writes(foo: OptimizerTemplate): JsValue = {
    val (prod: Product, sub) = foo match {
      case b: AdamOptimizerTemplate     => (b, Json.toJson(b)(AdamOptimizerTemplate.w))
      case b: NesterovOptimizerTemplate => (b, Json.toJson(b)(NesterovOptimizerTemplate.w))
    }
    JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> sub))
  }

  implicit val r: Reads[OptimizerTemplate] = OptimizerTemplate.reads
  implicit val w: Writes[OptimizerTemplate] = OptimizerTemplate.writes
}

case class AdamOptimizerTemplate(parameters: AdamOptimizerParameters)
    extends OptimizerTemplate
object AdamOptimizerTemplate {
  implicit val r: Reads[AdamOptimizerTemplate] = Json.reads
  implicit val w: Writes[AdamOptimizerTemplate] =
    Json.writes[AdamOptimizerTemplate]
}

case class NesterovOptimizerTemplate(parameters: NesterovOptimizerParameters)
    extends OptimizerTemplate
object NesterovOptimizerTemplate {
  implicit val r: Reads[NesterovOptimizerTemplate] = Json.reads
  implicit val w: Writes[NesterovOptimizerTemplate] =
    Json.writes[NesterovOptimizerTemplate]
}

//================================
// cost functions

/*
sealed trait CostTemplate
object CostTemplate {
  def reads(json: JsValue): JsResult[CostTemplate] = {

    def from(name: String, data: JsObject): JsResult[CostTemplate] =
      name match {
        case "CrossEntropy" =>
          Json.fromJson[CrossEntropy](data)
        case "Softmax" =>
          Json.fromJson[Softmax](data)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validate[JsObject]
      result <- from(name, data)
    } yield result
  }

  implicit val r: Reads[CostTemplate] = CostTemplate.reads
}

case class CrossEntropy() extends CostTemplate
object CrossEntropy {
  implicit val r: Reads[CrossEntropy] = Json.reads
}
case class Softmax() extends CostTemplate
object Softmax {
  implicit val r: Reads[Softmax] = Json.reads
}
 */

//=================================
// gates section

trait GateParameters

case class LinearGateParameters(
    shape: Array[Int],
    regularization: Double,
    optimizer: OptimizerTemplate,
    seed: Long
) extends GateParameters
object LinearGateParameters {
  implicit val r: Reads[LinearGateParameters] = Json.reads
  implicit val w: Writes[LinearGateParameters] = Json.writes[LinearGateParameters]
}

trait GateTemplate {
  def parameters: GateParameters
}
object GateTemplate {
  def reads(json: JsValue): JsResult[GateTemplate] = {

    def from(name: String, data: JsObject): JsResult[GateTemplate] =
      name match {
        case "LinearGateTemplate" =>
          Json.fromJson[LinearGateTemplate](data)
        case "OutputGateTemplate" =>
          Json.fromJson[OutputGateTemplate](data)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validate[JsObject]
      result <- from(name, data)
    } yield result
  }

  implicit val w: Writes[GateTemplate] = (foo: GateTemplate) => {
    val (prod: Product, sub: JsValue) = foo match {
      case b: LinearGateTemplate => (b, Json.toJson(b)(LinearGateTemplate.w))
      case b: OutputGateTemplate => (b, Json.toJson(b)(OutputGateTemplate.w))
    }
    JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> sub))
  }

  implicit val r: Reads[GateTemplate] = GateTemplate.reads
  // implicit val w: Writes[GateTemplate] = GateTemplate.writes
}

case class LinearGateTemplate(parameters: LinearGateParameters)
    extends GateTemplate
object LinearGateTemplate {
  implicit val r: Reads[LinearGateTemplate] = Json.reads
  implicit val w: Writes[LinearGateTemplate] = Json.writes[LinearGateTemplate]
}

case class OutputGateTemplate(parameters: OutputGateParameters)
    extends GateTemplate
object OutputGateTemplate {
  implicit val r: Reads[OutputGateTemplate] = Json.reads
  implicit val w: Writes[OutputGateTemplate] = Json.writes[OutputGateTemplate]
}

case class OutputGateParameters(costFunction: String) extends GateParameters
object OutputGateParameters {
  implicit val r: Reads[OutputGateParameters] = Json.reads
  implicit val w: Writes[OutputGateParameters] = Json.writes[OutputGateParameters]
}

// ======================
// network section

case class NetworkTemplate(gates: List[GateTemplate]) {}
object NetworkTemplate {
  implicit val r: Reads[NetworkTemplate] = Json.reads
  implicit val w: Writes[NetworkTemplate] = Json.writes[NetworkTemplate]
}
