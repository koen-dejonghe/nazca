package botkop.nn.templates

import botkop.numsca
import botkop.numsca.Tensor
import play.api.libs.json._

//============================
// optimizer section

sealed trait OptimizerTemplate {}

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
      case b: AdamOptimizerTemplate =>
        (b, Json.toJson(b)(AdamOptimizerTemplate.f))
      case b: NesterovOptimizerTemplate =>
        (b, Json.toJson(b)(NesterovOptimizerTemplate.f))
    }
    JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> sub))
  }

  implicit val r: Reads[OptimizerTemplate] = OptimizerTemplate.reads
  implicit val w: Writes[OptimizerTemplate] = OptimizerTemplate.writes
}

case class AdamOptimizerTemplate(
    learningRate: Double,
    learningRateDecay: Double,
    beta1: Double,
    beta2: Double,
    epsilon: Double
) extends OptimizerTemplate
object AdamOptimizerTemplate {
  implicit val f: Format[AdamOptimizerTemplate] = Json.format
}

case class NesterovOptimizerTemplate(learningRate: Double,
                                     learningRateDecay: Double,
                                     beta: Double)
    extends OptimizerTemplate
object NesterovOptimizerTemplate {
  implicit val f: Format[NesterovOptimizerTemplate] = Json.format
}

//================================
// cost functions

sealed trait CostTemplate {
  def costFunction(x: Tensor, y: Tensor): (Double, Tensor)
}

object CostTemplate {
  def reads(json: JsValue): JsResult[CostTemplate] = {

    def from(name: String): JsResult[CostTemplate] =
      name match {
        case "CrossEntropy" =>
          JsSuccess(CrossEntropy)
        case "Softmax" =>
          JsSuccess(Softmax)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      result <- from(name)
    } yield result
  }

  def writes(foo: CostTemplate): JsValue = {
    JsObject(Seq("class" -> JsString(foo.asInstanceOf[Product].productPrefix)))
  }

  implicit val r: Reads[CostTemplate] = CostTemplate.reads
  implicit val w: Writes[CostTemplate] = CostTemplate.writes
}

case object CrossEntropy extends CostTemplate {
  override def costFunction(yHat: Tensor, y: Tensor): (Double, Tensor) = {
    val m = y.shape(1)
    val cost = (-y.dot(numsca.log(yHat).T) -
      (1 - y).dot(numsca.log(1 - yHat).T)) / m

    val dal = -(y / yHat - (1 - y) / (1 - yHat))
    (cost.squeeze(), dal)
  }
}

case object Softmax extends CostTemplate {
  override def costFunction(xt: Tensor, yt: Tensor): (Double, Tensor) = {

    val x = xt.T
    val y = yt.T

    val shiftedLogits = x - numsca.max(x, axis = 1)
    val z = numsca.sum(numsca.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - numsca.log(z)
    val probs = numsca.exp(logProbs)
    val n = x.shape(0)
    val loss = -numsca.sum(logProbs(y)) / n

    val dx = probs
    dx.put(y, _ - 1)
    dx /= n

    (loss, dx.T)
  }
}

//=================================
// gates section

sealed trait GateTemplate {
  def +(other: GateTemplate): NetworkTemplate = NetworkTemplate(List(this, other))
  def *(i: Int): NetworkTemplate = NetworkTemplate(List.fill(i)(this))
}

object GateTemplate {
  def reads(json: JsValue): JsResult[GateTemplate] = {

    def from(name: String, data: Option[JsObject]): JsResult[GateTemplate] =
      name match {
        case "LinearGateTemplate" =>
          Json.fromJson[LinearGateTemplate](data.get)
        case "OutputGateTemplate" =>
          Json.fromJson[OutputGateTemplate](data.get)
        case "ReluGateTemplate" =>
          JsSuccess(ReluGateTemplate)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validateOpt[JsObject]
      result <- from(name, data)
    } yield result
  }

  implicit val w: Writes[GateTemplate] = (foo: GateTemplate) => {
    val (prod: Product, sub: Option[JsValue]) = foo match {
      case b: LinearGateTemplate =>
        (b, Some(Json.toJson(b)(LinearGateTemplate.f)))
      case b: OutputGateTemplate =>
        (b, Some(Json.toJson(b)(OutputGateTemplate.f)))
      case ReluGateTemplate => (ReluGateTemplate, None)
    }

    sub match {
      case None =>
        JsObject(Seq("class" -> JsString(prod.productPrefix)))
      case Some(s) =>
        JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> s))
    }
  }

  implicit val r: Reads[GateTemplate] = GateTemplate.reads
}

case class LinearGateTemplate(shape: List[Int],
                              regularization: Double,
                              optimizer: OptimizerTemplate,
                              seed: Long)
    extends GateTemplate
object LinearGateTemplate extends GateTemplate{
  implicit val f: Format[LinearGateTemplate] = Json.format
}

case class OutputGateTemplate(cost: CostTemplate) extends GateTemplate
object OutputGateTemplate {
  implicit val f: Format[OutputGateTemplate] = Json.format
}

case object ReluGateTemplate extends GateTemplate

// ======================
// network section

case class NetworkTemplate(gates: List[GateTemplate]) {
  def +(other: NetworkTemplate): NetworkTemplate = NetworkTemplate(this.gates ++ other.gates)
  def +(gate: GateTemplate): NetworkTemplate = NetworkTemplate(this.gates :+ gate)
  def *(i: Int): NetworkTemplate = NetworkTemplate(List.tabulate(i)(_ => gates).flatten)
}

object NetworkTemplate {
  implicit val f: Format[NetworkTemplate] = Json.format
}
