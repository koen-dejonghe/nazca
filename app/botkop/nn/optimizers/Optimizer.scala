package botkop.nn.optimizers

import botkop.numsca.Tensor
import play.api.libs.json._

trait Optimizer extends Serializable {

  /**
    * Update the parameters (weights and biases) of a Gate
    * @param parameters List of parameters where parameters(0) = weights, and parameters(1) = biases
    * @param gradients List of gradients where gradients(0) = gradients of the weights, and gradients(1) = gradients of the biases
    */
  def update(parameters: List[Tensor], gradients: List[Tensor]): Unit

  def setLearningRate(learningRate: Double): Unit

  def updateLearningRate(): Unit

}

/**
  * json boilerplate
  */
object Optimizer {
  def reads(json: JsValue): JsResult[Optimizer] = {

    def from(name: String, data: JsObject): JsResult[Optimizer] =
      name match {
        case "Adam" =>
          Json.fromJson[AdamOptimizer](data)
        case "GradientDescent" =>
          Json.fromJson[GradientDescentOptimizer](data)
        case "Momentum" =>
          Json.fromJson[MomentumOptimizer](data)
        case "Nesterov" =>
          Json.fromJson[NesterovOptimizer](data)
        case _ => JsError(s"Unknown class '$name'")
      }

    for {
      name <- (json \ "class").validate[String]
      data <- (json \ "data").validate[JsObject]
      result <- from(name, data)
    } yield result
  }

  def writes(foo: Optimizer): JsValue = {
    val (prod: Product, sub) = foo match {
      case b: AdamOptimizer =>
        (b, Json.toJson(b)(AdamOptimizer.f))
      case b: GradientDescentOptimizer =>
        (b, Json.toJson(b)(GradientDescentOptimizer.f))
      case b: MomentumOptimizer =>
        (b, Json.toJson(b)(MomentumOptimizer.f))
      case b: NesterovOptimizer =>
        (b, Json.toJson(b)(NesterovOptimizer.f))
    }
    JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> sub))
  }

  implicit val r: Reads[Optimizer] = Optimizer.reads
  implicit val w: Writes[Optimizer] = Optimizer.writes
}

