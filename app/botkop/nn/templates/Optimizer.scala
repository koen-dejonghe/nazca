package botkop.nn.templates

import botkop.numsca
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import play.api.libs.json._

sealed trait Optimizer extends Serializable {
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
  * Adam optimizer
  * @param learningRate
  * @param learningRateDecay
  * @param beta1
  * @param beta2
  * @param epsilon
  */
case class Adam(
    var learningRate: Double,
    learningRateDecay: Double,
    beta1: Double,
    beta2: Double,
    epsilon: Double
) extends Optimizer
    with LazyLogging {
  var t = 1

  var vs = List.empty[Tensor]
  var ss = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    // first time through
    // create the cache
    if (t == 1) {
      vs = parameters.map(numsca.zerosLike)
      ss = parameters.map(numsca.zerosLike)
    }

    t = t + 1

    parameters.indices.foreach { i =>
      update(parameters(i), gradients(i), vs(i), ss(i), t)
    }
  }

  def update(x: Tensor, dx: Tensor, m: Tensor, v: Tensor, t: Int): Unit = {
    m *= beta1
    m += (1 - beta1) * dx
    val mt = m / (1 - math.pow(beta1, t))

    v *= beta2
    v += (1 - beta2) * numsca.square(dx)
    val vt = v / (1 - math.pow(beta2, t))

    x -= learningRate * mt / (numsca.sqrt(vt) + epsilon)
  }

  override def setLearningRate(lr: Double): Unit = {
    logger.info(s"changing learning rate from $learningRate to $lr")
    learningRate = lr
  }

  override def updateLearningRate(): Unit = {
    learningRate *= learningRateDecay
    logger.info(s"learning rate is now $learningRate")
  }
}
object Adam {
  implicit val f: Format[Adam] = Json.format
}

/**
  * Nesterov optimizer
  * @param learningRate
  * @param learningRateDecay
  * @param beta
  */
case class Nesterov(var learningRate: Double,
                    learningRateDecay: Double,
                    beta: Double)
    extends Optimizer with LazyLogging {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    parameters.indices.foreach { i =>
      update(parameters(i), gradients(i), vs(i))
    }
  }

  def update(x: Tensor, dx: Tensor, v: Tensor): Unit = {
    val vPrev = v
    v *= beta
    v -= learningRate * dx
    x += (-beta * vPrev) + (1 + beta) * v
  }

  override def setLearningRate(lr: Double): Unit =
    this.learningRate = lr

  override def updateLearningRate(): Unit = {
    learningRate *= learningRateDecay
    logger.debug(s"learning rate is now $learningRate")
  }
}
object Nesterov {
  implicit val f: Format[Nesterov] = Json.format
}

/**
  * json boilerplate
  */
object Optimizer {
  def reads(json: JsValue): JsResult[Optimizer] = {

    def from(name: String, data: JsObject): JsResult[Optimizer] =
      name match {
        case "Adam" =>
          Json.fromJson[Adam](data)
        case "Nesterov" =>
          Json.fromJson[Nesterov](data)
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
      case b: Adam =>
        (b, Json.toJson(b)(Adam.f))
      case b: Nesterov =>
        (b, Json.toJson(b)(Nesterov.f))
    }
    JsObject(Seq("class" -> JsString(prod.productPrefix), "data" -> sub))
  }

  implicit val r: Reads[Optimizer] = Optimizer.reads
  implicit val w: Writes[Optimizer] = Optimizer.writes
}
