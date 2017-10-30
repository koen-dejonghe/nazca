package botkop.nn.templates

import botkop.numsca
import botkop.numsca.Tensor
import play.api.libs.json._

sealed trait Cost {
  def costFunction(x: Tensor, y: Tensor): (Double, Tensor)
}

case object CrossEntropy extends Cost {
  override def costFunction(yHat: Tensor, y: Tensor): (Double, Tensor) = {
    val m = y.shape(1)
    val cost = (-y.dot(numsca.log(yHat).T) -
      (1 - y).dot(numsca.log(1 - yHat).T)) / m

    val dal = -(y / yHat - (1 - y) / (1 - yHat))
    (cost.squeeze(), dal)
  }
}

case object Softmax extends Cost {
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

object Cost {
  def reads(json: JsValue): JsResult[Cost] = {

    def from(name: String): JsResult[Cost] =
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

  def writes(foo: Cost): JsValue = {
    JsObject(Seq("class" -> JsString(foo.asInstanceOf[Product].productPrefix)))
  }

  implicit val r: Reads[Cost] = Cost.reads
  implicit val w: Writes[Cost] = Cost.writes
}

