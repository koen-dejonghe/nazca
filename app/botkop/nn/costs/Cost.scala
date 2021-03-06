package botkop.nn.costs

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import play.api.libs.json._

trait Cost {
  def costFunction(x: Tensor, y: Tensor): (Double, Tensor)
}

object Cost extends LazyLogging {
  def reads(json: JsValue): JsResult[Cost] = {

    def from(name: String): JsResult[Cost] =
      name match {
        case "CrossEntropy" =>
          JsSuccess(CrossEntropy)
        case "Softmax" =>
          JsSuccess(Softmax)
        case _ =>
          logger.error(s"unknown class $name")
          JsError(s"Unknown class '$name'")
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

