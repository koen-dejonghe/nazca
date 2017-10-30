package botkop.nn

import akka.actor.ActorSystem
import botkop.numsca.Tensor
import play.api.libs.json.{Json, OFormat}

package object gates {

  case class Forward(x: Tensor, y: Tensor)
  object Forward {
    def apply(xy: (Tensor, Tensor)): Forward = {
      Forward(xy._1, xy._2)
    }
  }
  case class Backward(dz: Tensor)

  case class Eval(source: String, id: Int, x: Tensor, y: Tensor)
  object Eval {
    def apply(source: String, id: Int, xy: (Tensor, Tensor)): Eval = {
      Eval(source, id, xy._1, xy._2)
    }
  }

  case class Predict(x: Tensor)

  case object Persist
  case object Quit
  case object Start
  case object Pause
  case object NextBatch
  case class Epoch(epoch: Int, ts: Long = System.currentTimeMillis()) {
    def inc = Epoch(epoch + 1)
  }

  case class CostLogEntry(source: String, id: Int, cost: Double)
  object CostLogEntry {
    implicit val f: OFormat[CostLogEntry] = Json.format[CostLogEntry]
  }

  case class EvalEntry(source: String, id: Int, cost: Double, accuracy: Double)
  object EvalEntry {
    implicit val f: OFormat[EvalEntry] = Json.format
  }

  case class SetLearningRate(lr: Double)

  sealed trait Gate {
    def +(other: Gate)(implicit system: ActorSystem,
                       projectName: String): Network =
      Network(List(this, other))
    def *(i: Int)(implicit system: ActorSystem, projectName: String): Network =
      Network(List.fill(i)(this))
    def category: String
    def name(layer: Int)(implicit projectName: String) =
      s"${projectName}_$category-$layer"
  }

  case object Relu extends Gate {
    override val category = "relu"
  }

  case object Sigmoid extends Gate {
    override val category = "sigmoid"
  }

  case object Linear extends Gate {
    override val category = "linear"
  }

  case object Dropout extends Gate {
    override val category = "dropout"
  }

  case object BatchNorm extends Gate {
    override val category = "batchnorm"
  }
}
