package botkop.nn

import akka.actor.ActorSystem
import botkop.numsca.Tensor

package object gates {

  case class Forward(x: Tensor, y: Tensor)
  object Forward {
    def apply(xy: (Tensor, Tensor)): Forward = {
      Forward(xy._1, xy._2)
    }
  }

  case class Backward(dz: Tensor)
  case class Predict(x: Tensor)

  case object Persist
  case object Quit
  case object Start

  sealed trait Gate {
    def +(other: Gate)(implicit system: ActorSystem): Network =
      Network(List(this, other))
    def *(i: Int)(implicit system: ActorSystem): Network =
      Network(List.fill(i)(this))
  }

  case object Relu extends Gate {
    def name(layer: Int) = s"relu-$layer"
  }

  case object Sigmoid extends Gate {
    def name(layer: Int) = s"sigmoid-$layer"
  }

  case object Linear extends Gate {
    def name(layer: Int) = s"linear-$layer"
  }

  case object Dropout extends Gate {
    def name(layer: Int) = s"dropout-$layer"
  }

}
