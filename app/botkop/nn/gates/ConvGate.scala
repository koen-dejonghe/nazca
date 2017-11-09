package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca._
import botkop.{numsca => ns}
import org.nd4j.linalg.factory.Nd4j.PadMode

class ConvGate(next: ActorRef, config: ConvConfig)
    extends Actor
    with ActorLogging {

  // import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = {
    case Forward(x, y) =>
  }
}

object ConvGate {

  def convForward(x: Tensor,
                  w: Tensor,
                  b: Tensor,
                  config: ConvConfig): Tensor = {
    import config._

    val Array(samples, _, height, width) = x.shape
    val Array(filters, _, hh, ww) = w.shape
    val hPrime = 1 + (height + 2 * pad - hh) / stride
    val wPrime = 1 + (width + 2 * pad - ww) / stride
    val out = ns.zeros(samples, filters, hPrime, wPrime)

    for (n <- 0 until samples) {
      val xPad = ns.pad(x.slice(n),
                        Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
                        PadMode.CONSTANT)
      for (f <- 0 until filters) {
        for (hp <- 0 until hPrime) {
          for (wp <- 0 until wPrime) {
            val h1 = hp * stride
            val h2 = h1 + hh
            val w1 = wp * stride
            val w2 = w1 + ww
            val window = xPad(:>, h1 :> h2, w1 :> w2)
            val v = ns.sum(window * w.slice(f)) + b.squeeze(f, 0)
            out(n, f, hp, wp) := v
          }
        }
      }
    }
    out
  }

  def convBackward(dout: Tensor,
                   x: Tensor,
                   w: Tensor,
                   b: Tensor,
                   config: ConvConfig): (Tensor, Tensor, Tensor) = {
    val Array(numSamples, _, _, _) = x.shape
    val Array(filters, _, hh, ww) = w.shape
    val Array(_, _, h_prime, w_prime) = dout.shape
    import config._

    val dx = ns.zerosLike(x)
    val dw = ns.zerosLike(w)
    val db = ns.zerosLike(b)

    for (n <- 0 until numSamples) {
      val dxPad =
        ns.pad(dx.slice(n),
          Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
          PadMode.CONSTANT)

      val xPad =
        ns.pad(x.slice(n),
          Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
          PadMode.CONSTANT)

      for (f <- 0 until filters) {
        for (hp <- 0 until h_prime) {
          for (wp <- 0 until w_prime) {
            val h1 = hp * stride
            val h2 = h1 + hh
            val w1 = wp * stride
            val w2 = w1 + ww

            val z = dout.squeeze(n, f, hp, wp)

            dxPad(:>, h1 :> h2, w1 :> w2) += w.slice(f) * z
            dw.slice(f) += xPad(:>, h1 :> h2, w1 :> w2) * z
            db(f, 0) += z
          }
        }
      }
      dx.slice(n) := dxPad(:>, 1 :> -1, 1 :> -1)
    }
    (dx, dw, db)
  }

  def props(next: ActorRef, config: ConvConfig) =
    Props(new ConvGate(next, config))
}

case class ConvConfig(stride: Int, pad: Int)
