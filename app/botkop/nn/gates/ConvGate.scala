package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca._
import botkop.{numsca => ns}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import org.nd4j.linalg.indexing.NDArrayIndex

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
            val v = ns.sum(window * w.slice(f)) + b(f)
            val index = Array(n, f, hp, wp)
            out.put(index, v)
          }
        }
      }
    }
    out
  }

  /*
  def convBackward(dout: Tensor,
                   x: Tensor,
                   w: Tensor,
                   b: Tensor,
                   config: ConvConfig): (Tensor, Tensor, Tensor) = {
    val Array(n, channels, height, width) = x.shape
    val Array(filters, _, hh, ww) = w.shape
    val Array(_, _, h_prime, w_prime) = dout.shape
    import config._

    val dx = ns.zerosLike(x)
    val dw = ns.zerosLike(w)
    val db = ns.zerosLike(b)

    for (n <- 0 until n) {
      val dxPad =
        Nd4j.pad(dx.array.slice(n),
                 Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
                 Nd4j.PadMode.CONSTANT)

      val xPad =
        Nd4j.pad(x.array.slice(n),
                 Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
                 Nd4j.PadMode.CONSTANT)
      for (f <- 0 until filters) {
        for (hp <- 0 until h_prime) {
          for (wp <- 0 until w_prime) {
            val h1 = hp * stride
            val h2 = hp * stride + hh
            val w1 = wp * stride
            val w2 = wp * stride + ww

            val index = Array(n, f, hp, wp)
            val z = dout(index)
            val v = w.array.slice(f).mul(z)

            dxPad
              .get(NDArrayIndex.all(),
                   NDArrayIndex.interval(h1, h2),
                   NDArrayIndex.interval(w1, w2))
              .addi(v)

            val v2 = xPad
              .get(NDArrayIndex.all(),
                   NDArrayIndex.interval(h1, h2),
                   NDArrayIndex.interval(w1, w2))
              .mul(z)

            dw.array.slice(f).addi(v2)

            db.array.slice(f).addi(z)
          }
        }
      }

      val shape = dxPad.shape()

      dx.array
        .slice(n)
        .assign(
          dxPad.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(1, shape(1) - 1),
                    NDArrayIndex.interval(1, shape(2) - 1)))
    }

    (dx, dw, db)

  }

  */

  def convBackward(dout: Tensor,
                   x: Tensor,
                   w: Tensor,
                   b: Tensor,
                   config: ConvConfig): (Tensor, Tensor, Tensor) = {
    val Array(n, _, _, _) = x.shape
    val Array(filters, _, hh, ww) = w.shape
    val Array(_, _, h_prime, w_prime) = dout.shape
    import config._

    val dx = ns.zerosLike(x)
    val dw = ns.zerosLike(w)
    val db = ns.zerosLike(b)

    for (n <- 0 until n) {
      val dxPad =
        ns.pad(dx.slice(n),
          Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
          PadMode.CONSTANT)

      val xPad =
        ns.pad(x.slice(n),
          Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
          Nd4j.PadMode.CONSTANT)

      for (f <- 0 until filters) {
        for (hp <- 0 until h_prime) {
          for (wp <- 0 until w_prime) {
            val h1 = hp * stride
            val h2 = h1 + hh
            val w1 = wp * stride
            val w2 = w1 + ww

            val index = Array(n, f, hp, wp)
            val z = dout(index)
            val v = w.slice(f) * z

            dxPad(:>, h1 :> h2, w1 :> w2) += v
            dw.slice(f) += xPad(:>, h1 :> h2, w1 :> w2) * z
            db.slice(f) += z

            /*
            dxPad
              .get(NDArrayIndex.all(),
                NDArrayIndex.interval(h1, h2),
                NDArrayIndex.interval(w1, w2))
              .addi(v)

            val v2 = xPad
              .get(NDArrayIndex.all(),
                NDArrayIndex.interval(h1, h2),
                NDArrayIndex.interval(w1, w2))
              .mul(z)

            dw.array.slice(f).addi(v2)

            db.array.slice(f).addi(z)
              */
          }
        }
      }

      /*
      val shape = dxPad.shape()

      dx.array
        .slice(n)
        .assign(
          dxPad.get(NDArrayIndex.all(),
            NDArrayIndex.interval(1, shape(1) - 1),
            NDArrayIndex.interval(1, shape(2) - 1)))
      */

      val shape = dxPad.shape
      dx.slice(n) := dxPad(:>, 1 :> (shape(1)-1), 1 :> (shape(2)-1))

    }

    (dx, dw, db)

  }



  def props(next: ActorRef, config: ConvConfig) =
    Props(new ConvGate(next, config))
}

case class ConvConfig(stride: Int, pad: Int)
