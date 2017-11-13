package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.nn.optimizers.Optimizer
import botkop.numsca._
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.factory.Nd4j.PadMode

class ConvGate(next: ActorRef, config: ConvConfig)
    extends Actor
    with ActorLogging {

  import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  ns.rand.setSeed(seed)

  log.debug(self.path.toString)

  // todo: create Initializer classes
  val w: Tensor = weightScale * ns.randn(numFilters,
                                         numChannels,
                                         filterSize,
                                         filterSize)
  val b: Tensor = weightScale * ns.zeros(numFilters, 1)

  override def receive: Receive = {
    case Forward(x, y) =>
      log.debug("received msg")
      val xt = x.transpose // num samples as first dimension
      val numSamples = xt.shape.head
      val xta = xt.reshape(numSamples, numChannels, height, width)

      val z = ConvGate.convForward(xta, w, b, stride, pad)
      log.debug("sending response")
      next ! Forward(z, y)

  }
}

object ConvGate extends LazyLogging {

  def convForward(x: Tensor,
                  w: Tensor,
                  b: Tensor,
                  stride: Int,
                  pad: Int): Tensor = {

    val Array(samples, _, height, width) = x.shape
    val Array(filters, _, hh, ww) = w.shape
    val hPrime = 1 + (height + 2 * pad - hh) / stride
    val wPrime = 1 + (width + 2 * pad - ww) / stride
    val out = ns.zeros(samples, filters, hPrime, wPrime)

    val l = List.range(0, filters * hPrime * wPrime).par
    println(l.size)

    for (n <- (0 until samples).toList) {
      val xPad = ns.pad(x.slice(n),
                        Array(Array(0, 0), Array(pad, pad), Array(pad, pad)),
                        PadMode.CONSTANT)

      for (i <- l) {
        val f = (i / (hPrime * wPrime)) % filters
        val hp = (i / wPrime) % hPrime
        val wp = i % wPrime

        val h1 = hp * stride
        val h2 = h1 + hh
        val w1 = wp * stride
        val w2 = w1 + ww
        val window = xPad(:>, h1 :> h2, w1 :> w2)
        val v = ns.sum(window * w.slice(f)) + b.squeeze(f, 0)
        out(n, f, hp, wp) := v
      }
    }

    out
  }

  def convBackward(dout: Tensor,
                   x: Tensor,
                   w: Tensor,
                   b: Tensor,
                   stride: Int,
                   pad: Int): (Tensor, Tensor, Tensor) = {
    val Array(numSamples, _, _, _) = x.shape
    val Array(channels, _, hh, ww) = w.shape
    val Array(_, _, h_prime, w_prime) = dout.shape

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

      for (f <- 0 until channels) {
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

  /*
  // maybe use org.nd4j.linalg.convolution.DefaultConvolutionInstance.convn
  def convForwardIm2Col(x: Tensor,
                        w: Tensor,
                        b: Tensor,
                        stride: Int,
                        pad: Int): Tensor = {
    val Array(n, num_channels, height, width) = x.shape
    val Array(num_filters, _, filter_height, filter_width) = w.shape

    assert ((width + 2 * pad - filter_width) % stride == 0, "width does not work")
    assert ((height + 2 * pad - filter_height) % stride == 0, "height does not work")

    // Create output
    val out_height = (height + 2 * pad - filter_height) / stride + 1
    val out_width = (width + 2 * pad - filter_width) / stride + 1

    println(w.shape.toList)

    val zzz = new Tensor(Convolution.im2col(x.array, height, width, stride, stride, pad, pad, true))
    println(zzz.shape.toList)

    val x_cols = zzz
    println(x_cols.shape.toList)

    val res = w.reshape(w.shape.head, w.shape.tail.product).dot(x_cols) + b

    val out = res.reshape(w.shape.head, out_height, out_width, x.shape.head)
    out.transpose

  }
   */

  def props(next: ActorRef, config: ConvConfig) =
    Props(new ConvGate(next, config))
}

case class ConvConfig(numChannels: Int,
                      height: Int,
                      width: Int,
                      numFilters: Int,
                      filterSize: Int,
                      weightScale: Double,
                      regularization: Double,
                      optimizer: Optimizer,
                      stride: Int,
                      pad: Int,
                      seed: Long)
