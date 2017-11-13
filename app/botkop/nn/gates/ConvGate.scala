package botkop.nn.gates

import akka.actor.{Actor, ActorContext, ActorLogging, ActorRef, Props}
import botkop.nn.optimizers.Optimizer
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import ns._
import org.nd4j.linalg.factory.Nd4j.PadMode
import play.api.libs.json.{Format, Json}

class ConvGate(next: ActorRef, conf: ConvConfig)
    extends Actor
    with ActorLogging {

  import ConvGate._
  import conf._

  ns.rand.setSeed(seed)
  val w: Tensor = ns.randn(f, f, nCprev, nC)
  val b: Tensor = ns.zeros(1, 1, 1, nC)

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = listen()

  def listen(cache: Option[(ActorRef, Tensor)] = None): Receive = {

    case Forward(x, y) =>
      val xm = mould(x, nHprev, nWprev, nCprev) // should be done by the minibatcher
      val z = convForward(xm, w, b, stride, pad)
      val zm = z.reshape(z.shape.head, z.shape.tail.product).transpose // should be done by a special flattener
      next ! Forward(zm, y)
      context become listen(Some(sender(), xm))

    case Backward(dz) if cache.isDefined =>
      val (prev, a) = cache.get

      val nH = ((nHprev - f + 2 * pad) / stride) + 1
      val nW = ((nWprev - f + 2 * pad) / stride) + 1
      val dzm = dz.reshape(dz.shape.last, nH, nW, nC)

      val (da, dw, db) = convBackward(dzm, a, w, b, stride, pad)
      prev ! Backward(da)

      // adjusting regularization, if needed
      if (regularization != 0)
        dw += regularization * w

      optimizer.update(List(w, b), List(dw, db))

    case Eval(source, id, x, y) =>
      val z = convForward(x, w, b, stride, pad)
      next forward Eval(source, id, z, y)
  }

}

object ConvGate {

  def mould(t: Tensor, nHprev: Int, nWprev: Int, nCprev: Int): Tensor = {
    val m = t.shape.last
    t.transpose.reshape(m, nHprev, nWprev, nCprev)
  }

  /**
    * Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    * as illustrated in Figure 1.
    *
    * Argument:
    * X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    * pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    *
    * Returns:
    * X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    */
  def zeroPad(x: Tensor, pad: Int): Tensor = {
    ns.pad(x,
           Array(Array(0, 0), Array(pad, pad), Array(pad, pad), Array(0, 0)),
           mode = PadMode.CONSTANT)
  }

  /**
    * Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    * of the previous layer.

    * Arguments:
    * a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    * W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    * b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    */
  def convSingleStep(aSlicePrev: Tensor,
                     weights: Tensor,
                     bias: Tensor): Double = {

    // Element-wise product between a_slice and W. Do not add the bias yet.
    val s = weights * aSlicePrev

    // Sum over all entries of the volume s.
    // Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    ns.sum(s) + bias.squeeze()
  }

  /**
    * Implements the forward propagation for a convolution function

    * Arguments:
    * A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    * W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    * b -- Biases, numpy array of shape (1, 1, 1, n_C)
    * hparameters -- python dictionary containing "stride" and "pad"

    * Returns:
    * Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    * cache -- cache of values needed for the conv_backward() function
    */
  def convForward(aPrev: Tensor,
                  weights: Tensor,
                  biases: Tensor,
                  stride: Int,
                  pad: Int): Tensor = {

    // Retrieve dimensions from A_prev's shape
    val Array(m, nHprev, nWprev, nCprev) = aPrev.shape

    // Retrieve dimensions from W's shape
    val Array(f, _, _, nC) = weights.shape

    // Compute the dimensions of the CONV output volume using the formula given above.
    val nH = ((nHprev - f + 2 * pad) / stride) + 1
    val nW = ((nWprev - f + 2 * pad) / stride) + 1

    // Initialize the output volume Z with zeros
    val z = ns.zeros(m, nH, nW, nC)

    // Create A_prev_pad by padding A_prev
    val aPrevPad = zeroPad(aPrev, pad)

    for (i <- 0 until m) { // loop over the batch of training examples

      val app: Tensor = aPrevPad.slice(i) // Select ith training example's padded activation

      for (h <- 0 until nH) { // loop over vertical axis of the output volume
        for (w <- 0 until nW) { // loop over horizontal axis of the output volume
          for (c <- 0 until nC) { // loop over channels (= #filters) of the output volume

            // # Find the corners of the current "slice"
            val vertStart = h * stride
            val vertEnd = vertStart + f
            val horizStart = w * stride
            val horizEnd = horizStart + f

            // Use the corners to define the (3D) slice of a_prev_pad
            val aSlicePrev =
              app(vertStart :> vertEnd, horizStart :> horizEnd, :>)

            // println(aSlicePrev.shape.toList)

            // Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
            z(i, h, w, c) := convSingleStep(aSlicePrev,
                                            weights(:>, :>, :>, c :> c + 1),
                                            biases(:>, :>, :>, c :> c + 1))
          }
        }
      }
    }

    // Making sure your output shape is correct
    assert(z.shape sameElements Array(m, nH, nW, nC))

    z
  }

  /**
    * Implement the backward propagation for a convolution function

    * Arguments:
    * dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    * cache -- cache of values needed for the conv_backward(), output of conv_forward()

    * Returns:
    * dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
    *            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    * dW -- gradient of the cost with respect to the weights of the conv layer (W)
    *       numpy array of shape (f, f, n_C_prev, n_C)
    * db -- gradient of the cost with respect to the biases of the conv layer (b)
    *       numpy array of shape (1, 1, 1, n_C)
    */
  def convBackward(dZ: Tensor,
                   aPrev: Tensor,
                   weights: Tensor,
                   biases: Tensor,
                   stride: Int,
                   pad: Int): (Tensor, Tensor, Tensor) = {

    // Retrieve dimensions from A_prev's shape
    val Array(m, nHprev, nWprev, nCprev) = aPrev.shape

    // Retrieve dimensions from W's shape
    val Array(f, _, _, nC) = weights.shape

    // Retrieve dimensions from dZ's shape
    val Array(_, nH, nW, _) = dZ.shape

    // Initialize dA_prev, dW, db with the correct shapes
    val dAprev = ns.zeros(aPrev.shape)
    val dW = ns.zeros(weights.shape)
    val db = ns.zeros(biases.shape)

    // Pad A_prev and dA_prev
    val aPrevPad = zeroPad(aPrev, pad)
    val dAPrevPad = zeroPad(dAprev, pad)

    for (i <- 0 until m) { // loop over the training examples

      // select ith training example from A_prev_pad and dA_prev_pad
      val app = aPrevPad.slice(i)
      val dapp = dAPrevPad.slice(i)

      for (h <- 0 until nH) { // loop over vertical axis of the output volume
        for (w <- 0 until nW) { // loop over horizontal axis of the output volume
          for (c <- 0 until nC) { // loop over the channels of the output volume

            // Find the corners of the current "slice"
            val vertStart = h * stride
            val vertEnd = vertStart + f
            val horizStart = w * stride
            val horizEnd = horizStart + f

            // Use the corners to define the slice from a_prev_pad
            val aSlice = app(vertStart :> vertEnd, horizStart :> horizEnd, :>)

            // Update gradients for the window and the filter's parameters using the code formulas given above

            dapp(vertStart :> vertEnd, horizStart :> horizEnd, :>) +=
              weights(:>, :>, :>, c :> c + 1).reshape(weights.shape.init) * dZ(i, h, w, c)

            dW(:>, :>, :>, c :> c + 1).reshape(dW.shape.init) += aSlice * dZ(i, h, w, c)
            db(:>, :>, :>, c :> c + 1) += dZ(i, h, w, c)
          }
        }
      }

      dAprev.slice(i) := dapp(pad :> -pad, pad :> -pad, :>)

    }

    // Making sure your output shape is correct
    assert(dAprev.shape sameElements Array(m, nHprev, nWprev, nCprev))

    (dAprev, dW, db)
  }

  def props(next: ActorRef, conf: ConvConfig): Props =
    Props(new ConvGate(next, conf))
}

case class ConvConfig(nHprev: Int,
                      nWprev: Int,
                      nCprev: Int,
                      nC: Int,
                      f: Int,
                      stride: Int,
                      pad: Int,
                      regularization: Double,
                      optimizer: Optimizer,
                      seed: Long)
    extends GateConfig {

  override def materialize(next: Option[ActorRef], index: Int)(
      implicit context: ActorContext,
      projectName: String): ActorRef = {
    context.actorOf(ConvGate.props(next.get, this), Conv.name(index))
  }
}

object ConvConfig {
  implicit val f: Format[ConvConfig] = Json.format
}
