package botkop.nn

import botkop.numsca._

object TestUtil {

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray(f: (Tensor) => Tensor,
                                 x: Tensor,
                                 df: Tensor,
                                 h: Double = 1e-5): Tensor = {
    val grad = zeros(x.shape)
    val it = nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x(ix)

      x(ix) := oldVal + h
      val pos = f(x)
      x(ix) := oldVal - h
      val neg = f(x)

      x(ix) := oldVal
      val g = sum((pos - neg) * df) / (2.0 * h)
      // grad.put(ix, g)
      grad(ix) := g
    }
    grad
  }

  /**
  a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (array) to evaluate the gradient at
    */
  def evalNumericalGradient(f: (Tensor) => Double,
                            x: Tensor,
                            h: Double = 0.00001): Tensor = {
    val grad = zeros(x.shape)
    val it = nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x(ix).squeeze()

      x(ix) := oldVal + h
      val pos = f(x)

      x(ix) := oldVal - h
      val neg = f(x)

      x(ix) := oldVal

      val g = (pos - neg) / (2.0 * h)
      grad(ix) := g
    }
    grad
  }

  /**
    * returns relative error
    */
  def relError(x: Tensor, y: Tensor): Double = {
    val n = abs(x - y)
    val d = maximum(abs(x) + abs(y), 1e-8)
    max(n / d).squeeze()
  }

}
