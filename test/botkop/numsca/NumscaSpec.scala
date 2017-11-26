package botkop.numsca

import org.scalatest.{FlatSpec, Matchers}
import botkop.{numsca => ns}

import scala.language.postfixOps

class NumscaSpec extends FlatSpec with Matchers {

  val ta: Tensor = ns.arange(10)
  val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
  val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)

  "A Tensor" should "transpose over multiple dimensions" in {
    val x = ns.arange(6).reshape(1, 2, 3)
    val y = ns.transpose(x, 1, 0, 2)
    val z = ns.reshape(x, 2, 1, 3)
    val b = y == z
    assert(ns.prod(b) == 1.0)
  }

  // tests based on http://scipy-cookbook.readthedocs.io/items/Indexing.html

  // Elements
  it should "retrieve the correct elements" in {
    // todo: implicitly convert tensor to double when only 1 element?
    assert(ta(1).squeeze() == 1)
    assert(tb(1, 0).squeeze() == 3)
    assert(tc(1, 0, 2).squeeze() == 14)

    val i = List(1, 0, 1)
    assert(tc(i: _*).squeeze() == 13)

  }

  it should "access through an iterator" in {

    val expected = List(
      (Array(0, 0), 0.00),
      (Array(0, 1), 1.00),
      (Array(0, 2), 2.00),
      (Array(1, 0), 3.00),
      (Array(1, 1), 4.00),
      (Array(1, 2), 5.00),
      (Array(2, 0), 6.00),
      (Array(2, 1), 7.00),
      (Array(2, 2), 8.00)
    )

    ns.nditer(tb.shape).zipWithIndex.foreach {
      case (i1, i2) =>
        assert(i1 sameElements expected(i2)._1)
        assert(tb(i1).squeeze() == expected(i2)._2)
    }
  }

  it should "change array values in place" in {
    val t = ta.copy()
    t(3) := -5
    assert(t.data sameElements Array(0, 1, 2, -5, 4, 5, 6, 7, 8, 9))
    t(0) += 7
    assert(t.data sameElements Array(7, 1, 2, -5, 4, 5, 6, 7, 8, 9))
  }

  it should "do operations array-wise" in {
    val a2 = 2 * ta
    assert(a2.data sameElements Array(0, 2, 4, 6, 8, 10, 12, 14, 16, 18))
  }

  it should "slice over a single dimension" in {
    println(ta.shape.toList)

    // turn into a column vector
    val a0 = ta.copy().reshape(10, 1)

    // A[1:]
    val a1 = a0(1 :>)

    // A[:-1]
    val a2 = a0(0 :> -1)

    // A[1:] - A[:-1]
    val a3 = a1 - a2

    assert(
      a3.data sameElements Array(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.00))

    assert(ns.arrayEqual(ta(:>, 5 :>), Tensor(5, 6, 7, 8, 9)))
    assert(ns.arrayEqual(ta(:>, :>(5)), Tensor(0, 1, 2, 3, 4)))

    // assert(ns.arrayEqual(ta(:>, -3 :>), Tensor(7, 8, 9)))

  }

}
