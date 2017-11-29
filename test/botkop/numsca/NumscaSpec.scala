package botkop.numsca

import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

import scala.language.postfixOps

class NumscaSpec extends FlatSpec with Matchers {

  val ta: Tensor = ns.arange(10)
  val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
  val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)

  "A Tensor" should "transpose over multiple dimensions" in {
    val x = ns.arange(6).reshape(1, 2, 3)
    val y = ns.transpose(x, 1, 0, 2)
    val z = ns.reshape(x, 2, 1, 3)
    assert(ns.arrayEqual(y, z))
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


    // todo: this is extremely slow

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
    assert(ns.arrayEqual(ta(:>, -3 :>), Tensor(7, 8, 9)))

  }

  it should "update over a single dimension" in {
    val t = ta.copy()
    t(2 :> 5) := -ns.ones(3)
    val e1 =
      Tensor(0.00, 1.00, -1.00, -1.00, -1.00, 5.00, 6.00, 7.00, 8.00, 9.00)
    assert(ns.arrayEqual(t, e1))

    an[IllegalStateException] should be thrownBy {
      t(2 :> 5) := -ns.ones(4)
    }

    // this does not throw an exception !!!
    /*
    an[IllegalStateException] should be thrownBy {
      t(2 :> 6) := -ns.ones(4).reshape(2, 2)
    }
     */

    t(2 :> 5) := 33
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, 33.00, 33.00, 33.00, 5.00, 6.00, 7.00, 8.00, 9.00)))

    t(2 :> 5) -= 1
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, 32.00, 32.00, 32.00, 5.00, 6.00, 7.00, 8.00, 9.00)))

    t := -1
    assert(
      ns.arrayEqual(t,
                    Tensor(-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00,
                      -1.00, -1.00, -1.00)))

    val s = 3 :> -1
    assert(ns.arrayEqual(ta(:>, s), Tensor(3.00, 4.00, 5.00, 6.00, 7.00, 8.00)))

  }

  it should "broadcast with another tensor" in {

    // tests inspired by
    // https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    // http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc

    def verify(shape1: Array[Int],
               shape2: Array[Int],
               expectedShape: Array[Int]) = {
      val t1 = ns.ones(shape1)
      val t2 = ns.ones(shape2)
      val (s1, s2) = Ops.tbc(t1, t2)
      assert(s1.shape().sameElements(s2.shape()))
      assert(s1.shape().sameElements(expectedShape))
    }

    verify(Array(8, 1, 6, 1), Array(7, 1, 5), Array(8, 7, 6, 5))
    verify(Array(256, 256, 3), Array(3), Array(256, 256, 3))
    verify(Array(5, 4), Array(1), Array(5, 4))
    verify(Array(15, 3, 5), Array(15, 1, 5), Array(15, 3, 5))
    verify(Array(15, 3, 5), Array(3, 5), Array(15, 3, 5))
    verify(Array(15, 3, 5), Array(3, 1), Array(15, 3, 5))

    val x = ns.arange(4)
    val xx = x.reshape(4, 1)
    val y = ns.ones(5)
    val z = ns.ones(3, 4)

    an[IllegalArgumentException] should be thrownBy x + y

    (xx + y).shape shouldBe Array(4, 5)
    val s1 =
      Tensor(
        1, 1, 1, 1, 1, //
        2, 2, 2, 2, 2, //
        3, 3, 3, 3, 3, //
        4, 4, 4, 4, 4 //
      ).reshape(4, 5)
    assert(ns.arrayEqual(xx + y, s1))

    (x + z).shape shouldBe Array(3, 4)
    val s2 =
      Tensor(
        1, 2, 3, 4, //
        1, 2, 3, 4, //
        1, 2, 3, 4 //
      ).reshape(3, 4)
    assert(ns.arrayEqual(x + z, s2))

    // outer sum
    val a = Tensor(0.0, 10.0, 20.0, 30.0).reshape(4, 1)
    val b = Tensor(1.0, 2.0, 3.0)
    val c = Tensor(
      1.00, 2.00, 3.00, //
      11.00, 12.00, 13.00, //
      21.00, 22.00, 23.00, //
      31.00, 32.00, 33.00 //
    ).reshape(4, 3)

    assert(ns.arrayEqual(a + b, c))

    val observation = Tensor(111.0, 188.0)
    val codes = Tensor(
      102.0, 203.0, //
      132.0, 193.0, //
      45.0, 155.0, //
      57.0, 173.0 //
    ).reshape(4, 2)
    val diff = codes - observation
    val dist = ns.sqrt(ns.sum(ns.square(diff), axis = -1))
    val nearest = ns.argmin(dist).squeeze()
    assert(nearest == 0.0)
  }

  it should "do boolean indexing" in {
    val c = ta < 5 && ta > 1
    // c = [0.00,  0.00,  1.00,  1.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00]
    val d = ta(c)
    assert(ns.arrayEqual(d, Tensor(2, 3, 4)))

    val t = ta.copy()
    // this does not work, beuhueueue
    t(c) := -7 // has no effect, since selection is a new tensor, not a view
    // but this does
    t.put(-7)(c)
    assert(
      ns.arrayEqual(
        t,
        Tensor(0.00, 1.00, -7.00, -7.00, -7.00, 5.00, 6.00, 7.00, 8.00, 9.00)))
  }

  it should "do multidimensional boolean indexing" in {
    val c = tc(tc % 5 == 0)
    assert(ns.arrayEqual(c, Tensor(0.00, 5.00, 10.00, 15.00, 20.00)))
    assert(ns.any(tb < 5))
    assert(!ns.all(tb < 5))

    // stuff like this is not yet implemented
    // np.any(B<5, axis=1)
    // B[np.any(B<5, axis=1),:]
    // c = np.any(C<5,axis=2)
  }

  it should "do list-of-location indexing" in {

    def index(t: Tensor, selection: Tensor): Array[Float] = {
      val ta = t.array
      val ts = selection.array
      ts.data().asInt().map(i => ta.getFloat(i))
    }

    val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
    val idx = Tensor(3, 4, 1, 2, 2)
    val r = Tensor(index(primes, idx)).shapeLike(idx)
    println(r)


    val r2 = Tensor(index(primes, tb)).shapeLike(tb)
    println(r2)

    val tp = primes.reshape(3, 3)
    val r3 = Tensor(index(tp, tb)).shapeLike(tb)
    println(r3)

  }

  it should "do multi dim list-of-location indexing" in {

    def multiIndex(t: Tensor, selection: Tensor*): Tensor = {
      val rank = selection.head.shape(1)
      require(selection.forall(s => s.shape.head == 1 && s.shape(1) == rank))

      val data = (0 until rank).map { r =>
        val idx = selection.map( s => s.array.getInt(0, r)).toArray
        t.array.getFloat(idx)
      }.toArray

      Tensor(data).reshape(1, rank)
    }

    val a = ns.arange(6).reshape(3, 2) + 1
    val s1 = Tensor(0, 1, 2)
    val s2 = Tensor(0, 1, 0)

    val r1 = multiIndex(a, s1, s2)
    println(r1)

    val s3 = Tensor(0, 0)
    val s4 = Tensor(1, 1)

    val r2 = multiIndex(a, s3, s4)
    println(r2)

    val b = ns.arange(12).reshape(4, 3) + 1
    println(b)
    val c = Tensor(0, 2, 0, 1)

    println(multiIndex(b, arange(4), c))

  }

}
