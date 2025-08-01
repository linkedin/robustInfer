package robustinfer

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import robustinfer.TwoSample.{zeroTrimmedU, mwU, tTest, zeroTrimmedUDf, tTestDf}

class TwoSampleTest extends AnyFunSuite with BeforeAndAfterAll with Matchers {
    // 1) shared SparkSession & Context
  lazy val spark: SparkSession = SparkSession.builder()
    .master("local[2]")
    .appName(this.getClass.getSimpleName)
    .config("spark.ui.enabled", "false")
    .getOrCreate()

  implicit lazy val sc: SparkContext = spark.sparkContext
  import spark.implicits._

  override protected def afterAll(): Unit = {
    spark.stop()
    super.afterAll()
  }

  test("zeroTrimmedU on tiny RDD") {
    val x = sc.parallelize(Seq(0.0, 1.0, 2.0))
    val y = sc.parallelize(Seq(0.0, 3.0, 4.0))

    val (z, p, wScaled, (lo, hi)) = TwoSample.zeroTrimmedU(x, y, alpha = 0.05, scale = true)

    // sanity checks
    p should (be >= 0.0 and be <= 1.0)
    assert(java.lang.Double.isFinite(z))
    wScaled should (be >= 0.0 and be <= 1.0)
    lo should be <= wScaled
    hi should be >= wScaled
  }

  test("zeroTrimmedU has smaller variance when adjusted for ties") {
    val x = sc.parallelize(Seq(0.0, 1.0, 2.0, 2.0, 1.0))
    val y = sc.parallelize(Seq(0.0, 3.0, 1.0, 2.0))

    // Without tie correction
    val (z1, _, w1, _) = TwoSample.zeroTrimmedU(x, y, alpha = 0.05, scale = true, tieCorrection = false)

    // With tie correction
    val (z2, _, w2, _) = TwoSample.zeroTrimmedU(x, y, alpha = 0.05, scale = true, tieCorrection = true)

    assert(z1 < z2) // adjusted z should be larger (with smaller variance)
    assert(w1 == w2) // Adjusted U should be the same
  }

  test("compute average ranks") {
    val input = Seq(
      (5.0, true),
      (5.0, false),
      (4.0, true),
      (3.0, false),
      (3.0, true)
    )

    val expectedRanks = Map(
      (5.0, true)   -> 1.5,
      (5.0, false)  -> 1.5,
      (4.0, true)   -> 3.0,
      (3.0, false)  -> 4.5,
      (3.0, true)   -> 4.5
    )

    val rdd = sc.parallelize(input)
    val result = TwoSample.computeAverageRanks(rdd).collect().toMap

    expectedRanks.foreach {
      case (key, expected) =>
        val actual = result.getOrElse(key, Double.NaN)
        assert(
          math.abs(actual - expected) < 1e-6,
          s"[FAIL] For $key: expected $expected but got $actual"
        )
    }
  }

  test("mwU equals zeroTrimmedU when no zeros and scale=false") {
    val a = sc.parallelize(Seq(1.0, 2.0, 3.0))
    val b = sc.parallelize(Seq(4.0, 5.0, 6.0))
    val (_, _, w1, _) = TwoSample.zeroTrimmedU(a, b, scale = false)
    val (_, _, w2, _) = TwoSample.mwU(a, b, scale = false)
    w1 shouldBe w2
  }

  test("tTest returns correct mean difference") {
    val a = sc.parallelize(Seq(1.0, 2.0))
    val b = sc.parallelize(Seq(4.0, 6.0))
    val (_, _, md, _) = TwoSample.tTest(a, b)
    // mean(b)=5, mean(a)=1.5 → diff = 3.5
    md shouldBe (5.0 - 1.5)
  }

  test("zeroTrimmedUDf on DataFrame") {
    val df = Seq(
      ("ctl", 0.0), ("ctl", 1.0),
      ("trt", 0.0), ("trt", 2.0)
    ).toDF("grp","v")

    val (_, p, _, _) =
      TwoSample.zeroTrimmedUDf(df, "grp","v","ctl","trt", alpha = 0.1)

    p should (be >= 0.0 and be <= 1.0)
  }

  test("tTestDf computes expected difference") {
    val df = Seq(
      ("a", 10.0), ("a", 20.0),
      ("b", 30.0), ("b", 40.0)
    ).toDF("grp","v")

    val (_, _, md, _) = TwoSample.tTestDf(df, "grp","v","a","b", alpha=0.05)
    md shouldBe ( (30+40)/2.0 - (10+20)/2.0 )
  }

  test("TwoSample tests on simulated Cauchy data (DataFrame)") {
    import org.apache.commons.math3.distribution.CauchyDistribution
    import scala.util.Random

    val rng = new Random(234)
    val dist1 = new CauchyDistribution(0.0, 1.0)
    val dist2 = new CauchyDistribution(1.0, 1.0)

    // Sample function since Apache doesn't take custom random generator easily
    def sampleCauchy(dist: CauchyDistribution, n: Int): Seq[Double] =
      Seq.fill(n)(dist.sample())

    val cauchy1 = sampleCauchy(dist1, 500).map(v => ("grp1", math.max(0, v)))
    val cauchy2 = sampleCauchy(dist2, 500).map(v => ("grp2", math.max(0, v)))

    val df = (cauchy1 ++ cauchy2).toDF("grp", "v")

    // Run zeroTrimmedUDf test
    val (_, pZeroTrimmed, w, _) = TwoSample.zeroTrimmedUDf(df, "grp", "v", "grp1", "grp2", alpha = 0.05)
    println(s"Zero-Trimmed U Test (DataFrame): p = $pZeroTrimmed")
    println(s"U statistic = $w")
    assert(w > 0.5, "U statistic should be more than 0.5")
    pZeroTrimmed should (be >= 0.0 and be <= 1.0)

    // Run tTestDf test
    val (_, pTTest, meanDiff, _) = TwoSample.tTestDf(df, "grp", "v", "grp1", "grp2", alpha = 0.05)
    println(s"T-Test (DataFrame): p = $pTTest")
    println(s"T-Test (DataFrame): Mean Difference = $meanDiff")
    pTTest should (be >= 0.0 and be <= 1.0)
  }

  test("TwoSample tests on simulated Cauchy data (RDDs)") {
    import org.apache.commons.math3.distribution.CauchyDistribution
    import scala.util.Random

    val rng = new Random(234)
    val dist1 = new CauchyDistribution(0.0, 1.0)
    val dist2 = new CauchyDistribution(1.0, 1.0)

    // Sample function since Apache doesn't take custom random generator easily
    def samplePositiveCauchy(dist: CauchyDistribution, n: Int): Seq[Double] =
      Seq.fill(n)(math.max(0.0, dist.sample()))

    val cauchy1 = samplePositiveCauchy(dist1, 500)
    val cauchy2 = samplePositiveCauchy(dist2, 500)

    val rdd1 = sc.parallelize(cauchy1)
    val rdd2 = sc.parallelize(cauchy2)

    // Run ZeroTrimmedU test
    val (z, pZeroTrimmed, wScaled, (lo, hi)) = TwoSample.zeroTrimmedU(rdd1, rdd2, alpha = 0.05, scale = true)
    println(s"Zero-Trimmed U Test (RDD): z = $z, p = $pZeroTrimmed, wScaled = $wScaled, CI = [$lo, $hi]")
    assert(pZeroTrimmed >= 0.0 && pZeroTrimmed <= 1.0, "p-value should be between 0 and 1")
    assert(wScaled > 0.5, "U statistic should be more than 0.5")

    // Run Mann-Whitney U test
    val (_, _, wMW, _) = TwoSample.mwU(rdd1, rdd2, scale = true)
    println(s"Mann-Whitney U Test (RDD): wMW = $wMW")
    assert(wMW > 0.5, "Mann-Whitney U statistic should be more than 0.5")

    // Run t-test
    val (_, pTTest, meanDiff, _) = TwoSample.tTest(rdd1, rdd2)
    println(s"T-Test (RDD): p = $pTTest, Mean Difference = $meanDiff")
    assert(pTTest >= 0.0 && pTTest <= 1.0, "p-value should be between 0 and 1")
  }
}


