package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import breeze.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import scala.util.Random

class GEETest extends AnyFunSuite with BeforeAndAfterAll {

  lazy val spark: SparkSession = SparkSession
    .builder()
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

  test("GEE recovers true beta on synthetic data") {

    val rand = new Random(123)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, k) => xi * trueBeta(k) }.sum
        val prob = 1.0 / (1.0 + math.exp(-eta))
        val y = if (rand.nextDouble() < prob) 1.0 else 0.0
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)

    val gee = new GEE()
    val betaHat = gee.fit(df, maxIter = 10)
    val summary = gee.result()

    println(s"True beta: $trueBeta")
    println(s"Estimated beta: ${summary.coef}")
    println(s"Estimated variance:\n${summary.variance}")

    assert(norm(summary.coef - trueBeta) < 0.2, "Estimated beta should be close to true beta")
  }

  test("GEE handles within-cluster correlation and recovers true beta") {

    val rand = new Random(456)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.2 // induce correlation
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, k) => xi * trueBeta(k) }.sum + clusterEffect
        val prob = 1.0 / (1.0 + math.exp(-eta))
        val y = if (rand.nextDouble() < prob) 1.0 else 0.0
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)

    val gee = new GEE()
    val betaHat = gee.fit(df, maxIter = 10)
    val summary = gee.result()

    println(s"True beta: $trueBeta")
    println(s"Estimated beta: ${summary.coef}")
    println(s"Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.3,
      "Estimated beta should be close to true beta under correlation"
    )
  }

  test("GEE handles cluster size = 1 for Gaussian outcomes") {
    val rand = new Random(789)
    val trueBeta = DenseVector(0.0, 1.5, -2.0)
    val nClusters = 1000
    val obsPerCluster = 1 // Cluster size is 1

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val y =
          x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + rand.nextGaussian() * 0.3
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Gaussian)
    gee.fit(df, maxIter = 10)
    val summary = gee.result()

    println(s"[Gaussian - Cluster Size 1] True beta: $trueBeta")
    println(s"[Gaussian - Cluster Size 1] Estimated beta: ${summary.coef}")
    println(s"[Gaussian - Cluster Size 1] Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.2,
      "Estimated beta should be close to true beta (Gaussian - Cluster Size 1)"
    )
  }

  test("GEE recovers true beta for Gaussian outcomes") {
    val rand = new Random(789)
    val trueBeta = DenseVector(0.0, 1.5, -2.0)
    val nClusters = 1000
    val obsPerCluster = 4

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterNoise = rand.nextGaussian() * 0.2
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val y = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + clusterNoise + rand
          .nextGaussian() * 0.3
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Gaussian)
    gee.fit(df, maxIter = 10)
    val summary = gee.result()

    println(s"[Gaussian] True beta: $trueBeta")
    println(s"[Gaussian] Estimated beta: ${summary.coef}")
    println(s"[Gaussian] Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.2,
      "Estimated beta should be close to true beta (Gaussian)"
    )
  }

  test("GEE recovers true beta for Poisson outcomes") {
    import breeze.stats.distributions.RandBasis
    import org.apache.commons.math3.random.MersenneTwister
    import breeze.stats.distributions.ThreadLocalRandomGenerator

    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))

    val rand = new Random(135)
    val trueBeta = DenseVector(0.0, 0.5, -0.5)
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.3
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + clusterEffect
        val lambda = math.exp(eta)
        val y = breeze.stats.distributions.Poisson(lambda).draw().toDouble
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Poisson)
    gee.fit(df, maxIter = 10)
    val summary = gee.result()

    println(s"[Poisson] True beta: $trueBeta")
    println(s"[Poisson] Estimated beta: ${summary.coef}")
    println(s"[Poisson] Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.3,
      "Estimated beta should be close to true beta (Poisson)"
    )
  }

  test("GEE summary returns correct beta and variance") {
    val rand = new Random(123)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 100
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, k) => xi * trueBeta(k) }.sum
        val prob = 1.0 / (1.0 + math.exp(-eta))
        val y = if (rand.nextDouble() < prob) 1.0 else 0.0
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)

    val gee = new GEE()
    gee.fit(df, maxIter = 10)
    val summary = gee.result()

    assert(summary.coef.length == trueBeta.length, "Beta length should match true beta length")
    assert(
      summary.variance.rows == trueBeta.length && summary.variance.cols == trueBeta.length,
      "Variance matrix dimensions should match beta length"
    )
  }

  test("GEE summary returns correct DataFrame with beta and statistics") {
    val rand = new Random(123)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 500
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, k) => xi * trueBeta(k) }.sum
        val prob = 1.0 / (1.0 + math.exp(-eta))
        val y = if (rand.nextDouble() < prob) 1.0 else 0.0
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)

    val gee = new GEE()
    gee.fit(df, maxIter = 10)
    val summaryDf = gee.summary()

    // Check DataFrame schema
    assert(
      summaryDf.columns.toSet == Set("parameter", "estimate", "std_error", "z_score", "p_value"),
      "DataFrame should contain correct columns"
    )

    // Check number of rows
    assert(summaryDf.count() == trueBeta.length, "DataFrame should have one row per coefficient")

    // Check values
    val rows = summaryDf.collect()
    rows.zipWithIndex.foreach {
      case (row, i) =>
        val expectedName = if (i == 0) "intercept" else s"beta$i"
        assert(row.getString(0) == expectedName, s"Row $i should have name $expectedName")
        assert(
          math.abs(row.getDouble(1) - trueBeta(i)) < 0.2,
          s"Row $i coefficient should be close to true beta"
        )
        assert(row.getDouble(2) > 0.0, s"Row $i standard error should be positive")
        assert(
          row.getDouble(4) >= 0.0 && row.getDouble(4) <= 1.0,
          s"Row $i p-value should be between 0 and 1"
        )
    }
  }

  test("GEE recovers true beta for Negative Binomial outcomes") {
    import breeze.stats.distributions.RandBasis
    import org.apache.commons.math3.random.MersenneTwister
    import breeze.stats.distributions.ThreadLocalRandomGenerator

    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(789)))

    val rand = new Random(246)
    val trueBeta = DenseVector(1.0, 0.5, -0.5)
    val trueKappa = 0.2 // Overdispersion parameter
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.2
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + clusterEffect
        val mu = math.exp(eta)
        // Generate NB using Gamma-Poisson mixture
        // If Z ~ Gamma(1/kappa, mu*kappa), then Y|Z ~ Poisson(Z) gives Y ~ NB
        val scale = trueKappa * mu
        val shape = 1.0 / trueKappa
        val gamma = breeze.stats.distributions.Gamma(shape, scale).draw()
        val y = breeze.stats.distributions.Poisson(gamma).draw().toDouble
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = NegativeBinomial)
    gee.fit(df, maxIter = 20, verbose = false)
    val summary = gee.result()

    println(s"[Negative Binomial] True beta: $trueBeta")
    println(s"[Negative Binomial] Estimated beta: ${summary.coef}")
    println(s"[Negative Binomial] Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.5,
      "Estimated beta should be close to true beta (Negative Binomial)"
    )
  }

  test("GEE recovers true beta for Tweedie outcomes") {
    import breeze.stats.distributions.RandBasis
    import org.apache.commons.math3.random.MersenneTwister
    import breeze.stats.distributions.ThreadLocalRandomGenerator

    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(999)))

    val rand = new Random(357)
    val trueBeta = DenseVector(1.0, 0.4, -0.4)
    val p = 2.0 // Power parameter (p=2 is Gamma, which is easier to generate)
    val truePhi = 0.5 // Dispersion parameter
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.15
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + clusterEffect
        val mu = math.exp(eta)
        // For p=2 (Gamma case), Var(Y) = phi * mu^2
        // Gamma with mean=mu and variance=phi*mu^2
        // shape = mu^2 / (phi * mu^2) = 1/phi, scale = phi * mu
        val shape = 1.0 / truePhi
        val scale = truePhi * mu
        val y = breeze.stats.distributions.Gamma(shape, scale).draw()
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Tweedie(p))
    gee.fit(df, maxIter = 20, verbose = false)
    val summary = gee.result()

    println(s"[Tweedie p=$p] True beta: $trueBeta")
    println(s"[Tweedie p=$p] Estimated beta: ${summary.coef}")
    println(s"[Tweedie p=$p] Estimated variance:\n${summary.variance}")

    assert(
      norm(summary.coef - trueBeta) < 0.5,
      s"Estimated beta should be close to true beta (Tweedie p=$p)"
    )
  }

  test("GEE correlation methods - returns valid correlation matrix and DataFrames") {
    val rand = new Random(456)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 100
    val obsPerCluster = 4

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.2 // induce correlation
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val eta = x.zipWithIndex.map { case (xi, k) => xi * trueBeta(k) }.sum + clusterEffect
        val prob = 1.0 / (1.0 + math.exp(-eta))
        val y = if (rand.nextDouble() < prob) 1.0 else 0.0
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(corStruct = Exchangeable)
    gee.fit(df, maxIter = 10, verbose = false)

    // Test getCorrelationStructure
    val retrievedCorStruct = gee.getCorrelationStructure()
    assert(retrievedCorStruct == Exchangeable, "Correlation structure should be Exchangeable")

    // Test getCorrelationMatrix
    val R = gee.getCorrelationMatrix()
    assert(R.rows == obsPerCluster, "Correlation matrix rows should match cluster size")
    assert(R.cols == obsPerCluster, "Correlation matrix cols should match cluster size")

    // Check diagonal is 1.0
    (0 until obsPerCluster).foreach { i =>
      assert(math.abs(R(i, i) - 1.0) < 1e-10, s"Diagonal element R($i,$i) should be 1.0")
    }

    // Check symmetry
    (0 until obsPerCluster).foreach { i =>
      (0 until obsPerCluster).foreach { j =>
        assert(math.abs(R(i, j) - R(j, i)) < 1e-10, s"Matrix should be symmetric")
      }
    }

    // For Exchangeable structure, off-diagonals should be equal
    val offDiag = R(0, 1)
    (0 until obsPerCluster).foreach { i =>
      (0 until obsPerCluster).foreach { j =>
        if (i != j) {
          assert(
            math.abs(R(i, j) - offDiag) < 1e-10,
            s"For Exchangeable, all off-diagonals should be equal"
          )
        }
      }
    }

    // Test correlationSummary - wide format (default)
    val corrDfWide = gee.correlationSummary()
    val expectedCols = Set("time_point") ++ (0 until obsPerCluster).map(i => s"time_$i")
    assert(corrDfWide.columns.toSet == expectedCols, "Wide format should have correct columns")
    assert(corrDfWide.count() == obsPerCluster, s"Wide format should have $obsPerCluster rows")

    val wideRows = corrDfWide.collect()
    wideRows.foreach { row =>
      val timePoint = row.getInt(0)
      val diagValue = row.getDouble(timePoint + 1)
      assert(math.abs(diagValue - 1.0) < 1e-10, s"Diagonal value should be 1.0")
    }

    // Test correlationSummary - long format
    val corrDfLong = gee.correlationSummary("long")
    assert(
      corrDfLong.columns.toSet == Set("time_i", "time_j", "correlation"),
      "Long format should have correct columns"
    )
    assert(
      corrDfLong.count() == obsPerCluster * obsPerCluster,
      "Long format should have correct number of rows"
    )

    val longRows = corrDfLong.collect()
    longRows.foreach { row =>
      val i = row.getInt(0)
      val j = row.getInt(1)
      val corr = row.getDouble(2)
      if (i == j) {
        assert(math.abs(corr - 1.0) < 1e-10, s"Diagonal correlation should be 1.0")
      }
    }

    println(s"Correlation structure: $retrievedCorStruct")
    println(s"Correlation matrix:\n$R")
    println(s"Wide format row count: ${corrDfWide.count()}")
    println(s"Long format row count: ${corrDfLong.count()}")
  }

  test("GEE correlation methods - throws exceptions correctly") {
    val gee = new GEE()

    // Test exceptions before fit
    val exception1 = intercept[IllegalStateException] {
      gee.getCorrelationMatrix()
    }
    assert(
      exception1.getMessage.contains("not been fitted"),
      "getCorrelationMatrix should throw before fit"
    )

    val exception2 = intercept[IllegalStateException] {
      gee.correlationSummary()
    }
    assert(
      exception2.getMessage.contains("not been fitted"),
      "correlationSummary should throw before fit"
    )

    // Fit the model
    val rand = new Random(333)
    val nClusters = 50
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(rand.nextGaussian())
        val y = if (rand.nextDouble() < 0.5) 1.0 else 0.0
        Obs(clusterId.toString, x, y)
      }
    }

    val df = spark.createDataset(data)
    gee.fit(df, maxIter = 5, verbose = false)

    // Test invalid format exception
    val exception3 = intercept[IllegalArgumentException] {
      gee.correlationSummary("invalid_format")
    }
    assert(
      exception3.getMessage.contains("Invalid format"),
      "Should throw exception for invalid format"
    )
    assert(exception3.getMessage.contains("long"), "Error message should mention valid formats")
    assert(exception3.getMessage.contains("square"), "Error message should mention valid formats")

    println("All exception handling works correctly")
  }

  test("GEE correctly estimates correlation with strong cluster effects") {
    val rand = new Random(999)
    val trueBeta = DenseVector(1.0, 0.5)
    val nClusters = 200
    val obsPerCluster = 4
    val clusterEffectStd = 2.0 // Large cluster effect
    val noiseStd = 0.5 // Small noise

    // Generate data with STRONG within-cluster correlation
    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * clusterEffectStd
      (0 until obsPerCluster).map { _ =>
        val x = rand.nextGaussian()
        val y = trueBeta(0) + trueBeta(1) * x + clusterEffect + rand.nextGaussian() * noiseStd
        Obs(clusterId.toString, Array(x), y)
      }
    }

    val df = spark.createDataset(data)
    val geeExch = new GEE(Exchangeable, Gaussian)
    geeExch.fit(df, maxIter = 20, verbose = false)

    // Expected correlation (ICC)
    val expectedCorr =
      clusterEffectStd * clusterEffectStd / (clusterEffectStd * clusterEffectStd + noiseStd * noiseStd)

    val R = geeExch.getCorrelationMatrix()

    // Check diagonal is 1.0
    (0 until obsPerCluster).foreach { i =>
      assert(math.abs(R(i, i) - 1.0) < 1e-10, s"Diagonal element R($i,$i) should be 1.0")
    }

    // For Exchangeable, all off-diagonals should be equal
    val estimatedCorr = R(0, 1)
    (0 until obsPerCluster).foreach { i =>
      (0 until obsPerCluster).foreach { j =>
        if (i != j) {
          assert(
            math.abs(R(i, j) - estimatedCorr) < 1e-10,
            s"For Exchangeable, R($i,$j) should equal R(0,1)"
          )
        }
      }
    }

    // Estimated correlation should be reasonably close to expected (within 20%)
    println(s"Expected correlation: $expectedCorr")
    println(s"Estimated correlation: $estimatedCorr")
    println(s"Correlation matrix:\n$R")

    assert(
      estimatedCorr > 0.5,
      s"With strong cluster effects, correlation should be > 0.5, got $estimatedCorr"
    )
    assert(
      math.abs(estimatedCorr - expectedCorr) < 0.2,
      s"Estimated correlation ($estimatedCorr) should be within 20% of expected ($expectedCorr)"
    )
  }
}

class GEEUtilsTest extends AnyFunSuite {

  test("computeClusterStats on small Binomial data") {
    val beta = DenseVector(0.5, -0.5)
    val eps = 1e-6
    val R = DenseMatrix.eye[Double](3) // Identity matrix for simplicity

    val cluster = Seq(
      Obs("c1", Array(1.0, 0.0), 1.0),
      Obs("c1", Array(1.0, 1.0), 0.0),
      Obs("c1", Array(1.0, 2.0), 1.0)
    )

    val UB = GEEUtils.computeClusterStats(cluster, beta, R, eps, family = Binomial)
    val U = UB._1
    val B = UB._2

    println(f"U: $U")
    println(f"B: \n$B")

    // Optionally check against expected values or properties
    assert(U.length == 2)
    assert(B.rows == 2 && B.cols == 2)
    assert(B(0, 0) > 0.0 && B(1, 1) > 0.0, "Diagonal elements of B should be positive")
  }
}
