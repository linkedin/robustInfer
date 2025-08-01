package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import breeze.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import scala.util.Random

class GEETest extends AnyFunSuite with BeforeAndAfterAll {

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
    val summary = gee.summary()

    println(s"True beta: $trueBeta")
    println(s"Estimated beta: ${summary.beta}")
    println(s"Estimated variance:\n${summary.variance}")

    assert(norm(summary.beta - trueBeta) < 0.2, "Estimated beta should be close to true beta")
  }

  test("GEE handles within-cluster correlation and recovers true beta") {

    val rand = new Random(456)
    val trueBeta = DenseVector(0.0, 1.0, -1.0)
    val nClusters = 1000
    val obsPerCluster = 3

    val data = (0 until nClusters).flatMap { clusterId =>
      val clusterEffect = rand.nextGaussian() * 0.2  // induce correlation
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
    val summary = gee.summary()

    println(s"True beta: $trueBeta")
    println(s"Estimated beta: ${summary.beta}")
    println(s"Estimated variance:\n${summary.variance}")

    assert(norm(summary.beta - trueBeta) < 0.3, "Estimated beta should be close to true beta under correlation")
  }

  test("GEE handles cluster size = 1 for Gaussian outcomes") {
    val rand = new Random(789)
    val trueBeta = DenseVector(0.0, 1.5, -2.0)
    val nClusters = 1000
    val obsPerCluster = 1 // Cluster size is 1

    val data = (0 until nClusters).flatMap { clusterId =>
      (0 until obsPerCluster).map { _ =>
        val x = Array(1.0, rand.nextGaussian(), rand.nextGaussian())
        val y = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + rand.nextGaussian() * 0.3
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Gaussian)
    gee.fit(df, maxIter = 10)
    val summary = gee.summary()

    println(s"[Gaussian - Cluster Size 1] True beta: $trueBeta")
    println(s"[Gaussian - Cluster Size 1] Estimated beta: ${summary.beta}")
    println(s"[Gaussian - Cluster Size 1] Estimated variance:\n${summary.variance}")

    assert(norm(summary.beta - trueBeta) < 0.2, "Estimated beta should be close to true beta (Gaussian - Cluster Size 1)")
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
        val y = x.zipWithIndex.map { case (xi, j) => xi * trueBeta(j) }.sum + clusterNoise + rand.nextGaussian() * 0.3
        Obs(clusterId.toString, x.drop(1), y)
      }
    }

    val df = spark.createDataset(data)
    val gee = new GEE(family = Gaussian)
    gee.fit(df, maxIter = 10)
    val summary = gee.summary()

    println(s"[Gaussian] True beta: $trueBeta")
    println(s"[Gaussian] Estimated beta: ${summary.beta}")
    println(s"[Gaussian] Estimated variance:\n${summary.variance}")

    assert(norm(summary.beta - trueBeta) < 0.2, "Estimated beta should be close to true beta (Gaussian)")
  }

  test("GEE recovers true beta for Poisson outcomes") {
    import breeze.stats.distributions.RandBasis
    import org.apache.commons.math3.random.MersenneTwister
    import breeze.stats.distributions.ThreadLocalRandomGenerator

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))

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
    val summary = gee.summary()

    println(s"[Poisson] True beta: $trueBeta")
    println(s"[Poisson] Estimated beta: ${summary.beta}")
    println(s"[Poisson] Estimated variance:\n${summary.variance}")

    assert(norm(summary.beta - trueBeta) < 0.3, "Estimated beta should be close to true beta (Poisson)")
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
    val summary = gee.summary()

    assert(summary.beta.length == trueBeta.length, "Beta length should match true beta length")
    assert(summary.variance.rows == trueBeta.length && summary.variance.cols == trueBeta.length, "Variance matrix dimensions should match beta length")
  }

  test("GEE dfSummary returns correct DataFrame with beta and statistics") {
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
    val summaryDf = gee.dfSummary()

    // Check DataFrame schema
    assert(summaryDf.columns.toSet == Set("names", "coef", "se", "z", "p-value"), "DataFrame should contain correct columns")

    // Check number of rows
    assert(summaryDf.count() == trueBeta.length, "DataFrame should have one row per coefficient")

    // Check values
    val rows = summaryDf.collect()
    rows.zipWithIndex.foreach { case (row, i) =>
      val expectedName = if (i == 0) "intercept" else s"beta$i"
      assert(row.getString(0) == expectedName, s"Row $i should have name $expectedName")
      assert(math.abs(row.getDouble(1) - trueBeta(i)) < 0.2, s"Row $i coefficient should be close to true beta")
      assert(row.getDouble(2) > 0.0, s"Row $i standard error should be positive")
      assert(row.getDouble(4) >= 0.0 && row.getDouble(4) <= 1.0, s"Row $i p-value should be between 0 and 1")
    }
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

    println(f"U: ${U}")
    println(f"B: \n${B}")

    // Optionally check against expected values or properties
    assert(U.length == 2)
    assert(B.rows == 2 && B.cols == 2)
    assert(B(0, 0) > 0.0 && B(1, 1) > 0.0, "Diagonal elements of B should be positive")
  }
}
