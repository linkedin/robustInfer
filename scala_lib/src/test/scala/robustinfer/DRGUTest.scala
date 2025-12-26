package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import breeze.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import scala.util.Random
import robustinfer.UGEEUtils.generateData

class DRGUTest extends AnyFunSuite with BeforeAndAfterAll {

  implicit lazy val spark: SparkSession = SparkSession
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

  test("DRGU works on small data") {
    val data = generateData(
      numClusters = 30,
      numObsPerCluster = 1,
      p = 1,
      etaTrue = Array(0.0, 0.0), // p + 1
      betaTrue = Array(0.0, 0.0, 0.0) // p + 2
    )

    val drgu = new DRGU()
    drgu.fit(data, maxIter = 10, tol = 1e-6, verbose = true)
    val result = drgu.result()
    val summary = drgu.summary()

    // check results
    assert(result != null)
    assert(result.coef.length == 6) // 1 + (p + 1) + (2 * p + 1)
    assert(result.variance.rows == 6 && result.variance.cols == 6)

    // check result columns
    assert(summary.columns === Array("parameter", "estimate", "std_error", "z_score", "p_value"))

    // check p-values (all should be non-significant)
    assert(
      summary.select("p_value").as[Double].collect().forall(_ > 0.05),
      "All p-values should be greater than 0.05"
    )
  }

  test("DRGU on medium data under null") {
    val data = generateData(
      numClusters = 500,
      numObsPerCluster = 1,
      p = 1,
      etaTrue = Array(0.0, 0.0), // p + 1
      betaTrue = Array(0.0, 0.0, 0.0) // p + 2
    )

    val drgu = new DRGU()
    drgu.fit(data, maxIter = 10, tol = 1e-6, verbose = true)
    val result = drgu.result()
    val summary = drgu.summary()

    // check result columns
    assert(summary.columns === Array("parameter", "estimate", "std_error", "z_score", "p_value"))

    // check estimates are close to zero
    assert(
      result.coef.forall(math.abs(_) < 1.0),
      "All coefficients should be close to zero under null hypothesis"
    )

    // check p-values (all should be non-significant)
    assert(
      summary.select("p_value").as[Double].collect().forall(_ > 0.05),
      "All p-values should be greater than 0.05"
    )
    // check delta is close to 0.5
    assert(
      math.abs(summary.select("estimate").as[Double].collect().head - 0.5) < 0.2,
      "Delta estimate should be close to 0.5 under null hypothesis"
    )
  }

  test("DRGU on medium data under alternative") {
    val data = generateData(
      numClusters = 500,
      numObsPerCluster = 1,
      p = 3,
      etaTrue = Array(0.0, 0.5, -0.5, 0.2), // p + 1
      betaTrue = Array(0.0, 5.5, 0.5, 0.0, 0.1) // p + 2
    )

    val drgu = new DRGU()
    drgu.fit(data, maxIter = 10, tol = 1e-6, verbose = true)
    val result = drgu.result()
    val summary = drgu.summary()

    // check result columns
    assert(summary.columns === Array("parameter", "estimate", "std_error", "z_score", "p_value"))

    // check estimates for delta are away from zero
    assert(math.abs(result.coef(0)) > 0.2, "(Delta - 0.5) estimate should be away from 0.2")

    // check p-values for delta are significant
    assert(
      summary.select("p_value").as[Double].collect().head < 0.05,
      "Delta p-value should be significant"
    )
  }

}
