package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.{SparkSession, Dataset}
import breeze.linalg.{DenseVector, norm}
import robustinfer.UGEEUtils._

class DRGUMiniBatchTest extends AnyFunSuite with Matchers {

  // Create SparkSession for testing
  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("DRGUMiniBatchTest")
    .master("local[1]") // Use single thread for faster testing
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  test("fitMiniBatch should complete without errors on small dataset") {
    // Create very small synthetic data for fast testing
    val n = 20 // Much smaller dataset
    val p = 1 // Fewer features
    val data = (0 until n)
      .map { i =>
        Obs(
          i = i.toString,
          x = Array(scala.util.Random.nextGaussian()),
          y = scala.util.Random.nextGaussian(),
          timeIndex = Some(0),
          z = Some(if (i % 2 == 0) 1.0 else 0.0) // Deterministic for speed
        )
      }
      .toDS()

    val drgu = new DRGU()

    // Should not throw any exceptions
    noException should be thrownBy {
      drgu.fitMiniBatch(
        data = data,
        k = 3, // Small k
        maxEpochs = 3, // Few epochs
        pairsPerBatch = 50,
        ema = 0.5,
        lambda = 1e-3,
        s_variance = 10, // Reasonable number of anchors for small dataset
        m_variance = 5, // Partners per anchor for variance
        verbose = false
      )
    }
  }

  test("fitMiniBatch should produce valid parameter estimates") {
    // Small dataset for fast testing
    val n = 30
    val p = 1
    val data = (0 until n)
      .map { i =>
        val x = Array(i.toDouble / n) // Simple deterministic pattern
        val z = if (i % 2 == 0) 1.0 else 0.0
        val y = x(0) + scala.util.Random.nextGaussian() * 0.1 // Simple relationship

        Obs(
          i = i.toString,
          x = x,
          y = y,
          timeIndex = Some(0),
          z = Some(z)
        )
      }
      .toDS()

    val drgu = new DRGU()
    drgu.fitMiniBatch(
      data = data,
      k = 4,
      maxEpochs = 5,
      pairsPerBatch = 100,
      ema = 0.3,
      lambda = 1e-4,
      s_variance = 15, // More anchors for better variance estimation
      m_variance = 8, // Partners per anchor
      verbose = false
    )

    val result = drgu.result()

    // Check that parameters are reasonable (not NaN or infinite)
    result.coef.data.foreach { param =>
      param should not be Double.NaN
      param should not be Double.PositiveInfinity
      param should not be Double.NegativeInfinity
    }

    // Check that variance matrix has positive diagonal elements
    for (i <- 0 until result.variance.rows)
      result.variance(i, i) should be > 0.0
  }

  test("fitMiniBatch should work with summary() method") {
    val n = 25
    val p = 1
    val data = (0 until n)
      .map { i =>
        Obs(
          i = i.toString,
          x = Array(scala.util.Random.nextGaussian()),
          y = scala.util.Random.nextGaussian(),
          timeIndex = Some(0),
          z = Some(if (i % 2 == 0) 1.0 else 0.0)
        )
      }
      .toDS()

    val drgu = new DRGU()
    drgu.fitMiniBatch(
      data = data,
      k = 3,
      maxEpochs = 4,
      s_variance = 12, // Anchors for variance estimation
      m_variance = 6, // Partners per anchor
      verbose = false
    )

    // Should be able to generate summary without errors
    val summary = drgu.summary()

    // Check that summary has expected structure
    summary.columns should contain allOf ("parameter", "estimate", "std_error", "z_score", "p_value")
    summary.count() should be > 0L

    // Check that estimates are not NaN
    val estimates = summary.select("estimate").collect().map(_.getDouble(0))
    estimates.foreach(_ should not be Double.NaN)
  }
}
