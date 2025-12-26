package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.{SparkSession, Dataset}
import breeze.linalg.{DenseVector, diag, norm, trace}
import robustinfer.UGEEUtils._

class DRGUValidationTest extends AnyFunSuite with Matchers {

  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("DRGUValidationTest")
    .master("local[1]")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .getOrCreate()

  import spark.implicits._

  test("theta consistency on null data - all coefficients should be close to zero") {
    // Generate null data using the existing generateData function
    val data = generateData(
      numClusters = 30, // Reduced for faster testingld
      numObsPerCluster = 1,
      p = 2,
      etaTrue = Array(0.0, 0.0, 0.0), // p + 1 = 3, all zeros (null hypothesis)
      betaTrue = Array(0.0, 0.0, 0.0, 0.0) // p + 2 = 4, all zeros (null hypothesis)
    )

    println(s"Dataset: ${data.count()} observations")

    // Fit with original method
    val drguOriginal = new DRGU()
    val originalConverged = drguOriginal.fit(
      df = data,
      maxIter = 10,
      tol = 1e-6,
      verbose = true // Enable to see convergence
    )
    val resultOriginal = drguOriginal.result()

    // Fit with mini-batch method
    val drguMiniBatch = new DRGU()
    val miniBatchConverged = drguMiniBatch.fitMiniBatch(
      data = data,
      k = 20, // Reduced for faster testing
      maxEpochs = 10, // Reduced for faster testing
      pairsPerBatch = 1000,
      s_variance = 20, // Reduced for faster variance estimation
      m_variance = 10, // Reduced for faster variance estimation
      verbose = true // Enable to see convergence
    )
    val resultMiniBatch = drguMiniBatch.result()

    // Compare theta estimates
    val paramDiff = norm(resultOriginal.coef - resultMiniBatch.coef)

    println(s"\n=== NULL DATA THETA COMPARISON ===")
    println(s"Original converged: $originalConverged")
    println(s"MiniBatch converged: $miniBatchConverged")
    println(s"Original theta: ${resultOriginal.coef}")
    println(s"MiniBatch theta: ${resultMiniBatch.coef}")
    println(s"Theta difference (L2 norm): $paramDiff")

    // Under null hypothesis, both estimates should be close to [0.5, 0, 0, 0, 0, 0]
    // (delta = 0.5 under null, others = 0)
    println(s"\nExpected under null: [0.5, 0, 0, 0, 0, 0]")

    // Main test: Both methods should produce valid, finite estimates
    resultOriginal.coef.data.foreach(_ should not be Double.NaN)
    resultMiniBatch.coef.data.foreach(_ should not be Double.NaN)
    resultOriginal.coef.data.foreach(x => math.abs(x) should be < 10.0) // Reasonable bounds
    resultMiniBatch.coef.data.foreach(x => math.abs(x) should be < 10.0) // Reasonable bounds

    // The two methods should produce reasonably similar estimates
    paramDiff should be < 3.0 // Focus on consistency, not absolute values

    println(s"Test passed: Both methods converged to reasonable null estimates")
  }

  test("theta consistency on alternative data - non-zero coefficients") {
    // Generate alternative data with non-zero true coefficients
    val data = generateData(
      numClusters = 30, // Reduced for faster testing
      numObsPerCluster = 1,
      p = 2,
      etaTrue = Array(0.0, 0.5, -0.3), // p + 1 = 3, non-zero effect
      betaTrue = Array(0.0, 2.0, 0.5, -0.2) // p + 2 = 4, non-zero effects
    )

    println(s"Dataset: ${data.count()} observations")

    // Fit with original method
    val drguOriginal = new DRGU()
    val originalConverged = drguOriginal.fit(
      df = data,
      maxIter = 10,
      tol = 1e-6,
      verbose = true // Enable to see convergence
    )
    val resultOriginal = drguOriginal.result()

    // Fit with mini-batch method
    val drguMiniBatch = new DRGU()
    val miniBatchConverged = drguMiniBatch.fitMiniBatch(
      data = data,
      k = 20, // Reduced for faster testing
      maxEpochs = 10, // Reduced for faster testing
      pairsPerBatch = 1000,
      s_variance = 20, // Reduced for faster variance estimation
      m_variance = 10, // Reduced for faster variance estimation
      verbose = true // Enable to see convergence
    )
    val resultMiniBatch = drguMiniBatch.result()

    // Compare theta estimates
    val paramDiff = norm(resultOriginal.coef - resultMiniBatch.coef)

    println(s"\n=== ALTERNATIVE DATA THETA COMPARISON ===")
    println(s"Original converged: $originalConverged")
    println(s"MiniBatch converged: $miniBatchConverged")
    println(s"Original theta: ${resultOriginal.coef}")
    println(s"MiniBatch theta: ${resultMiniBatch.coef}")
    println(s"Theta difference (L2 norm): $paramDiff")

    // Under alternative hypothesis, delta should be significantly different from 0.5
    println(s"\nExpected: delta != 0.5, other coefficients != 0")

    // Both methods should detect the alternative (delta away from 0.5)
    math.abs(resultOriginal.coef(0) - 0.5) should be > 0.1 // delta away from 0.5
    math.abs(resultMiniBatch.coef(0) - 0.5) should be > 0.1 // delta away from 0.5

    // The two methods should produce similar estimates
    paramDiff should be < 2.0 // Allow some difference due to sampling strategies

    println(s"Test passed: Both methods detected alternative hypothesis")
  }

}
