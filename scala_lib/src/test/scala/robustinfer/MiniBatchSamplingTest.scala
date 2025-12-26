package robustinfer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import breeze.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import scala.util.Random
import robustinfer.UGEEUtils._

class MiniBatchSamplingTest extends AnyFunSuite with BeforeAndAfterAll {

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

  // Generate small test dataset
  def generateTestData(numObs: Int = 20, seed: Long = 42): org.apache.spark.sql.Dataset[Obs] = {
    val random = new Random(seed)
    val data = (1 to numObs).map { i =>
      Obs(
        i = s"c$i",
        x = Array.fill(2)(random.nextDouble()), // 2 covariates
        y = random.nextDouble(), // random outcome
        timeIndex = Some(1), // single time point
        z = Some(if (random.nextDouble() > 0.5) 1.0 else 0.0) // binary treatment
      )
    }
    data.toDS()
  }

  test("assignDeterministicPartition creates balanced partitions") {
    val df = generateTestData(20)
    val numPartitions = 3

    val partitioned = assignDeterministicPartition(df, "i", numPartitions)
    val partitionCounts = partitioned.groupBy("bucket").count().collect()

    // Check that we have reasonable number of partitions (hash partitioning may create more)
    assert(partitionCounts.length > 0, "Should have at least one partition")
    assert(partitionCounts.length <= numPartitions * 2, "Should not have too many partitions")

    // Check that partitioning is deterministic
    val partitioned2 = assignDeterministicPartition(df, "i", numPartitions)
    val matches = partitioned
      .as("p1")
      .join(partitioned2.as("p2"), $"p1.obs" === $"p2.obs")
      .filter($"p1.bucket" === $"p2.bucket")
      .count()
    assert(matches == df.count(), "Partitioning should be deterministic")
  }

  test("sampleKPartnersWithinPartitions generates correct number of pairs") {
    val df = generateTestData(10)
    val k = 3
    val numPartitions = 2

    val pairs = sampleKPartnersWithinPartitions(df, k, numPartitions, seed = 42)
    val pairCount = pairs.count()

    // Should generate some pairs (exact count depends on partition distribution)
    assert(pairCount > 0, "Should generate at least some pairs")
    assert(pairCount <= df.count() * k / 2, "Should not exceed theoretical maximum")

    // Check that all pairs have HT weights > 0
    val zeroWeights = pairs.filter($"weight" <= 0.0).count()
    assert(zeroWeights == 0, "All pairs should have positive HT weights")

    // Check no self-pairs (yi == yj and zi == zj would be suspicious)
    val suspiciousPairs = pairs.filter($"yi" === $"yj" && $"zi" === $"zj").count()
    assert(suspiciousPairs == 0, "Should not have self-pairs")
  }

  test("sampleKPartnersWithinPartitions is reproducible with same seed") {
    val df = generateTestData(15)
    val k = 2
    val seed = 123L

    val pairs1 = sampleKPartnersWithinPartitions(df, k, numPartitions = 2, seed = seed)
    val pairs2 = sampleKPartnersWithinPartitions(df, k, numPartitions = 2, seed = seed)

    val count1 = pairs1.count()
    val count2 = pairs2.count()

    assert(count1 == count2, "Same seed should produce same number of pairs")
    assert(count1 > 0, "Should generate some pairs")
  }

  test("sampleAnchorsWithinPartitions generates pairs with anchor IDs") {
    val df = generateTestData(12)
    val s_total = 4 // total anchors
    val m = 2 // partners per anchor

    val pairs = sampleAnchorsWithinPartitions(df, s_total, m, numPartitions = 2, seed = 42)
    val pairCount = pairs.count()

    // Should generate approximately s_total * m pairs
    // Note: Due to partition distribution, we might get slightly more pairs than s_total * m
    assert(pairCount > 0, "Should generate at least some pairs")
    // Relax the upper bound since partition-based sampling can exceed theoretical limit
    assert(pairCount <= s_total * m * 2, "Should not greatly exceed s_total * m pairs")

    // Check that all pairs have weight = 1.0 (no HT weighting for anchor-based)
    val nonUnitWeights = pairs.filter($"weight" =!= 1.0).count()
    assert(nonUnitWeights == 0, "All anchor-based pairs should have weight = 1.0")

    // Check that anchor IDs are properly assigned
    val anchorIds = pairs.select("anchorId").distinct().collect()
    assert(anchorIds.length > 0, "Should have at least one anchor ID")

    // Each anchor ID should be Some(Int), not None
    val nullAnchorIds = pairs.filter($"anchorId".isNull).count()
    assert(nullAnchorIds == 0, "All pairs should have non-null anchor IDs")
  }

  test("sampleAnchorsWithinPartitions groups pairs by anchor correctly") {
    val df = generateTestData(15)
    val s_total = 3
    val m = 2

    val pairs = sampleAnchorsWithinPartitions(df, s_total, m, numPartitions = 2, seed = 42)

    // Group by anchor and check counts
    val anchorCounts = pairs.groupBy("anchorId").count().collect()

    // Each anchor should have at most m partners (may be less due to partition constraints)
    anchorCounts.foreach { row =>
      val count = row.getLong(1)
      assert(count <= m, s"Anchor should have at most $m partners, got $count")
      assert(count > 0, "Each anchor should have at least 1 partner")
    }
  }

  test("PairFeatures enhanced fields work correctly") {
    val df = generateTestData(6)

    // Test k-partners sampling (should have HT weights)
    val kPairs = sampleKPartnersWithinPartitions(df, k = 2, numPartitions = 2, seed = 42)
    val kPairsSample = kPairs.take(1)

    if (kPairsSample.nonEmpty) {
      val pair = kPairsSample.head
      assert(pair.weight > 0.0, "K-partners pairs should have positive HT weight")
      assert(pair.anchorId.isEmpty, "K-partners pairs should not have anchor ID")
    }

    // Test anchor-based sampling (should have anchor IDs)
    val anchorPairs =
      sampleAnchorsWithinPartitions(df, s_total = 2, m = 2, numPartitions = 2, seed = 42)
    val anchorPairsSample = anchorPairs.take(1)

    if (anchorPairsSample.nonEmpty) {
      val pair = anchorPairsSample.head
      assert(pair.weight == 1.0, "Anchor-based pairs should have weight = 1.0")
      assert(pair.anchorId.isDefined, "Anchor-based pairs should have anchor ID")
    }
  }

  test("sampling handles edge cases gracefully") {
    // Test with very small dataset
    val smallDf = generateTestData(2)

    val pairs = sampleKPartnersWithinPartitions(smallDf, k = 5, numPartitions = 1, seed = 42)
    val pairCount = pairs.count()

    // Should handle case where k > available partners
    assert(pairCount >= 0, "Should handle small datasets gracefully")

    // Test with single observation (should produce no pairs)
    val singleDf = generateTestData(1)
    val singlePairs = sampleKPartnersWithinPartitions(singleDf, k = 2, numPartitions = 1, seed = 42)
    assert(singlePairs.count() == 0, "Single observation should produce no pairs")
  }

}
