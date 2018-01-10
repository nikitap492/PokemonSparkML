package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Dataset, SparkSession}

object SparkPokemonMLExample {

  // default params
  val maxBins = 200
  val maxDepth = 5
  val seed = 42
  val tol = 1e-6
  val gbIters = 30
  val lrIters = 200
  val treesNum = 20

  def main(args: Array[String]): Unit = {

    val session = SparkSession.builder()
      .appName("Pokemons")
      .master("local[4]")
      .getOrCreate()


    import session.implicits._

    val pokemons = session.sparkContext.textFile("./pokemon.csv")
      .map(_.toUpperCase)
      .map(_.split(","))
      .map(parser)
      .toDF()

    pokemons.createOrReplaceTempView("pokemons")


    val indexed = new Pipeline().setStages(Array("primaryType", "secondaryType")
      .map(columnName => new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}Index")))
      .fit(pokemons)
      .transform(pokemons)

    val data = indexed.map(p => LabeledPoint(binaryMapper(p.getBoolean(1)),
      Vectors.dense(List.range(5, 14).map(p.getDouble).toArray)))

    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed)
    val gbRoc = evaluate(train, test, gb())
    val lrRoc = evaluate(train, test, lr())
    val dtcRoc = evaluate(train, test, dtc())
    val rfRoc = evaluate(train, test, rf())
    val nbRoc = evaluate(train, test, new NaiveBayes())

    println(s"Accuracy:\nDecisionTreeClassifier\t$dtcRoc\nLogisticRegression\t$lrRoc" +
      s"\nGradientBoosting\t$gbRoc\nRandomForestClassifier\t$rfRoc\nNaiveBayes\t$nbRoc")


  }

  /**
    * evaluate accuracy (area under precision-recall curve)
    * @param train train dataset
    * @param test test data
    * @param classifier classifier for evaluation
    * @return accuracy
    */
  def evaluate(train: Dataset[_], test: Dataset[_],
               classifier: ProbabilisticClassifier[_, _, _ <: ProbabilisticClassificationModel[_, _]]): Double = {

    classifier.setLabelCol("label")
    classifier.setFeaturesCol("features")

    val predictions = classifier
      .fit(train)
      .transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderPR")

    val accuracy = evaluator.evaluate(predictions)
    println("error = " + (1.0 - accuracy))
    accuracy
  }

  /**
    * @return  DecisionTreeClassifier with default params
    */
  def dtc(): DecisionTreeClassifier = {
    new DecisionTreeClassifier()
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
  }

  /**
    * @return  LogisticRegression with default params
    */
  def lr(): LogisticRegression = {
    new LogisticRegression()
      .setMaxIter(lrIters)
      .setTol(tol)
  }

  /**
    * @return  GradientBoostingTreeClassifier with default params
    */
  def gb(): GBTClassifier = {
    new GBTClassifier().setMaxIter(gbIters)
  }


  /**
    * @return  RandomForestClassifier with default params
    */
  def rf(): RandomForestClassifier = {
    new RandomForestClassifier().setNumTrees(treesNum)
  }

  case class Pokemon(id: Long, legendary: Boolean, name: String, primaryType: String, secondaryType: String, hp: Double, attack: Double,
                     defense: Double, attackSpeed: Double, attackDefense: Double, speed: Double, generation: Double)

  /**
    * parse pokemon from csv
    * @param p is pokemon
    * @return parsed pokemon
    */
  def parser(p: Array[String]): Pokemon = {
    Pokemon(p(0).trim.toLong, p(11).trim.toBoolean, p(1), p(2), if (p(3) == "") "NONE" else p(3), p(4).toDouble, p(5).toDouble, p(6).toDouble,
      p(7).toDouble, p(8).toDouble, p(9).toDouble, p(10).toDouble)
  }

  /**
    * boolean - double mapper
    */
  def binaryMapper(flag: Boolean): Double = {
    if (flag) 1.0 else 0.0
  }

}
