package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object SparkPokemonMLExample {

  def main(args: Array[String]): Unit = {

    val session = SparkSession.builder()
      .appName("Pokemons")
      .master("local[4]")
      .getOrCreate()


    import session.implicits._

    val pokemons = session.sparkContext.textFile("./pokemon.csv")
      .map(_.toUpperCase)
      .map(_.split(","))
      .map(mapper)
      .toDF()

    pokemons.createTempView("pokemons")


    val indexed = new Pipeline().setStages(Array("primaryType", "secondaryType")
      .map(columnName => new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}Index")))
      .fit(pokemons)
      .transform(pokemons)

    val data = indexed.map(p => LabeledPoint(binaryMapper(p.getBoolean(1)),
      Vectors.dense(List.range(5, 14).map(p.getDouble).toArray)))

    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), 42)

    val model = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .fit(train)

    val predictions = model.transform(test)


    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

  }


  case class Pokemon(id: Long, legendary: Boolean, name: String, primaryType: String, secondaryType: String, hp: Double, attack: Double,
                     defense: Double, attackSpeed: Double, attackDefense: Double, speed: Double, generation: Double)

  def mapper(p: Array[String]): Pokemon = {
    Pokemon(p(0).trim.toLong, p(11).trim.toBoolean,  p(1), p(2), if (p(3) == "") "NONE" else p(3), p(4).toDouble, p(5).toDouble, p(6).toDouble,
      p(7).toDouble, p(8).toDouble, p(9).toDouble, p(10).toDouble)
  }

  def binaryMapper(flag: Boolean): Double = {
    if (flag) 1.0 else 0.0
  }

}
