package example

import scala.math.random
import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.{SQLContext, SparkSession}

import scala.io.Source

object SparkPokemonMLExample {

  def main(args: Array[String]) {
    //    val conf = new SparkConf().setAppName()
    //    val sc = new SparkContext(conf)
    //    val sqlContext = new SQLContext(sc)

    val session = SparkSession.builder()
      .appName("Spark Sql example")
      .master("local[4]")
      .getOrCreate()

    //working_dir
    //val dir = args(0)

    //    val pokemons = Source.fromFile().mkString
    //    val combats = Source.fromFile(dir + "/combats.csv").mkString


    import session.implicits._

    val pokemons = session.sparkContext.textFile("./pokemon.csv")
      .map(_.toUpperCase)
      .map(_.split(","))
      .map(mapper)
      .toDF()

    pokemons.createTempView("pokemons")

    val s = session.sqlContext.sql("SELECT * FROM pokemons").groupBy("legendary").count()

    val indexedDf =  new Pipeline()
      .setStages(Array("primaryType", "secondaryType")
        .map(columnName => new StringIndexer()
          .setInputCol(columnName)
          .setOutputCol(s"${columnName}Index"))
      ).fit(pokemons).transform(pokemons)

    val oneHotEncodedDf = new Pipeline().setStages(
      indexedDf.columns.filter(_ contains "Index")
      .map(index => new OneHotEncoder()
        .setInputCol(index)
        .setOutputCol(s"${index}Vec"))
    ).fit(indexedDf).transform(indexedDf)


  }


  case class Pokemon(id: Long, name: String, primaryType: String, secondaryType: String, hp: Int, attack: Int,
                     defense: Int, attackSpeed: Int, attackDefense: Int, speed: Int, generation: Int, legendary: Boolean)


  def mapper(p: Array[String]): Pokemon = {
    Pokemon(p(0).trim.toLong, p(1), p(2), if (p(3) == "") "NONE" else p(3), p(4).toInt, p(5).toInt, p(6).toInt,
      p(7).toInt, p(8).toInt, p(9).toInt, p(10).toInt, p(11).trim.toBoolean)
  }

}
