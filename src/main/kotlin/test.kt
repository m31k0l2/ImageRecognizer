val n = 9

fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(100).filter { it.index in (0..n) }
    var err = 0
    val a = 7 // решение, если проголосовало больше a
    batch.filter { it.index in (0..n) }.forEach { image ->
        var res = (0..n).map { it to Detectors.detect(it, image) }.filter { it.second > a }
        val r = when {
            res.size == 1 -> res.first().first
            res.size > 1 -> {
                res = res.map { it.first }.map { i -> i to (res.find { it.first == i }!!.second + Detectors.detect2(i, image)) }
               res.maxBy { it.second }!!.first
            }
            else -> {
                Detectors.detect3(image)
            }
        }
        println("${image.index} -> $r")
        if (r != image.index) err++
    }
    println("Ошибочно распознано: $err / ${batch.size}")
    println("Успех: ${((batch.size-err)*1000.0/batch.size).toInt()/10.0}%")
}

object Detectors {
    private val ds = (0..n).map { n -> (0..9).map { "nets/nw$n$it.net" }.mapNotNull { NetworkIO().load(it) } }
    private val de = (0..n).map { n -> (0..9).map { "nets/nw$it$n.net" }.mapNotNull { NetworkIO().load(it) } }

    private fun detect(ds: List<List<Network>>, n: Int, image: Image) = ds[n].map { it.activate(image) }.filter { it[n] > 0.8 }.size
    fun detect(n: Int, image: Image) = detect(ds, n, image)
    fun detect2(n: Int, image: Image) = detect(de, n, image)
    fun detect3(image: Image) = ds.flatMap { it }.map { it.activate(image) }.reduce { acc, list ->
        acc.zip(list, {a, b -> a + b})
    }.let {
        it.indexOf(it.max())
    }
}