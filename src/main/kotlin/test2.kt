val n = 8

fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(100)
    var err1 = 0
    var err2 = 0
    val a = 5 // решение, если проголосовало больше a
    batch.filter { it.index in (0..n) }.forEach { image ->
        var res = (0..n).map { it to Detectors.detect(it, image) }.filter { it.second > a }
        if (res.size > 1) {
            err1++
            println("${image.index} -> ?")
        } else {
            res.maxBy { it.second }?.let { (n, r) ->
                println("${image.index} -> $n [$r]")
                if (n != image.index) err2++
            } ?: run {
                res = (0..n).map { it to Detectors.detect2(it, image) }.filter { it.second > a }
                if (res.size > 1 || res.isEmpty()) {
                    err1++
                    println("${image.index} -> ?")
                } else {
                    res.maxBy { it.second }?.let { (n, r) ->
                        println("${image.index} -> $n [$r]")
                        if (n != image.index) err2++
                    } ?: run {
                        err1++

                        println("${image.index} -> ?")
                    }
                }
            }
        }
    }
    println("Не распознано: $err1")
    println("Ошибочно распознано: $err2")
}

object Detectors {
    private val ds = (0..n).map { n -> (0..9).map { "nets/nw$n$it.net" }.mapNotNull { NetworkIO().load(it) } }
    private val de = (0..n).map { n -> (0..9).map { "nets/nw$it$n.net" }.mapNotNull { NetworkIO().load(it) } }

    private fun detect(ds: List<List<Network>>, n: Int, image: Image) = ds[n].map { it.activate(image) }.filter { it[n] > 0.8 }.size
    fun detect(n: Int, image: Image) = detect(ds, n, image)
    fun detect2(n: Int, image: Image) = detect(de, n, image)
    fun detect3(n: Int, image: Image) = ds.flatMap { it }.map { it.activate(image) }
}