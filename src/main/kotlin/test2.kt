fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(100)
    var err1 = 0
    var err2 = 0
    val a = 6 // решение, если проголосовало больше a
    batch.forEach { image ->
        val res = (0..2).map { it to Detectors.detect(it, image) }.filter { it.second > a }
        if (res.size > 1) {
            err1++
            println("${image.index} -> ?")
        } else {
            res.maxBy { it.second }?.let { (n, r) ->
                println("${image.index} -> $n [$r]")
                if (n != image.index) err2++
            } ?: run {
                if (image.index in (0..2)) err1++
                println("${image.index} -> NO")
            }
        }
    }
    println("Не распознано: $err1")
    println("Ошибочно распознано: $err2")
}

object Detectors {
    private val ds = (0..2).map { n -> (0..9).filter { it != 0 }.map { "nets/nw$n$it.net" }.mapNotNull { NetworkIO().load(it) } }

    fun detect(n: Int, image: Image) = ds[n].map { it.activate(image) }.filter { it[n] > 0.8 }.size
}