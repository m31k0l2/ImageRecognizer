import kotlin.math.round

data class Example(val image: Image, var w: Double)

fun adaBoost(number: Int, examples: List<Example>, h: List<Network>): List<Double> {
    val z = MutableList(h.size) { 0.0 }
    for (m in 0 until h.size) {
        var error = 0.0
        for (j in 0 until examples.size) {
            val example = examples[j]
            val r = h[m].activate(example.image, 15.0)[0]
            if (r >= 0.5 && example.image.index != number || round(r) < 0.5 && example.image.index == number) {
                error += example.w
            }
        }
        for (j in 0 until examples.size) {
            val example = examples[j]
            val r = h[m].activate(example.image, 15.0)[0]
            if (r >= 0.5 && example.image.index == number || r < 0.5 && example.image.index != number) {
                example.w = example.w*error/(1-error)
            }
        }
        val w = examples.map { it.w }.sum()
        examples.forEach { it.w /= w }
        z[m] = Math.log((1-error)/error)
    }
    return z.toList()
}

fun test(number: Int, h: List<Network>, w: List<Double>) {
    var batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index == number }
    var counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r > 0.5) counter++
//        println("${image.index} -> $r ")
    }
    val c1 = counter*1.0/batch.count()
    batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index != number }
    counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r < 0.5) counter++
//        println("${image.index} -> $r ")
    }
    val c2 = counter*1.0/batch.count()
    println("$c1, $c2")
}

fun boost(number: Int, weights: List<Double>?=null) {
    val h = (1..10).mapNotNull { CNetwork().load("nets/nw$number$it.net") }
    println(h)
    val n = 500
    val examples = MNIST.buildBatch(n).map { Example(it, 1.0/n) }
    val b = examples.filter { it.image.index == number }
    b.forEach { it.w = 1.0/b.size }
    val w = weights ?: adaBoost(number, b, h)
    println(w)
    test(number, h, w)
}

fun main(args: Array<String>) {
//    val w = listOf(1.27, 0.65, 0.69, 0.32, -0.28) // 0
//    val w = listOf(3.18, -0.56, 1.06, -0.29, 2.44) // 1
    val w = listOf(0.41, 0.17, 0.88, 0.59, 0.84, 0.16, -0.46, -0.14, 0.06) // 2
    boost(2, w)
}