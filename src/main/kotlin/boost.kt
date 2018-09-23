import kotlin.math.round

val h = (1..5).map { CNetwork().load("nets/nw0$it.net") }.filterNotNull()

data class Example(val image: Image, var w: Double)

fun adaBoost(examples: List<Example>, h: List<Network>): List<Double> {
    val z = MutableList(h.size) { 0.0 }
    for (m in 0 until h.size) {
        var error = 0.0
        for (j in 0 until examples.size) {
            val example = examples[j]
            val r = h[m].activate(example.image, 15.0)[0]
            if (r >= 0.5 && example.image.index != 0 || round(r) < 0.5 && example.image.index == 0) {
                error += example.w
            }
        }
        for (j in 0 until examples.size) {
            val example = examples[j]
            val r = h[m].activate(example.image, 15.0)[0]
            if (r >= 0.5 && example.image.index == 0 || r < 0.5 && example.image.index > 0) {
                example.w = example.w*error/(1-error)
            }
        }
        val w = examples.map { it.w }.sum()
        examples.forEach { it.w /= w }
        z[m] = Math.log((1-error)/error)
    }
    return z.toList()
}

fun main(args: Array<String>) {
    println(h)
    val n = 500
    val examples = MNIST.buildBatch(n).map { Example(it, 1.0/n) }
    val b = examples.filter { it.image.index == 0 }
    b.forEach { it.w = 1.0/b.size }
    val w = adaBoost(b, h)
//    val w = listOf(1.27, 0.65, 0.69, 0.32, -0.28) // 0
    println(w)
    var batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index == 0 }
    var counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r > 0.5) counter++
        println("${image.index} -> $r ")
    }
    val c1 = counter*1.0/batch.count()
    batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index != 0 }
    counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r < 0.5) counter++
        println("${image.index} -> $r ")
    }
    val c2 = counter*1.0/batch.count()
    println("$c1, $c2")
}