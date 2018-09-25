import kotlin.math.round

data class Example(val image: Image, var w: Double)

fun adaBoost(number: Int, examples: List<Example>, h: List<Network>, weights: List<Double>?): List<Double> {
    val z = weights?.toMutableList() ?: MutableList(h.size) { 1.0 }
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
        val w = Math.sqrt(examples.asSequence().map{ it.w*it.w }.sum())
        examples.forEach { it.w /= w }
        z[m] *= Math.log((1-error)/error)
    }
    return z.toList()
}

fun test(number: Int, w: List<Double>, b: Set<Image>) {
    println(w)
    val h = (1..100).mapNotNull { CNetwork().load("nets/nw$number$it.net") }
    var batch = b.asSequence().filter { it.index == number }
    var counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r > 0.5) counter++
//        println("${image.index} -> $r ")
    }
    val c1 = counter*1.0/batch.count()
    batch = b.asSequence().filter { it.index != number }
    counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r < 0.5) counter++
//        println("${image.index} -> $r ")
    }
    val c2 = counter*1.0/batch.count()
    println("$c1, $c2")
}

fun boost(number: Int, batch: Set<Image>, weights: List<Double>?=null): List<Double> {
    val h = (1..11).mapNotNull { CNetwork().load("nets/nw$number$it.net") }
    val examples = batch.asSequence().map { Example(it, 1.0/batch.size) }.toList()
    val w = adaBoost(number, examples, h, weights).asSequence().map { it*100 }.map { it.toInt() }.map { it / 100.0 }.toList()
    println(w)
    return w
}

fun main(args: Array<String>) {
//    val w = listOf(1.27, 0.65, 0.69, 0.32, 0.0) // 0
//    val w = listOf(3.18, 0.0, 1.06, 0.0, 2.44) // 1
//    val w0 = listOf(1.0856164383561644, 0.08561643835616438, 0.08561643835616438, 0.08561643835616438, 0.08561643835616438, 0.5445205479452055, 0.08904109589041097, 0.6335616438356165, 0.0, 0.085616438356164382)
    val b = MNIST.buildBatch(1000)
    val w = boost(2, b)
    val tb = MNIST.buildBatch(1000, MNIST.mnistTestPath)
    test(2, w, tb)
    test(2, w.map { if (it < 0) 0.0 else it }, tb)
    val min = w.min()!!
    val max = w.max()!!
    val s = Math.sqrt(w.map { it*it }.sum())
    val s2 = Math.sqrt(w.map { (it - min)*(it - min) }.sum())
    test(2, w.map { it - min }, tb)
    test(2, w.map { (it - min)/max }, tb)
    test(2, w.map { it/s }, tb)
    test(2, w.map { (it - min)/s2 }, tb)
}