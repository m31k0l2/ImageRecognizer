import java.util.*
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
        val w = examples.map { it.w }.sum()
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
    var totalCounter = counter
    val c1 = (counter*1.0/batch.count()*100).toInt()/100.0
    batch = b.asSequence().filter { it.index != number }
    counter = 0
    batch.forEach { image ->
        val r = h.asSequence().map { it.activate(image, 15.0)[0] }.mapIndexed { i, d -> d*w[i] }.sum()
        if (r < 0.5) counter++
//        println("${image.index} -> $r ")
    }
    totalCounter += counter
    val c2 = (counter*1.0/batch.count()*100).toInt()/100.0
    val c3 = (totalCounter*1.0/b.count()*100).toInt()/100.0
    println("$c1, $c2, $c3")
}

fun boost(number: Int, batch: Set<Image>, weights: List<Double>?=null): List<Double> {
    val h = (1..100).mapNotNull { CNetwork().load("nets/nw$number$it.net") }
    val examples = batch.asSequence().map { Example(it, 1.0/batch.size) }.toList()
    return adaBoost(number, examples, h, weights).asSequence().map { it*100 }.map { it.toInt() }.map { it / 100.0 }.toList()
}

fun main(args: Array<String>) {
    val number = 6
    val tb = MNIST.buildBatch(1000, MNIST.mnistTestPath)
//    val w = listOf(1.27, 0.65, 0.69, 0.32, 0.0) // 0
//    val w = listOf(3.18, 0.0, 1.06, 0.0, 2.44) // 1
//    val w = listOf(2.21, 1.34, 0.82, 0.39, 0.3) // 2

    val b = MNIST.buildBatch(500)
    val w = boost(number, b)
    w.forEachIndexed { i, d ->
        if (d < 0) {
            val path = "nets/nw$number${i+1}.net"
            delete(path)
        }
    }
    test(number, w, tb)
    test(number, w.map { if (it < 0) 0.0 else it }, tb)
    var n = 1
    for (i in 1..100) {
        val path = "nets/nw$number$i.net"
        CNetwork().load(path)?.let {
            if (n != i) {
                saveAs(path, "nets/nw$number$n.net")
                delete(path)
            }
            n++
        }
    }
    println(w.filter { it > 0 })
}