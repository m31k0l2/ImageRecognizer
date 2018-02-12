fun testNet(name: String): Double {
    println("test $name")
    val nw = NetworkIO().load(name)!!
    return testNet(nw)
}

fun testNet(nw: Network): Double {
    var y = 0.0
    var counter = 0
    MNIST.batch.forEach {
        val o = nw.activate(it.colorsMatrix)
        val result = o[it.index]
        if (result != o.max()) {
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [${o.indexOf(o.max())} (${(o.max()!! * 10000).toInt() / 100.0}%)]")
        } else {
            counter++
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [OK]")
        }
        y += result
    }
    y = (y*1000).toInt()/(MNIST.batch.size*10.0)
    println("средний успех: $y%")
    println("$counter / ${MNIST.batch.size}")
    return y
}

fun testBest(): Boolean {
    val nw = NetworkIO().load("nets/nw.net")!!
    val image = MNIST.next()
    val result = nw.activate(image.colorsMatrix)
    val isOk = image.index == result.indexOf(result.max())
    println("[${image.index}] - $isOk")
    result.forEachIndexed { index, r ->
        val percent = (r*1000).toInt()/10.0
        if (percent > 0) println("$index => $percent")
    }
    return isOk
}

fun main(args: Array<String>) {
    MNIST.createBatch(100)
    testNet("nets/nw.net")
    var counter = 0
    for (i in 0..10) {
        if (testBest()) counter++
        println()
    }
    println("$counter / 10")
}