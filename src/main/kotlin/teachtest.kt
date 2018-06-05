/*

evolution(time=1500, population=60) {
    network {
        CNNLayers(6, 6, 4, 4)
        FullConnectedLayers(60, 10) {
            alpha(1.0)
        }
    }
}

 */

fun Network.convLayer(neuronCount: Int, divider: MatrixDivider, pooler: Pooler?=null) {

    val layer = CNNLayer(divider, pooler, neuronCount)
    layers.add(layer)
}

fun Network.fullConnectedLayer(neuronCount: Int, alpha: () -> Double) {
    layers.add(FullConnectedLayer(alpha(), neuronCount))
}

fun alpha(alpha: Double) = alpha

abstract class NewEvolution(val time: Int, val popSize: Int) : NetEvolution()

fun evolution(time: Int, population: Int, init: NetEvolution.() -> Network): NewEvolution {
    class Evolution : NewEvolution(time, population) {
        override fun createNet() = init()
    }
    return Evolution()
}

fun network(init: Network.() -> Unit): Network {
    return CNetwork().apply(init)
}

fun NewEvolution.run(numbers: IntArray, trainFullConnectedLayers: Boolean): Network {
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
    batch = MNIST.buildBatch(500).filter { it.index in numbers }
    if (trainFullConnectedLayers) {
        val nw = NetworkIO().load("nets/nw.net")!!
        batch.forEach {
            it.o = nw.activate(it)
        }
        trainLayers = listOf(4, 5)
    }
    evolute(time, popSize)
    return leader!!.nw
}

fun Network.saveToFile(path: String) {
    NetworkIO().save(this, path)
}

fun evolute(structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, numbers: IntArray): Network {
    return evolution(150, 60) {
        buildNetwork(structure, alpha)
    }.run(numbers, trainFullConnectedLayers)
}

val testBatch = MNIST.buildBatch(500)

fun Network.rate(vararg numbers: Int): Double {
    Neuron.alpha = 15.0
    return testMedianNet(this, testBatch, numbers)
}

fun train(structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray): Double {
    var rate = NetworkIO().load("nets/nw.net")?.rate(*teachNumbers) ?: 0.0
    for (alpha in listOf(15.0, 3.0, 2.0, 1.0, 2.0, 3.0)) {
        log.info("alpha: $alpha")
        log.info("structure: ${getStructure("nets/nw.net").toList()}")
        log.info("trainFullConnectedLayers: $trainFullConnectedLayers")
        val nw = evolute(structure, trainFullConnectedLayers, alpha, teachNumbers)
        val curRate = nw.rate(*teachNumbers)
        log.info("rate: $curRate")
        if (curRate > rate) {
            nw.saveToFile("nets/nw.net")
            log.info("SAVE")
            rate = curRate
        }
        if (rate > 0.98) break
    }
    return rate
}

fun fullTrain(structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray) {
    var r1 = 0.0
    while (true) {
        val r2 = train(structure, trainFullConnectedLayers, teachNumbers)
        log.info("\r\nresult $r1 -> $r2")
        if (r2 < r1) break
        r1 = r2
    }
    saveAs("nets/nw.net", "nets/nwx.net")
}

fun main(args: Array<String>) {
    setupLog(log)
    val teachNumbers = (0..2).toList().toIntArray()
    var structure = if (NetworkIO().load("nets/nw.net") != null) getStructure("nets/nw.net") else intArrayOf(6,6,4,4,40,10)
//    fullTrain(structure, true, teachNumbers)
    while (true) {
        fullTrain(structure, false, teachNumbers)
        rebuild(teachNumbers, 40)
        structure = getStructure("nets/nw.net")
        if (structure.sum() == getStructure("nets/nwx.net").sum()) break
        if (structure.take(4).sum() < 5) break
        fullTrain(structure, true, teachNumbers)
        saveAs("nets/nwx.net", "nets/nw${teachNumbers.joinToString("")}_${structure.joinToString("-")}.net")
    }
}

