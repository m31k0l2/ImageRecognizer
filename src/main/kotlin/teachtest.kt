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

fun Network.convLayer(neuronCount: Int) {
    layers.add(Layer(neuronCount))
}

fun Network.fullConnectedLayer(neuronCount: Int, alpha: () -> Double) {
    Neuron.alpha = alpha()
    layers.add(Layer(neuronCount))
}

fun alpha(alpha: Double) = alpha

abstract class NewEvolution(val time: Int, val popSize: Int) : NetEvolution()

fun evolution(time: Int, population: Int, init: Network.() -> Unit): NewEvolution {
    class Evolution : NewEvolution(time, population) {
        override fun createNet() = CNetwork().apply(init)
    }
    return Evolution()
}

fun network(init: Layer.() -> Unit) {
    Layer().apply(init)
}

fun NewEvolution.run(numbers: IntArray, trainLayers: IntArray): Network {
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
    batch = MNIST.buildBatch(1000).filter { it.index in numbers }
    this.trainLayers = trainLayers.toList()
    evolute(time, popSize)
    return leader!!.nw
}

fun Network.saveToFile(path: String) {
    NetworkIO().save(this, path)
}

fun evolute(structure: IntArray, trainLayers: IntArray, alpha: Double, numbers: IntArray): Network {
    return evolution(3000, 60) {
        network {
            convLayer(structure[0])
            convLayer(structure[1])
            convLayer(structure[2])
            convLayer(structure[3])
            fullConnectedLayer(structure[4]) {
                alpha(alpha)
            }
            fullConnectedLayer(structure[5]) {
                alpha(alpha)
            }
        }
    }.run(numbers, trainLayers)
}

fun Network.rate(vararg numbers: Int): Double {
    Neuron.alpha = 15.0
    return testMedianNet(this, MNIST.buildBatch(500), numbers)
}

fun train(structure: IntArray, trainLayers: IntArray, teachNumbers: IntArray) {
    for (alpha in listOf(1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)) {
        log.info("alpha: $alpha")
        log.info("structure: ${getStructure("nets/nw.net").toList()}")
        log.info("train layers: ${trainLayers.toList()}")
        val nw = evolute(structure, trainLayers, alpha, teachNumbers)
        nw.saveToFile("nets/nw.net")
        val rate = nw.rate(*teachNumbers)
        log.info("rate: $rate")
        if (rate > 0.98) break
    }
    saveAs("nets/nw.net", "nets/nwx.net")
    rebuild(teachNumbers, 40)
}

fun main(args: Array<String>) {
    setupLog(log)
    val teachNumbers = intArrayOf(7, 8, 9)
    var structure = if (NetworkIO().load("nets/nw.net") != null) getStructure("nets/nw.net") else intArrayOf(6,6,4,4,40,10)
    while (true) {
        train(structure, intArrayOf(), teachNumbers)
        structure = getStructure("nets/nw.net")
        if (structure.sum() == getStructure("nets/nwx.net").sum()) break
        if (structure.take(4).sum() < 5) break
        train(structure, intArrayOf(4, 5), teachNumbers)
    }
    saveAs("nets/nwx.net", "nets/nw${teachNumbers.joinToString("")}_${structure.joinToString("-")}.net")
}