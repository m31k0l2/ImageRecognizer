import java.awt.Toolkit
import java.util.logging.*

val log: Logger = Logger.getLogger("logger")

fun beep() {
    for (i in 1..60) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

fun getStructure(path: String): IntArray {
    val nw = CNetwork().load(path) ?: return emptyArray<Int>().toIntArray()
    return nw.layers.map { it.neurons.size }.toIntArray()
}

fun rebuild(teachNumbers: IntArray, hiddenLayerNeurons: Int=40) {
    val map = clean(teachNumbers)
    var list = mutableListOf<Int>()
    for (i in 0..3) {
        map[i]?.let { list.add(it.size) } ?: list.add(1)
    }
    if (list.reduce { acc, i -> acc * i } == 1) {
        list = mutableListOf(1, 1, 2, 2)
    }
    list.add(hiddenLayerNeurons)
    list.add(10)
    println(list)
    map.map { it.key to it.value.map { it.first } }.toMap()
    changeStructure("nets/nwx.net", "nets/nw.net",
            listOf(0,1,2,3),
            map.map { it.key to it.value.map { it.first } }.toMap(),
            buildNetwork(*list.toIntArray())
    )
}

fun setupLog(log: Logger) {
    val fh = FileHandler("log.txt")
    log.addHandler(fh)
    class NoTimeStampFormatter : SimpleFormatter() {
        override fun format(record: LogRecord?): String {
            return if (record?.level == Level.INFO) {
                record?.message + "\r\n"
            } else {
                super.format(record)
            }
        }
    }
    fh.formatter = NoTimeStampFormatter()
}

fun Network.convLayer(neuronCount: Int, divider: MatrixDivider, pooler: Pooler?=null) {

    val layer = CNNLayer(divider, pooler, neuronCount)
    layers.add(layer)
}

fun Network.fullConnectedLayer(neuronCount: Int) {
    layers.add(FullConnectedLayer(neuronCount))
}

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

fun NewEvolution.run(numbers: IntArray, trainFullConnectedLayers: Boolean, alpha: Double): Network {
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
    batch = MNIST.buildBatch(500).filter { it.index in numbers }
    if (trainFullConnectedLayers) {
        val nw = CNetwork().load("nets/nw.net")!!
        batch.forEach {
            it.o = nw.activateConvLayers(it)
        }
        trainLayers = listOf(4, 5)
    }
    evolute(time, popSize, alpha)
    return leader!!.nw
}

fun buildNetwork(vararg structure: Int)= network {
    convLayer(structure[0], CNetwork.cnnDividers[0], Pooler(CNetwork.poolDividers[0]!!))
    convLayer(structure[1], CNetwork.cnnDividers[1], Pooler(CNetwork.poolDividers[1]!!))
    convLayer(structure[2], CNetwork.cnnDividers[2])
    convLayer(structure[3], CNetwork.cnnDividers[3])
    fullConnectedLayer(structure[4])
    fullConnectedLayer(structure[5])
}

fun evolute(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, numbers: IntArray): Network {
    return evolution(time, 60) {
        buildNetwork(*structure)
    }.run(numbers, trainFullConnectedLayers, alpha)
}

val testBatch = MNIST.buildBatch(500)

fun Network.rate(vararg numbers: Int): Double {
    return testMedianNet(this, testBatch, numbers)
}

fun train(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray): Double {
    var rate = CNetwork().load("nets/nw.net")?.rate(*teachNumbers) ?: 0.0
    for (alpha in listOf(15.0, 3.0, 2.0, 1.0, 2.0, 3.0)) {
        log.info("alpha: $alpha")
        log.info("structure: ${getStructure("nets/nw.net").toList()}")
        log.info("trainFullConnectedLayers: $trainFullConnectedLayers")
        while (true) {
            val nw = evolute(time, structure, trainFullConnectedLayers, alpha, teachNumbers)
            val curRate = nw.rate(*teachNumbers)
            log.info("rate: $curRate")
            if (curRate > rate) {
                nw.save("nets/nw.net")
                log.info("SAVE")
                rate = curRate
            } else break
        }
        if (rate > 0.98) return rate
    }
    return rate
}

fun fullTrain(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray) {
    var r1 = 0.0
    while (true) {
        val r2 = train(time, structure, trainFullConnectedLayers, teachNumbers)
        log.info("\r\nresult $r1 -> $r2")
        if (r2 <= r1) break
        r1 = r2
    }
    saveAs("nets/nw.net", "nets/nwx.net")
}

fun main(args: Array<String>) {
    setupLog(log)
    val teachNumbers = IntArray(5, { 5+it })
    var structure = if (CNetwork().load("nets/nw.net") != null) getStructure("nets/nw.net") else intArrayOf(6,6,4,4,40,10)
//    fullTrain(500, structure, true, teachNumbers)
    while (true) {
        fullTrain(150, structure, false, teachNumbers)
        rebuild(teachNumbers, 40)
        structure = getStructure("nets/nw.net")
        if (structure.sum() == getStructure("nets/nwx.net").sum()) break
        if (structure.take(4).sum() < 5) break
        fullTrain(500, structure, true, teachNumbers)
        saveAs("nets/nwx.net", "nets/nw${teachNumbers.joinToString("")}_${structure.joinToString("-")}.net")
    }
}

