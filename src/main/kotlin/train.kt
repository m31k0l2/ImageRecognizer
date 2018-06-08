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
        list = mutableListOf(2, 2, 2, 2)
    }
    list.add(hiddenLayerNeurons)
    list.add(teachNumbers.size)
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
    batch.forEach {
        it.index = numbers.indexOf(it.index)
    }
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

fun train(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray, alpha: Double): Pair<Double, Double> {
    saveAs("nets/nw.net", "nets/nw_back.net")
    var rate = 0.0
    log.info("alpha: $alpha")
    log.info("structure: ${getStructure("nets/nw.net").toList()}")
    log.info("trainFullConnectedLayers: $trainFullConnectedLayers")
    var a = alpha
    while (true) {
        val nw = evolute(time, structure, trainFullConnectedLayers, a, teachNumbers)
        val curRate = nw.rate(*teachNumbers)
        log.info("rate: $curRate")
        if (curRate > rate) {
            nw.save("nets/nw.net")
            log.info("SAVE $a")
            rate = curRate
        } else break
        a += 1.0
    }
    CNetwork().load("nets/nw_back.net")?.let {
        val r = testMedianNet(it, testBatch, teachNumbers)
        if (r > rate) saveAs("nets/nw_back.net", "nets/nw.net")
    }
    return a to rate
}

fun fullTrain(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray) {
    var alpha = 1.0
    var r1 = 0.0
    while (true) {
        val (a, r2) = train(time, structure, trainFullConnectedLayers, teachNumbers, alpha)
        log.info("\r\nresult $r1 -> $r2")
        if (r2 > r1) r1 = r2
        else if (a > 10.0) break
        alpha = a
    }
    saveAs("nets/nw.net", "nets/nwx.net")
    return
}

fun main(args: Array<String>) {
    setupLog(log)
    teach(true, 4,5,6)
}

fun trainAndRebuild(teachNumbers: IntArray, trainFullConnectedLayers: Boolean) {
    val structure = if (CNetwork().load("nets/nw.net") != null) getStructure("nets/nw.net") else intArrayOf(4,4,4,4,40,3)
    fullTrain(150, structure, trainFullConnectedLayers, teachNumbers)
    rebuild(teachNumbers, 40)
    if (!testStructure()) saveAs("nets/nwx.net", "nets/nw.net")
}

fun testStructure(): Boolean {
    val structure = getStructure("nets/nw.net")
    if (structure.sum() == getStructure("nets/nwx.net").sum()) return false
    return (0..3).map { structure[it] < 2 }.reduce { acc, b -> acc && b }
}

fun saveStructure(teachNumbers: IntArray) {
    saveAs("nets/nwx.net", "nets/nw${teachNumbers.joinToString("")}_${getStructure("nets/nwx.net").joinToString("-")}.net")
}

fun teach(pretrain: Boolean, vararg teachNumbers: Int) {
    if (pretrain) {
        while (true) {
            trainAndRebuild(teachNumbers, true)
            saveStructure(teachNumbers)
            if (!testStructure()) break
        }
    }
    while (true) {
        trainAndRebuild(teachNumbers, false)
        if (!testStructure()) break
        trainAndRebuild(teachNumbers, true)
        if (!testStructure()) break
        saveStructure(teachNumbers)
    }
    if (!getStructure("nets/nw.net").contentEquals(intArrayOf(2,2,2,2,40,3))) {
        changeStructure("nets/nwx.net", "nets/nw.net", (0..5).toList(), emptyMap(), buildNetwork(2, 2, 2, 2, 40, 3))
        fullTrain(150, getStructure("nets/nw.net"), false, teachNumbers)
    }
    saveStructure(teachNumbers)
}

