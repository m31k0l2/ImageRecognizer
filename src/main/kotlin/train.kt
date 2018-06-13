import java.awt.Toolkit
import java.io.File
import java.util.logging.*

val log = Logger.getLogger("logger")!!
var testBatch = MNIST.buildBatch(1000)
var rateCount = 5

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

fun rebuild(teachNumbers: IntArray, hiddenLayerNeurons: IntArray) {
    val map = clean(teachNumbers)
    val list = mutableListOf<Int>()
    for (i in 0..3) {
        map[i]?.let { list.add(it.size) } ?: list.add(1)
    }
    if (list.sum() < 5) return
    hiddenLayerNeurons.forEach {
        list.add(it)
    }
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

abstract class NewEvolution(val time: Int, val popSize: Int) : NetEvolution(0.2, rateCount)

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
//    testBatch = MNIST.buildBatch(500).union(MNIST.buildBatch(500, MNIST.errorPath))
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
//    val testBatch = MNIST.buildBatch(50)
    batch = testBatch.filter { it.index in numbers }
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
    for (i in 4 until structure.size) {
        fullConnectedLayer(structure[i])
    }
}

fun evolute(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, numbers: IntArray): Network {
    return evolution(time, 60) {
        buildNetwork(*structure)
    }.run(numbers, trainFullConnectedLayers, alpha)
}

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
        if (trainFullConnectedLayers) a += 1.0
    }
    CNetwork().load("nets/nw_back.net")?.let {
//        val testBatch = MNIST.buildBatch(50)
        val r = testMedianNet(it, testBatch, teachNumbers)
        if (r > rate) saveAs("nets/nw_back.net", "nets/nw.net")
    }
    return a to rate
}

fun fullTrain(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray): Double {
    var alpha = 1.0.takeIf { trainFullConnectedLayers } ?: 15.0
    var r1 = 0.0
    while (true) {
        val (a, r2) = train(time, structure, trainFullConnectedLayers, teachNumbers, alpha)
        log.info("\r\nresult $r1 -> $r2")
        if (r2 > r1) r1 = r2
        else if (a > 10.0) break
        alpha = a
    }
    saveAs("nets/nw.net", "nets/nwx.net")
    return r1
}

fun main(args: Array<String>) {
    setupLog(log)
////    (3..11).forEach {
//        teachOne(4, 7, 8, 9)
////    }
    rateCount = 17
    val trainNumbers = intArrayOf(7,8,9)
//    fullTrain(100, intArrayOf(6,6,6,6,40,10,4), false, intArrayOf(7,8,9))
    fullTrain(1500, getStructure("nets/nw.net"), false, trainNumbers)
//    fullTrain(3000, getStructure("nets/nw.net"), true, trainNumbers)
//    rebuild(trainNumbers, intArrayOf(4))
    File("nets/nw_back.net").delete()
//    fullTrain(150, intArrayOf(6,6,6,6,40,10,3), true, intArrayOf(3,4,8))
//    fullTrain(150, intArrayOf(6,6,6,6,40,10,3), true, intArrayOf(5,7,9))
}

fun teachOne(id: Int, vararg teachNumbers: Int) {
    val dir = teachNumbers.map { "$it" }.reduce { acc, a -> "$acc$a" }
    val path = "nets/$dir/nw$id.net"
    saveAs(path, "nets/nw.net")
    do {
        teach(7, 8, 9)
        saveAs("nets/nw.net", path)
        saveStructure(teachNumbers)
        val n1 = getStructure("nets/nw.net").sum()
        rebuild(teachNumbers, intArrayOf(30, 10))
        val n2 = getStructure("nets/nw.net").sum()
        File("nets/nw_back.net").delete()
        File("nets/nwx.net").delete()
    } while (n1 != n2)
    File("nets/nw.net").delete()
}

fun saveStructure(teachNumbers: IntArray) {
    saveAs("nets/nwx.net", "nets/nw${teachNumbers.joinToString("")}_${getStructure("nets/nwx.net").joinToString("-")}.net")
}

fun teach(vararg teachNumbers: Int): Double {
    val structure = intArrayOf(6,6,4,4,30,10,3)
    fullTrain(200, structure, true, teachNumbers)
    val r = fullTrain(150, structure, false, teachNumbers)
    if (r > 0.8) return r
    return fullTrain(200, structure, true, teachNumbers)
}

