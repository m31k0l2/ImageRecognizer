import java.awt.Toolkit
import java.io.File
import java.util.logging.*

val log = Logger.getLogger("logger")!!
var testBatch = MNIST.buildBatch(500)
var rateCount = 5

fun beep() {
    for (i in 1..10) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

fun getStructure(path: String): IntArray? {
    val nw = CNetwork().load(path) ?: return null
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

fun Network.convLayer(id: String, neuronCount: Int, divider: MatrixDivider, pooler: Pooler?=null) {

    val layer = CNNLayer(id, divider, pooler, neuronCount)
    layers.add(layer)
}

fun Network.fullConnectedLayer(id: String, neuronCount: Int) {
    layers.add(FullConnectedLayer(id, neuronCount))
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

fun NewEvolution.run(numbers: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, trainFinal: Boolean): Network {
//    testBatch = MNIST.buildBatch(500).union(MNIST.buildBatch(500, MNIST.errorPath))
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
//    val testBatch = MNIST.buildBatch(50)
    batch = testBatch.filter { it.index in numbers }
    if (trainFinal) {
        val nw = CNetwork().load("nets/nw.net")!! as CNetwork
        batch.forEach {
            it.y = nw.activateLayers(it, 15.0)
        }
        trainLayers = listOf(nw.layers.size-2, nw.layers.size-1)
    } else if (trainFullConnectedLayers) {
        val nw = CNetwork().load("nets/nw.net")!!
        batch.forEach {
            it.o = nw.activateConvLayers(nw.layers.filter { it is CNNLayer }.map { it as CNNLayer }, it)
        }
        trainLayers = listOf(4, 5)
    }
    evolute(time, popSize, alpha)
    return leader!!.nw
}

fun buildNetwork(vararg structure: Int)= network {
    convLayer("0", structure[0], CNetwork.cnnDividers[0], Pooler(CNetwork.poolDividers[0]!!))
    convLayer("0", structure[1], CNetwork.cnnDividers[1], Pooler(CNetwork.poolDividers[1]!!))
    convLayer("0", structure[2], CNetwork.cnnDividers[2])
    convLayer("0", structure[3], CNetwork.cnnDividers[3])
    for (i in 4 until structure.size) {
        fullConnectedLayer("0", structure[i])
    }
}

fun buildMultiNetwork(count: Int, vararg structure: Int)= network {
    for (i in 0 until count) {
        convLayer("$i", structure[0+i*5], CNetwork.cnnDividers[0], Pooler(CNetwork.poolDividers[0]!!))
        convLayer("$i", structure[1+i*5], CNetwork.cnnDividers[1], Pooler(CNetwork.poolDividers[1]!!))
        convLayer("$i", structure[2+i*5], CNetwork.cnnDividers[2])
        convLayer("$i", structure[3+i*5], CNetwork.cnnDividers[3])
        fullConnectedLayer("$i", structure[4+i*5])
    }
    fullConnectedLayer("final", structure[structure.size-2])
    fullConnectedLayer("final", structure.last())
}

fun evolute(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, numbers: IntArray, trainFinal: Boolean=false): Network {
    return evolution(time, 60) {
        buildNetwork(*structure)
    }.run(numbers, trainFullConnectedLayers, alpha, trainFinal)
}

fun Network.rate(vararg numbers: Int): Double {
    return testMedianNet(this, testBatch, numbers)
}

fun train(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray, alpha: Double, trainFinalLayer: Boolean=false): Pair<Double, Double> {
    saveAs("nets/nw.net", "nets/nw_back.net")
    var rate = 0.0
    log.info("alpha: $alpha")
    log.info("structure: ${getStructure("nets/nw.net")?.toList()}")
    log.info("trainFullConnectedLayers: $trainFullConnectedLayers")
    var a = alpha
    while (true) {
        val nw = evolute(time, structure, trainFullConnectedLayers, a, teachNumbers, trainFinalLayer)
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

fun fullTrain(time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray, trainFinalLayer: Boolean=false): Double {
    var alpha = 1.0.takeIf { trainFullConnectedLayers } ?: 15.0
    var r1 = 0.0
    while (true) {
        val (a, r2) = train(time, structure, trainFullConnectedLayers, teachNumbers, alpha, trainFinalLayer)
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
    train(100, intArrayOf(7, 8, 9), intArrayOf(6), 200, 600, false)
//    rateCount = 100
//    fullTrain(1500, getStructure("nets/nw.net")!!, true, intArrayOf(7,8,9), true)
}

fun train(rate: Int, trainNumbers: IntArray, hiddenLayerNeurons: IntArray, time1: Int, time2: Int, doRebuild: Boolean) {
    rateCount = rate
    val structure = getStructure("nets/nw.net") ?: intArrayOf(6,6,6,6,40,10,4)
    fullTrain(time1, structure, false, trainNumbers)
    val r = fullTrain(time2, structure, true, trainNumbers)
    File("nets/nw_back.net").delete()
    if (doRebuild && r > 0.7) {
        rebuild(trainNumbers, hiddenLayerNeurons)
        fullTrain(time2, getStructure("nets/nw.net")!!, true, trainNumbers)
    } else if (doRebuild) {
        train(2*rate, trainNumbers, hiddenLayerNeurons, 2*time1, 2*time2, doRebuild)
    }
}

