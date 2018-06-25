import java.awt.Toolkit
import java.io.File
import java.util.logging.*

val log = Logger.getLogger("logger")!!
var testBatch = MNIST.buildBatch(100)
var rateCount = 5

fun beep() {
    Toolkit.getDefaultToolkit().beep()
    Thread.sleep(1000)
}

fun getStructure(path: String): IntArray? {
    val nw = CNetwork().load(path) ?: return null
    return nw.layers.map { it.neurons.size }.toIntArray()
}

fun rebuild(number: Int, teachNumbers: IntArray, hiddenLayerNeurons: IntArray) {
    val map = clean(number, teachNumbers)
    val list = mutableListOf<Int>()
    for (i in 0..3) {
        map[i]?.let { list.add(it.size) } ?: list.add(1)
    }
    if (list.sum() < 5) return
    hiddenLayerNeurons.forEach {
        list.add(it)
    }
    list.add(1)
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

abstract class NewEvolution(number: Int, val time: Int, val popSize: Int) : NetEvolution(number, 0.2, rateCount)

fun evolution(number: Int, time: Int, population: Int, init: NetEvolution.() -> Network): NewEvolution {
    class Evolution : NewEvolution(number, time, population) {
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

fun buildTotalMultiNetwork(count1: Int, count2: Int, vararg structure: Int)= network {
    for (i in 0 until count1) {
        convLayer("$i", structure[0+i*5], CNetwork.cnnDividers[0], Pooler(CNetwork.poolDividers[0]!!))
        convLayer("$i", structure[1+i*5], CNetwork.cnnDividers[1], Pooler(CNetwork.poolDividers[1]!!))
        convLayer("$i", structure[2+i*5], CNetwork.cnnDividers[2])
        convLayer("$i", structure[3+i*5], CNetwork.cnnDividers[3])
        fullConnectedLayer("$i", structure[4+i*5])
    }
    for (i in 0 until count2) {
        fullConnectedLayer("connected", structure[structure.size-1 - i])
    }
    fullConnectedLayer("final", structure.last())
}

fun evolute(number: Int, time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, numbers: IntArray, trainFinal: Boolean=false): Network {
    return evolution(number, time, 60) {
        buildNetwork(*structure)
    }.run(numbers, trainFullConnectedLayers, alpha, trainFinal)
}

fun Network.rate(number: Int): Double {
    return testMedianNet(number, this, testBatch)
}

fun train(number: Int, time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray, alpha: Double, trainFinalLayer: Boolean=false): Pair<Double, Double> {
    saveAs("nets/nw.net", "nets/nw_back.net")
    var rate = 0.0
    log.info("alpha: $alpha")
    log.info("structure: ${getStructure("nets/nw.net")?.toList()}")
    log.info("trainFullConnectedLayers: $trainFullConnectedLayers")
    var a = alpha
    while (true) {
        val nw = evolute(number, time, structure, trainFullConnectedLayers, a, teachNumbers, trainFinalLayer)
        val curRate = nw.rate(number)
        log.info("rate: $curRate")
        if (curRate > rate) {
            nw.save("nets/nw.net")
            log.info("SAVE $a")
            rate = curRate
        } else break
        if (trainFullConnectedLayers) a += 1.0
    }

    CNetwork().load("nets/nw_back.net")?.let {
        if (it.rate(number) > rate) saveAs("nets/nw_back.net", "nets/nw.net")
    }
    return a to rate
}

fun fullTrain(number: Int, time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, teachNumbers: IntArray, trainFinalLayer: Boolean=false): Double {
    var alpha = 1.0.takeIf { trainFullConnectedLayers } ?: 15.0
    var r1 = 0.0
    while (true) {
        val (a, r2) = train(number, time, structure, trainFullConnectedLayers, teachNumbers, alpha, trainFinalLayer)
        log.info("\r\nresult $r1 -> $r2")
        if (r2 > r1) r1 = r2
        else if (a > 10.0) break
        if (r1 > 0.99) return r1
        alpha = a
    }
    saveAs("nets/nw.net", "nets/nwx.net")
    return r1
}

val number = 9
fun main(args: Array<String>) {
    val teachNumbers = (0..9).toList().toIntArray()
    setupLog(log)
    train(number,5, teachNumbers, intArrayOf(30), 100, 300, true, 0.3)
    testBatch = MNIST.buildBatch(500)
    train(number, 1000, teachNumbers, intArrayOf(30), 200, 600, false, 0.3)
    beep()
//    rateCount = 500
//    fullTrain(1500, getStructure("nets/nw.net")!!, true, intArrayOf(7, 8, 9), true)
//    rateCount = 100
//    for (i in 1..100) {
//        testBatch = testBatch.union(MNIST.buildBatch(100))
//        log.info("$i -> testBatch: ${testBatch.size}")
//        fullTrain(1500, getStructure("nets/nw.net")!!, true, intArrayOf(7, 8, 9), true)
//    }
}

fun train(number: Int, rate: Int, trainNumbers: IntArray, hiddenLayerNeurons: IntArray, time1: Int, time2: Int, doRebuild: Boolean, limit: Double) {
    rateCount = rate
    val structure = getStructure("nets/nw.net") ?: intArrayOf(6,6,6,6,30,1)
    var r = CNetwork().load("nets/nw.net")?.let { fullTrain(number, time2, structure, true, trainNumbers) } ?: 0.0
    if (r < limit) r = fullTrain(number, time1, structure, false, trainNumbers)
    File("nets/nw_back.net").delete()
    if (doRebuild && r > limit) {
        rebuild(number, trainNumbers, hiddenLayerNeurons)
        fullTrain(number, time2, getStructure("nets/nw.net")!!, true, trainNumbers)
    } else if (doRebuild) {
        train(number, rate+2, trainNumbers, hiddenLayerNeurons, time1, time2, doRebuild, limit-0.1)
    }
}

