import java.awt.Toolkit
import java.io.File
import java.util.logging.*
import kotlin.math.max

val log = Logger.getLogger("logger")!!
var testBatch = MNIST.buildBatch(500)
var rateCount = 100

fun beep() {
    Toolkit.getDefaultToolkit().beep()
    Thread.sleep(1000)
}

fun getStructure(path: String): IntArray? {
    val nw = CNetwork().load(path) ?: return null
    return nw.layers.map { it.neurons.size }.toIntArray()
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

fun NewEvolution.run(trainFullConnectedLayers: Boolean, alpha: Double, trainFinal: Boolean): Network? {
//    testBatch = MNIST.buildBatch(500).union(MNIST.buildBatch(500, MNIST.errorPath))
    mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
//    val testBatch = MNIST.buildBatch(50)
    batch = testBatch.toList()
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
    return leader?.nw
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

fun evolute(number: Int, time: Int, structure: IntArray, trainFullConnectedLayers: Boolean, alpha: Double, trainFinal: Boolean=false): Network? {
    return evolution(number, time, 60) {
        buildNetwork(*structure)
    }.run(trainFullConnectedLayers, alpha, trainFinal)
}

fun Network.rate(number: Int, batch: Set<Image>): Double {
    return testMedianNet(number, this, batch)
}

fun train(number: Int, pos: Int, time: Int) {
    testBatch = MNIST.buildBatch(1500)
    var nw: Network
//    val structure = intArrayOf(5, 5, 5, 5, 120, 40, 1)
    val structure = intArrayOf(3, 3, 4, 4, 60, 40, 1)
    var rate = 0.0
    for (alpha in 1..15 step 2) {
        var curRate = 0.0
        var counter = 0
        do {
            rate = max(curRate, rate)
            log.info("> $alpha) $rate")
            nw = evolute(number, time + (alpha/5)*50-counter*25, structure, alpha > 5, alpha.toDouble()) ?: buildNetwork(*structure)
            nw = nw.rebuild(number, testBatch)
            curRate = nw.rate(number, testBatch)
            if (rate < curRate) nw.save("nets/nw.net")
            counter++
        } while (rate < curRate)
        if (curRate > 0.99) break
    }
    saveAs("nets/nw.net", "nets/nw$number$pos.net")
    File("nets/nw.net").delete()
}

fun main(args: Array<String>) {
    setupLog(log)
    for (i in 10..15) {
        log.info("------------- $i -----------------")
        train(2, i, 100)
    }
}

