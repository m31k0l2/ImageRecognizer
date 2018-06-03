import java.awt.Toolkit
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

class TrainSettings {
    var trainLayers = (0..4).toList()
        set(value) {
            field = value
            CNetwork.teachFromLayer = value.min() ?: 0
        }
    val initTestBatch = MNIST.buildBatch(500)
    lateinit var testBatch: List<Image>
    var testNumbers: IntArray = IntArray(9, {it})
        set(value) {
            field = value
            testBatch = initTestBatch.filter { it.index in testNumbers }
        }
    var count = 100
    var addBatchSize = 100
    var exitIfError = 1
    var populationSize = 60
    var epochSize = 300
}
val log: Logger = Logger.getLogger("logger")

fun beep() {
    for (i in 1..60) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

fun main(args: Array<String>) {
    setupLog(log)
    val teachNumbers = (0..8).toList().toIntArray()
//    trainNet(teachNumbers, listOf(), intArrayOf(6,6,4,4,40,10))
    retrainNet(teachNumbers)
    var hiddenLayerNeurons = 60
    while (true) {
        val sum = getStructure("nets/nwx.net").sum()
        rebuild(teachNumbers, hiddenLayerNeurons)
        if (sum == getStructure("nets/nw.net").sum()) {
            if (hiddenLayerNeurons == 60) {
                hiddenLayerNeurons = 40
                continue
            }
            else break
        }
        retrainNet(teachNumbers)
    }
    beep()
}

fun retrainNet(teachNumbers: IntArray) {
    val structure = getStructure("nets/nw.net")
    val r1 = trainNet(teachNumbers, listOf(4,5), structure)
    saveAs("nets/nw.net", "nets/nw_back.net")
    val r2 = trainNet(teachNumbers, listOf(), structure)
    if (r2 < 0.98 && r1 > 0.98) {
        saveAs("nets/nw_back.net", "nets/nwx.net")
        saveAs("nets/nw_back.net", "nets/nw.net")
    }
    saveAs("nets/nw.net", "nets/nw${teachNumbers.joinToString("")}_${structure.joinToString("-")}.net")
}

fun getStructure(path: String): IntArray {
    val nw = NetworkIO().load(path) ?: return emptyArray<Int>().toIntArray()
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
    changeStructure("nets/nwx.net", "nets/nw.net", listOf(0,1,2,3), map.map { it.key to it.value.map { it.first } }.toMap(), CNetwork(*list.toIntArray()))
}

fun trainNet(teachNumbers: IntArray, trainLayers: List<Int>, structure: IntArray): Double {
    var r = 0.0
    for (a in listOf(1,2,3,5,7,10,15)) {
        r = train(teachNumbers, trainLayers, a.toDouble(), structure)
        if (r > 0.98) break
    }
    saveAs("nets/nw.net", "nets/nwx.net")
    return r
}

fun rateNet(teachNumbers: IntArray): Double {
    val nw = NetworkIO().load("nets/nw.net") ?: return 0.0
    Neuron.alpha = 15.0
    return testMedianNet(nw, MNIST.buildBatch(500).filter { it.index in teachNumbers }, teachNumbers)
}

fun saveAs(from: String, to: String) {
    val nw = NetworkIO().load(from)!!
    NetworkIO().save(nw, to)
}

fun setupLog(log: Logger) {
    val fh = FileHandler("log.txt")
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
}

fun train(teachNumbers: IntArray, trainLayers: List<Int>, alpha: Double, structure: IntArray): Double {
    class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
        override fun createNet() = CNetwork(*structure)
    }
    val net = ImageNetEvolution()
    val settings = TrainSettings().apply { exitIfError = 2; testNumbers = teachNumbers; epochSize = 150 }
    Network.useSigma = true
    Neuron.alpha = alpha
    settings.trainLayers = trainLayers
    val nw = NetworkIO().load("nets/nw.net")
    var r0 = if (nw == null) 0.0 else testMedianNet(nw, settings.testBatch, teachNumbers)
    log.info("init result: $r0")
    while (true) {
        val r = train(settings, r0, teachNumbers, net)
        log.info("result: $r")
        if (r > 0.98 || r <= r0) break
        r0 = r
    }
    return rateNet(teachNumbers)
}

fun train(settings: TrainSettings, res: Double, teachNumbers: IntArray, net: NetEvolution): Double = with(settings) {
    log.info("trainLayers: $trainLayers")
    log.info("testNumbers: ${testNumbers.joinToString()}")
    log.info("alpha: ${Neuron.alpha}")
    val structure = getStructure("nets/nw.net")
    log.info("structure: ${structure.joinToString()}")
    testBatch.forEach { it.y = null; it.o = null }
    var exitIfError = exitIfError
    net.trainLayers = trainLayers
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
//    net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
    net.batch = testBatch
    var r0 = res
    if (r0 > 0.98) return r0
    for (i in 1..count) {
        net.evolute(epochSize, populationSize, 3)
        val r1 = testMedianNet(net.leader!!.nw, testBatch, teachNumbers)
        if (r1 > r0) {
            NetworkIO().save(net.leader!!.nw, net.name)
            log.warning("ok, $r1")
            r0 = r1
        } else {
            log.info("-")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
            if (--exitIfError == 0) break
        }
        if (r0 > 0.98) break
        if (addBatchSize > 0) net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
    }
    return r0
}