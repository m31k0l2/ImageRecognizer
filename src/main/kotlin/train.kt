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
    val initTestBatch = MNIST.buildBatch(100)
    lateinit var testBatch: List<Image>
    var testNumbers: List<Int> = (0..9).toList()
        set(value) {
            field = value
            testBatch = initTestBatch.filter { it.index in testNumbers }
        }
    var count = 100
    var initBatchSize = 500
    var addBatchSize = 100
    var exitIfError = 1
    var populationSize = 60
    var epochSize = 300
}
val log = Logger.getLogger("logger")
val teachNumbers: List<Int> = listOf(0,1,2)

fun beep() {
    for (i in 1..60) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

val train_layers: List<Int> = listOf(1,2,3,4,5)
val epoch_size = 200

fun main(args: Array<String>) {
    val fh = FileHandler("log.txt")
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val settings = TrainSettings().apply { exitIfError = 1; testNumbers = teachNumbers; epochSize = epoch_size }
    Network.useSigma = true
    Neuron.alpha = 2.0
    settings.trainLayers = train_layers
    val nw = NetworkIO().load("nets/nw.net")
    var r0 = if (nw == null) 0.0 else testMedianNet(nw, settings.testBatch)
    log.info("init result: $r0")
    while (true) {
        val r = train(settings, r0)
        log.info("result: $r")
        if (r > 0.95 || r <= r0) break
        r0 = r
    }
}

fun train(settings: TrainSettings, res: Double): Double = with(settings) {
    log.info("trainLayers: $trainLayers")
    log.info("testNumbers: $testNumbers")
    testBatch.forEach { it.y = null; it.o = null }
    var exitIfError = exitIfError
    val net = ImageNetEvolution()
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
        val r1 = testMedianNet(net.leader!!.nw, testBatch)
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