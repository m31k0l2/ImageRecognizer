import java.awt.Toolkit
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

class TrainSettings {
    var trainLayers = (0..4).toList()
        set(value) {
            field = value
            CNetwork.teachFromLayer = value.min() ?: 0
        }
    val initTestBatch = MNIST.buildBatch(500)
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
    var epochSize = 200
}

fun beep() {
    for (i in 1..60) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val settings = TrainSettings().apply { exitIfError = 1; testNumbers = (0..5).toList() }
    val nw = NetworkIO().load("nets/nw.net")
    var r0 = if (nw == null) 0.0 else testMedianNet(nw, settings.testBatch)
    val lastLayerIndex = ImageNetEvolution().createNet().layers.count() - 1
    while (true) {
        log.info("init result -> $r0")
        for (i in lastLayerIndex downTo 0) {
            val r1 = train(settings.apply { trainLayers = listOf(i) }, r0)
            if (r1 > r0) r0 = r1
            else if (i != lastLayerIndex) train(settings.apply { trainLayers = (i..lastLayerIndex).toList() }, r0)
            if (r0 > 0.98) return
        }
    }
//    trainGroup()
//    beep()
}

fun trainGroup() {
    val net = ImageNetEvolution()
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50 - e) / 50.0)
            else -> 0.2
        }
    }
    net.batch = MNIST.buildBatch(1000).filter { it.index in (0..3) }
    println(net.batch.size)
    val population = List(2, { Individual(NetworkIO().load("nets/nw$it.net")!!) })
    net.evolute(500, population)
    NetworkIO().save(net.leader!!.nw, net.name)
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
    net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
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