import java.awt.Toolkit
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

class TrainSettings {
    var trainLayers = (0..3).toList()
        set(value) {
            field = value
            CNetwork.teachFromLayer = value.min() ?: 0
        }
    val initTestBatch = MNIST.buildBatch(500)
    var testNumbers = (0..9).toList()
        set(value) {
            field = value
            testBatch = initTestBatch.filter { it.index in testNumbers }
        }
    var count = 100
    var initBatchSize = 500
    var addBatchSize = 200
    var isUpdated = false
    var exitIfError = 1
    var testBatch = initTestBatch
    var populationSize = 60
    var epochSize = 200
    var dropout = 0.0
    var exitIfBest = true
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
    val settings = TrainSettings().apply { exitIfError = 1 }
    var r0 = 0.0
    while (true) {
        log.info("init result -> $r0")
        for (i in 5 downTo 0) {
            var drop = 0.0
            var exitIfErr = 1
            while (true) {
                train(settings.apply { trainLayers = listOf(i); dropout=drop; exitIfError = exitIfErr })
                val r1 = testMedianNet(NetworkIO().load("nets/nw.net")!!, settings.initTestBatch)
                if ((r1*10000).toInt() > (r0*10000).toInt()) {
                    r0 = r1
                    break
                }
                drop += 0.001
                if (drop == 0.2) drop = 0.0
                exitIfErr = 2
            }
        }
        val r1 = train(settings.apply { trainLayers = listOf() })
        if (r1 > r0) r0 = r1
        log.info("new result -> $r0")
        if (r0 == 0.98) break
//        settings.testBatch = settings.testBatch.takeLast(settings.testBatch.size/2).union(MNIST.buildBatch(settings.initBatchSize).filter { it.index in settings.testNumbers }).toList()
//        r0 = testMedianNet(NetworkIO().load("nets/nw.net")!!, settings.testBatch)
    }
//    trainGroup()
    beep()
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

fun train(settings: TrainSettings): Double = with(settings) {
    log.info("trainLayers: $trainLayers")
    log.info("testNumbers: $testNumbers")
    var res0 = 0.0
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
    if (dropout != 0.0) {
        log.info("Dropout")
        var nw = NetworkIO().load(net.name)!!
        res0 = testMedianNet(nw, testBatch)
        NetworkIO().save(nw, net.name + ".back")
        nw = nw.dropout(trainLayers, dropout)
        NetworkIO().save(nw, net.name)
        log.info("before res $res0")
    }
    var res2 = NetworkIO().load(net.name)?.let { testMedianNet(it, testBatch) } ?: 0.0
    if (res2 > 0.98) return res2
    log.info("init result: $res2, testBatchSize: ${testBatch.size}")
    for (i in 1..count) {
        net.evolute(epochSize, populationSize, 3)
        val res = testMedianNet(net.leader!!.nw, testBatch)
        if (res > res2) {
            NetworkIO().save(net.leader!!.nw, net.name)
            log.warning("$i) SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            res2 = res
            if (settings.exitIfBest) break
        } else {
            log.info("$i) NO SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
            if (--exitIfError == 0) break
        }
        if (res2 > 0.98) break
        if (addBatchSize > 0) net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
        else if (isUpdated) net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
    }
    if (dropout != 0.0) {
        val nw =  NetworkIO().load("${net.name}.back")!!
        if (res0 > res2) {
            NetworkIO().save(nw, net.name)
            log.info("$res0 > $res2; back load")
        }
    }
    return res2
}