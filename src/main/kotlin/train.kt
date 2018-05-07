import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

class TrainSettings {
    var trainLayers = (0..5).toList()
        set(value) {
            field = value
            CNetwork.teachFromLayer = value.min() ?: 0
        }
    val initTestBatch = MNIST.buildBatch(1000)
    var testNumbers = (0..9).toList()
        set(value) {
            field = value
            testBatch = initTestBatch.filter { it.index in testNumbers }
        }
    var count = 10
    var initBatchSize = 30
    var addBatchSize = 30
    var isUpdated = false
    var exitIfError = 10
    var testBatch = initTestBatch
    var populationSize = 80
    var epochSize = 500
    var dropout = 0.0
}

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val settings = TrainSettings().apply {
        initBatchSize = 500
        addBatchSize = 250
        count = 10
        testNumbers = listOf(0, 1, 2, 3, 4, 5)
    }
//    NetworkIO().load("nets/nw.net")?.dropout(listOf(0, 1, 2, 3, 4, 5), 0.2)?.let {NetworkIO().save(it, "nets/nw.net") }
//    (3 downTo 2).forEach { trainLayer(4, settings, 80, it, 200, 0.0 ) }
    train(settings.apply { populationSize = 60; trainLayers = listOf(3); epochSize = 150; exitIfError=2; dropout=0.0 })
}

private fun trainLayer(toClassNumber: Int, settings: TrainSettings, popSize: Int, lNum: Int, epSize: Int, drop: Double=0.0) {
    var nw = NetworkIO().load("nets/nw.net")
    val r0 = nw?.let {
//        NetworkIO().save(it, "nets/nw.net.back.layer")
        testMedianNet(it, settings.initTestBatch)
    } ?: 0.0
    log.info("r0 = $r0")
    val numbers = (0..toClassNumber).shuffled()
    (1..toClassNumber).forEach {
        train(settings.apply { populationSize = popSize; trainLayers = listOf(lNum); epochSize = epSize; exitIfError = 1; dropout = if (it == 1 ) drop else 0.0; testNumbers = (0..it).map { numbers[it] } })
    }
    nw = NetworkIO().load("nets/nw.net")!!
    val r1 = testMedianNet(nw, settings.initTestBatch)
    if (r1 < r0) {
//        nw = NetworkIO().load("nets/nw.net.back.layer")!!
//        NetworkIO().save(nw, "nets/nw.net")
        log.info("roll back ($r1 < $r0")
    } else {
        log.info("$r1 > $r0")
    }
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
            log.info("$i) SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            res2 = res
        } else {
            log.warning("$i) NO SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
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