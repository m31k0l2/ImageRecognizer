import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

class TrainSettings {
    var trainLayers = listOf<Int>()
    var testNumbers = (0..9).toList()
        set(value) {
            field = value
            testBatch = testBatch.filter { it.index in testNumbers }
        }
    var rateCount = 3
    var count = 1
    var initBatchSize = 30
    var addBatchSize = 0
    var isUpdated = false
    var exitIfError = false
    var testBatch = MNIST.buildBatch(1000)
    var populationSize = 80
}

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    CNetwork.teachFromLayer = 4
    val settings = TrainSettings().also {
        it.count = 20
        it.addBatchSize = 30
        it.populationSize = 200
        it.trainLayers = (4..5).toList()
        it.testNumbers = (0..5).toList()
    }
    train(settings)
//    train(TrainSettings().also { it.count = 20; it.initBatchSize = 500; it.isUpdated = true })
//    train(TrainSettings().also { it.count = 10; it.initBatchSize = 500; it.isUpdated = true; it.rateCount = 5 })
}

fun train(settings: TrainSettings) = with(settings) {
    log.info("trainLayers: $trainLayers, rateCount: $rateCount")
    val net = ImageNetEvolution(rateCount)
    net.trainLayers = trainLayers
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50-e)/50.0)
            else -> 0.2
        }
    }
    net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
    var res2 = NetworkIO().load(net.name)?.let { testMedianNet(it, testBatch) } ?: 0.0
    log.info("init result: $res2, testBatchSize: ${testBatch.size}")
    for (i in 1..count) {
        net.evolute(500, populationSize, 10)
        val res = testMedianNet(net.leader!!.nw, testBatch)
        if (res > res2) {
            NetworkIO().save(net.leader!!.nw, net.name)
            log.info("SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            res2 = res
        } else {
            log.warning("NO SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
            if (exitIfError) return
        }
        if (res2 > 0.98) return
        if (addBatchSize > 0) net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
        else if (isUpdated) net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
    }
}