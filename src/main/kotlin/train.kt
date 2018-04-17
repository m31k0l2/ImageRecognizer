import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val testNumbers = (0..5).toList()
    val testBatch = MNIST.buildBatch(1000).filter { it.index in testNumbers }
    val trainLayers: List<Int> = (2..4).toList()
    train(20, testNumbers, trainLayers, 3, testBatch, 30, 30)
    train(20, testNumbers, trainLayers, 3, testBatch, 500, 0, true)
    train(10, testNumbers, trainLayers, 5, testBatch, 500, 0, true)
}

private fun train(count: Int, testNumbers: List<Int>, trainLayers: List<Int>, rateCount: Int, testBatch: List<Image>, initBatchSize: Int, addBatchSize: Int=0, isUpdated: Boolean = false, exitIfError: Boolean=false) {
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
    for (i in 1..count) {
        net.evolute(500, 60, 10)
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
        if (res > 0.98) return
        if (addBatchSize > 0) net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
        else if (isUpdated) net.batch = MNIST.buildBatch(initBatchSize).filter { it.index in testNumbers }
    }
}