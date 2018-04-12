import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val testBatch = MNIST.buildBatch(1000)
    val testNumbers = (0..4).toList()
    train(testNumbers, listOf(), 3, testBatch, 500, false, false)
}

private fun train(testNumbers: List<Int>, trainLayers: List<Int>, rateCount: Int, testBatch: List<Image>, addBatchSize: Int, isAdder: Boolean, isUpdated: Boolean = false) {
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
    net.batch = MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }
    var res2 = NetworkIO().load(net.name)?.let { testMedianNet(it, testBatch) } ?: 0.0
    for (i in 1..10) {
        net.evolute(150, 60, 10)
        val res = testMedianNet(net.leader!!.nw, testBatch)
        if (res > res2) {
            NetworkIO().save(net.leader!!.nw, net.name)
            log.info("SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            res2 = res
        } else {
            log.warning("NO SAVE, batch: ${net.batch.size}, res = $res [old $res2]")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
        }
        if (isAdder) net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
        else if (isUpdated) net.batch = MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }
    }
}