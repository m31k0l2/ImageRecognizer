import java.util.*
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter

val log = Logger.getLogger("logger")
val fh = FileHandler("log.txt")

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val testBatch = MNIST.buildBatch(1000)
    val testNumbers = (0..5).toList()
    for (rateCount in 3..5 step 2) {
        train(testNumbers, listOf(0), rateCount, testBatch)
        train(testNumbers, listOf(1), rateCount, testBatch)
        train(testNumbers, listOf(2), rateCount, testBatch)
        train(testNumbers, listOf(3), rateCount, testBatch)
        train(testNumbers, listOf(4), rateCount, testBatch)
        train(testNumbers, listOf(5), rateCount, testBatch)
        train(testNumbers, listOf(4, 5), rateCount, testBatch)
        train(testNumbers, listOf(0, 4, 5), rateCount, testBatch)
        train(testNumbers, listOf(1, 4, 5), rateCount, testBatch)
        train(testNumbers, listOf(2, 4, 5), rateCount, testBatch)
        train(testNumbers, listOf(3, 4, 5), rateCount, testBatch)
        train(testNumbers, (0..3).toList(), rateCount, testBatch)
        train(testNumbers, emptyList(), rateCount, testBatch)
    }
}

private fun train(testNumbers: List<Int>, trainLayers: List<Int>, rateCount: Int, testBatch: List<Image>) {
    println("trainLayers: $trainLayers")
    log.info("trainLayers: $trainLayers, rateCount: $rateCount")
    val net = ImageNetEvolution(rateCount)
    net.trainLayers = trainLayers
    val addBatchSize = 200
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 50 -> ((50-e)/50.0)
            else -> 0.2*Random().nextDouble()
        }
    }
    net.batch = MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }
    var res2 = NetworkIO().load(net.name)?.let { testMedianNet(it, testBatch) } ?: 0.0
    for (i in 1..10) {
        println("$i) ${net.batch.size}")
        net.evolute(150, 160, 10)
        val res = testMedianNet(net.leader!!.nw, testBatch)
        if (res > res2) {
            NetworkIO().save(net.leader!!.nw, net.name)
            println("SAVE")
            log.info("SAVE")
            res2 = res
        } else {
            println("NO SAVE")
            log.info("NO SAVE")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
            break
        }
        net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
    }
}