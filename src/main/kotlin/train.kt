import java.io.PrintWriter

fun main(args: Array<String>) {
    val log = PrintWriter("log.txt")
    val testNumbers = (0..8).toList()
    val net = ImageNetEvolution(3)
    net.trainLayers = listOf(6)
    val addBatchSize = 150
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 20 -> 0.0
            e < 50 -> 1.0
            e < 100 -> 0.5
            else -> 0.2
        }
    }
    net.batch = MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }
    var res2 = testMedianNet(NetworkIO().load(net.name)!!, net.batch)
    for (i in 1..10) {
        println("$i) ${net.batch.size}")
        net.evolute(1000, 80, 10)
        val res = testMedianNet(net.leader!!.nw, net.batch)
        if (res > res2) {
            res2 = res
            NetworkIO().save(net.leader!!.nw, net.name)
            println("SAVE")
            log.println("$i) save $res2")
        } else {
            println("NO SAVE")
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
            log.println("$i) no save $res2")
            return
        }
        log.close()
        net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
    }
}